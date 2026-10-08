import io
import subprocess
import threading

import numpy as np
from numpy import random
import cv2 as cv
from .utils import probability

from ..constants import JPEG_SUBSAMPLING, VIDEO_SUBSAMPLING
from ..utils.random import safe_randint
from ..utils.registry import register_class
import logging
import video_backend

try:
    import av
    _HAS_PYAV = True
except ImportError:
    _HAS_PYAV = False

_CHROMA_ORDER = ["yuv444p", "yuv422p", "yuv420p"]
_FFMPEG_TIMEOUT_S = 30  # one frame through the encoder and the decoder
_FALLBACKS_LOGGED = set()  # (ffmpeg path, encoder) already reported as running on PyAV

# algorithm -> (encoder, PyAV container, the raw elementary stream piped between the
# two ffmpeg processes as (muxer, demuxer))
_VIDEO_FORMATS = {
    "h264": ("libx264", "mp4", ("h264", "h264")),
    "hevc": ("libx265", "mp4", ("hevc", "hevc")),
    "mpeg2": ("mpeg2video", "mpegts", ("mpeg2video", "mpegvideo")),
    "mpeg4": ("mpeg4", "mp4", ("m4v", "m4v")),
    "vp9": ("libvpx-vp9", "webm", ("ivf", "ivf")),
}


def _video_options(algorithm: str, quality: int) -> dict:
    """Encoder options, the same for PyAV and the ffmpeg CLI (as -key value)."""
    q = str(quality)
    if algorithm == "h264":
        return {"preset": "ultrafast", "crf": q}
    if algorithm == "hevc":
        return {"preset": "ultrafast", "crf": q, "x265-params": "log-level=0"}
    if algorithm == "vp9":
        return {"cpu-used": "8", "crf": q, "b:v": "0", "row-mt": "1"}
    return {"qscale:v": q, "qmax": q, "qmin": q}  # mpeg2, mpeg4: fixed quantizer


def _nearest_chroma(sampling: str, supported) -> str:
    """Step down to the nearest chroma format the encoder supports (mpeg2video
    has no 4:4:4, mpeg4 only 4:2:0); unchanged when the list is empty."""
    if supported and sampling not in supported:
        fallback = _CHROMA_ORDER[_CHROMA_ORDER.index(sampling):]
        sampling = next(f for f in fallback if f in supported)
    return sampling


def _feed(proc: subprocess.Popen, data: bytes, errors: list) -> None:
    """Write data to proc's stdin and close it, then collect proc's stderr."""
    try:
        proc.stdin.write(data)
        proc.stdin.close()
    except OSError:
        pass  # the process exited early; its exit code and stderr say why
    errors.append(proc.stderr.read())


def _failure(stage: str, returncode: int, stderr: bytes) -> str:
    """ffmpeg's own message, its first line on the first line."""
    if returncode >= 2**31:  # Windows reports exit codes unsigned
        returncode -= 2**32
    text = stderr.decode("utf-8", "replace").strip() or "no message"
    return f"ffmpeg {stage} failed (exit {returncode}): {text}"


def _encode_decode(encode: list, decode: list, data: bytes, encoder_name: str) -> bytes:
    """Pipe data through two ffmpeg processes, `encode` feeding `decode`
    directly, and return what the decoder writes."""
    with subprocess.Popen(
        encode, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=video_backend.POPEN_FLAGS,
    ) as encoder:
        try:
            decoder = subprocess.Popen(
                decode, stdin=encoder.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=video_backend.POPEN_FLAGS,
            )
        except OSError:
            encoder.kill()
            raise
        with decoder:
            encoder.stdout.close()  # the decoder holds the read end now
            errors = []
            feeder = threading.Thread(target=_feed, args=(encoder, data, errors), daemon=True)
            feeder.start()
            try:
                output, decode_errors = decoder.communicate(timeout=_FFMPEG_TIMEOUT_S)
                encoder.wait(timeout=_FFMPEG_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                encoder.kill()
                decoder.kill()
                decoder.communicate()
                raise RuntimeError(
                    f"ffmpeg {encoder_name} did not finish within {_FFMPEG_TIMEOUT_S} s"
                ) from None
            finally:
                feeder.join()
    # Either failure fails the other (no input, or nowhere to write): report both
    failures = []
    if encoder.returncode:
        failures.append(_failure(f"{encoder_name} encode", encoder.returncode, b"".join(errors)))
    if decoder.returncode:
        failures.append(_failure(f"{encoder_name} decode", decoder.returncode, decode_errors))
    if failures:
        raise RuntimeError("\n".join(failures))
    return output


@register_class("compress")
class Compress:
    """Class for compressing images or videos using various algorithms and parameters.

    Args:
        compress_dict (dict): A dictionary containing compression settings.
            It should include the following keys:
                - "algorithm" (list of str): List of compression algorithms to be used.
                - "comp" (list of int, optional): Range of compression values for algorithms. Defaults to [90, 100].
                - "target_compress" (dict, optional): Target compression values for specific algorithms.
                    Defaults to None.
                - "probability" (float, optional): Probability of applying compression. Defaults to 1.0.
                - "jpeg_sampling" (list of str, optional): List of JPEG subsampling factors. Defaults to ["4:2:2"].
    """

    def __init__(self, compress_dict: dict):
        self.algorithm = compress_dict["algorithm"]
        compress = compress_dict.get("compress", [90, 100])
        target = compress_dict.get("target_compress")
        self.probability = compress_dict.get("probability", 1.0)
        self.jpeg_sampling = compress_dict.get("jpeg_sampling", ["4:2:2"])
        self.video_sampling = compress_dict.get("video_sampling", ["444", "422", "420"])
        if target:
            self.target_compress = {
                "jpeg": target.get("jpeg", compress),
                "webp": target.get("webp", compress),
                "h264": target.get("h264", compress),
                "hevc": target.get("hevc", compress),
                "vp9": target.get("vp9", compress),
                "mpeg2": target.get("mpeg2", compress),
                "mpeg4": target.get("mpeg4", compress),
            }
        else:
            self.target_compress = {
                "jpeg": compress,
                "webp": compress,
                "h264": compress,
                "hevc": compress,
                "vp9": compress,
                "mpeg2": compress,
                "mpeg4":  compress,
            }

    @staticmethod
    def __pad_to_chroma(lq: np.ndarray, sampling: str):
        """Reflect-pad to even dims required by chroma subsampling, return (padded, orig_h, orig_w)."""
        h, w = lq.shape[:2]
        # yuv420p/yuv422p need even width; yuv420p also needs even height
        need_w = 2 if "420" in sampling or "422" in sampling else 1
        need_h = 2 if "420" in sampling else 1
        pad_w = (need_w - w % need_w) % need_w
        pad_h = (need_h - h % need_h) % need_h
        if pad_w == 0 and pad_h == 0:
            return lq, h, w
        padded = np.pad(lq, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        return padded, h, w

    def __video_core_pyav(
        self, lq: np.ndarray, codec: str, options: dict, container_fmt: str, sampling: str
    ) -> np.ndarray:
        """In-process video codec roundtrip via PyAV. No subprocess spawning."""
        orig_height, orig_width, channel = lq.shape
        supported = {f.name for f in av.codec.Codec(codec, "w").video_formats or ()}
        sampling = _nearest_chroma(sampling, supported)

        lq, _, _ = self.__pad_to_chroma(lq, sampling)
        height, width = lq.shape[:2]

        buf = io.BytesIO()
        output = av.open(buf, mode="w", format=container_fmt)
        stream = output.add_stream(codec, rate=1)
        stream.width = width
        stream.height = height
        stream.pix_fmt = sampling
        stream.gop_size = 1
        stream.options = options
        stream.codec_context.thread_type = "AUTO"
        stream.codec_context.thread_count = 0

        frame = av.VideoFrame.from_ndarray(lq, format="rgb24")
        for pkt in stream.encode(frame):
            output.mux(pkt)
        for pkt in stream.encode(None):
            output.mux(pkt)
        output.close()

        buf.seek(0)
        dec_container = av.open(buf)
        dec_stream = dec_container.streams.video[0]
        dec_stream.codec_context.thread_type = "AUTO"
        dec_stream.codec_context.thread_count = 0
        for decoded_frame in dec_container.decode(dec_stream):
            result = decoded_frame.to_ndarray(format="rgb24")
            break
        dec_container.close()

        logging.debug(f"Compress - {codec} (PyAV) subsampling: {sampling}")
        return result[:orig_height, :orig_width, :]

    def __video_core(
        self, lq: np.ndarray, ffmpeg: str, encoder: str, options: dict, stream: tuple,
        sampling: str,
    ) -> np.ndarray:
        """Video codec roundtrip through the ffmpeg executable: one process
        encodes the frame (intra only) into a raw elementary stream, a second
        decodes it."""
        orig_height, orig_width, channel = lq.shape
        sampling = _nearest_chroma(sampling, video_backend.pixel_formats(ffmpeg, encoder))

        # Pad odd dimensions so chroma subsampling doesn't reject them
        lq, _, _ = self.__pad_to_chroma(lq, sampling)
        height, width = lq.shape[:2]

        encode = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "1",
            "-i", "pipe:",
            "-c:v", encoder, "-pix_fmt", sampling, "-g", "1", "-bf", "0", "-threads", "0",
        ]
        for key, value in options.items():
            encode += [f"-{key}", value]
        muxer, demuxer = stream
        encode += ["-f", muxer, "pipe:"]
        decode = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "0",
            "-f", demuxer, "-i", "pipe:",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:",
        ]
        raw = _encode_decode(encode, decode, lq.tobytes(), encoder)
        size = height * width * channel
        if len(raw) < size:
            raise RuntimeError(f"ffmpeg {encoder} decoded {len(raw)} bytes, expected {size}")
        frame_data = np.frombuffer(raw, dtype=np.uint8, count=size).reshape(
            (height, width, channel)
        )
        logging.debug(f"Compress - {encoder} (ffmpeg) subsampling: {sampling}")

        # Crop back to original dimensions
        return frame_data[:orig_height, :orig_width, :]

    def __video(self, lq: np.ndarray, algorithm: str, quality: int) -> np.ndarray:
        """Video codec roundtrip on the backend video_backend.detect() picks;
        an encoder the system ffmpeg lacks runs on PyAV."""
        encoder, container, stream = _VIDEO_FORMATS[algorithm]
        options = _video_options(algorithm, quality)
        sampling = VIDEO_SUBSAMPLING[random.choice(self.video_sampling)]
        backend = video_backend.detect()
        if backend.kind == "ffmpeg":
            if encoder in backend.encoders:
                return self.__video_core(lq, backend.path, encoder, options, stream, sampling)
            if (backend.path, encoder) not in _FALLBACKS_LOGGED:
                _FALLBACKS_LOGGED.add((backend.path, encoder))
                logging.warning("%s has no %s encoder; %s runs on PyAV", backend.path, encoder,
                                algorithm)
        if _HAS_PYAV:
            return self.__video_core_pyav(lq, encoder, options, container, sampling)
        raise RuntimeError(
            f"{algorithm} needs FFmpeg: locate an ffmpeg with {encoder}, or install PyAV (av)"
        )

    def __jpeg(self, lq: np.ndarray, quality: int) -> np.ndarray:
        """Compresses an image using JPEG format.

        Args:
            lq (numpy.ndarray): The input image in RGB format.
            quality (int): The quality level for compression.

        Returns:
            numpy.ndarray: The compressed image.
        """
        jpeg_sampling = random.choice(self.jpeg_sampling)
        encode_param = [
            int(cv.IMWRITE_JPEG_QUALITY),
            quality,
            cv.IMWRITE_JPEG_SAMPLING_FACTOR,
            JPEG_SUBSAMPLING[jpeg_sampling],
        ]
        logging.debug(f"Compress - jpeg sampling: {jpeg_sampling}")
        _, encimg = cv.imencode(".jpg", lq, encode_param)
        return cv.imdecode(encimg, 1).copy()

    def __webp(self, lq: np.ndarray, quality: int) -> np.ndarray:
        """Compresses an image using WebP format.

        Uses Pillow with method=0 (fastest libwebp setting) instead of cv2.
        cv2 doesn't expose the WebP method parameter. method=0 is 1.8× faster.

        Args:
            lq (numpy.ndarray): The input image in BGR uint8 format.
            quality (int): The quality level for compression (1-100).

        Returns:
            numpy.ndarray: The compressed image in BGR uint8.
        """
        from PIL import Image
        import io

        rgb = cv.cvtColor(lq, cv.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="webp", quality=quality, method=0)
        buf.seek(0)
        decoded = np.array(Image.open(buf))
        return cv.cvtColor(decoded, cv.COLOR_RGB2BGR)

    def run(self, lq: np.ndarray, hq: np.ndarray) -> (np.ndarray, np.ndarray):
        """Compresses the input image.

        Args:
            lq (numpy.ndarray): The low-quality image.
            hq (numpy.ndarray): The corresponding high-quality image.

        Returns:
            tuple: A tuple containing the compressed low-quality image
                and the corresponding high-quality image.
        """
        if probability(self.probability):
            return lq, hq
        gray = False
        if lq.ndim == 3 and lq.shape[2] == 3:
            lq = (lq * 255.0).astype(np.uint8)
            lq = cv.cvtColor(lq, cv.COLOR_RGB2BGR)
        else:
            lq = cv.cvtColor((lq * 255.0).astype(np.uint8), cv.COLOR_GRAY2BGR)
            gray = True

        algorithm = random.choice(self.algorithm)
        random_comp = safe_randint(self.target_compress[algorithm])
        logging.debug(f"Compress - algorithm: {algorithm} compress: {random_comp}")
        if algorithm in _VIDEO_FORMATS:
            lq = self.__video(lq, algorithm, random_comp)
        else:
            lq = {"jpeg": self.__jpeg, "webp": self.__webp}[algorithm](lq, random_comp)

        if gray:
            lq = cv.cvtColor(lq, cv.COLOR_BGR2GRAY)
        else:
            lq = cv.cvtColor(lq, cv.COLOR_BGR2RGB)
        return lq.astype(np.float32) / 255.0, hq
