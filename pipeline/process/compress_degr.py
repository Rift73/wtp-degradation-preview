import io
import subprocess
import sys

import numpy as np

# Hide console windows on Windows when spawning ffmpeg
_POPEN_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
from numpy import random
import cv2 as cv
from .utils import probability

from ..constants import JPEG_SUBSAMPLING, VIDEO_SUBSAMPLING
from ..utils.random import safe_randint
from ..utils.registry import register_class
import logging

try:
    import av
    _HAS_PYAV = True
except ImportError:
    _HAS_PYAV = False

try:
    import torch
    from torchcodec.encoders import VideoEncoder
    from torchcodec.decoders import VideoDecoder
    _HAS_TORCHCODEC = True
except ImportError:
    _HAS_TORCHCODEC = False

# TorchCodec codec name + container format mapping
_TC_CODEC_MAP = {
    "h264": ("libx264", "mp4"),
    "hevc": ("libx265", "mp4"),
    "mpeg4": ("mpeg4", "mp4"),
}


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
        self, lq: np.ndarray, codec: str, options: dict, container_fmt: str = "mp4"
    ) -> np.ndarray:
        """In-process video codec roundtrip via PyAV. No subprocess spawning."""
        orig_height, orig_width, channel = lq.shape
        sampling = VIDEO_SUBSAMPLING[random.choice(self.video_sampling)]

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

        frame = av.VideoFrame.from_ndarray(lq, format="rgb24")
        for pkt in stream.encode(frame):
            output.mux(pkt)
        for pkt in stream.encode(None):
            output.mux(pkt)
        output.close()

        buf.seek(0)
        dec_container = av.open(buf)
        for decoded_frame in dec_container.decode(video=0):
            result = decoded_frame.to_ndarray(format="rgb24")
            break
        dec_container.close()

        logging.debug(f"Compress - {codec} (PyAV) subsampling: {sampling}")
        return result[:orig_height, :orig_width, :]

    def __video_core_torchcodec(
        self, lq: np.ndarray, codec_key: str, quality: int
    ) -> np.ndarray:
        """In-process video codec roundtrip via TorchCodec.

        Fastest path for h264/hevc/mpeg4. Native tensor I/O,
        single encode call, no subprocess. Intra-only (gop=1, bf=0)
        for semantic equivalence with per-image compression.
        """
        orig_height, orig_width = lq.shape[:2]
        sampling = random.choice(self.video_sampling)
        pix_fmt = VIDEO_SUBSAMPLING.get(sampling, "yuv420p")

        # Pad to even dims if needed by chroma subsampling
        need_w = 2 if "420" in pix_fmt or "422" in pix_fmt else 1
        need_h = 2 if "420" in pix_fmt else 1
        pad_w = (need_w - orig_width % need_w) % need_w
        pad_h = (need_h - orig_height % need_h) % need_h
        if pad_h or pad_w:
            lq = np.pad(lq, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        codec_name, fmt = _TC_CODEC_MAP[codec_key]

        # HWC uint8 → CHW → NCHW tensor (batch=1)
        tensor = torch.from_numpy(lq.transpose(2, 0, 1)).unsqueeze(0)

        # Intra-only encoding: gop=1 ensures no inter-frame prediction
        extra_options: dict[str, str] = {"g": "1", "bf": "0"}
        if codec_name == "libx264":
            extra_options["preset"] = "ultrafast"
        elif codec_name == "libx265":
            extra_options["preset"] = "ultrafast"
            extra_options["x265-params"] = "log-level=0"

        enc = VideoEncoder(frames=tensor, frame_rate=1)
        encode_kwargs: dict = {
            "format": fmt,
            "codec": codec_name,
            "crf": quality,
            "pixel_format": pix_fmt,
            "extra_options": extra_options,
        }
        encoded = enc.to_tensor(**encode_kwargs)

        dec = VideoDecoder(encoded)
        result = dec[0].numpy().transpose(1, 2, 0)  # CHW → HWC uint8

        logging.debug(f"Compress - {codec_key} (TorchCodec) subsampling: {pix_fmt}")
        return result[:orig_height, :orig_width, :]

    def __video_core(
        self, lq: np.ndarray, codec: str, output_args: list, container: str = "mpeg"
    ) -> np.ndarray:
        orig_height, orig_width, channel = lq.shape
        sampling = VIDEO_SUBSAMPLING[random.choice(self.video_sampling)]

        # Pad odd dimensions so chroma subsampling doesn't reject them
        lq, _, _ = self.__pad_to_chroma(lq, sampling)
        height, width = lq.shape[:2]

        process1 = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-threads", "0",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                "30",
                "-i",
                "pipe:",
                "-vcodec",
                codec,
                "-an",
                "-f",
                container,
                "-pix_fmt",
                f"{sampling}",
            ]
            + output_args
            + ["pipe:"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            creationflags=_POPEN_FLAGS,
        )

        process1.stdin.write(lq.tobytes())
        process1.stdin.flush()
        process1.stdin.close()

        process2 = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-threads", "0",
                "-f",
                container,
                "-i",
                "pipe:",
                "-pix_fmt",
                "rgb24",
                "-f",
                "image2pipe",
                "-vcodec",
                "rawvideo",
                "pipe:",
            ],
            stdin=process1.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_POPEN_FLAGS,
        )
        raw_frame = process2.stdout.read()[:(height * width * channel)]
        process2.stdout.close()
        process2.stderr.close()
        process1.wait()
        process2.wait()
        frame_data = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
            (height, width, channel)
        )
        logging.debug(f"Blur - {codec} subsampling: {sampling}")

        # Crop back to original dimensions
        return frame_data[:orig_height, :orig_width, :]

    def __h264(self, lq: np.ndarray, quality: int) -> np.ndarray:
        if _HAS_TORCHCODEC:
            return self.__video_core_torchcodec(lq, "h264", quality)
        if _HAS_PYAV:
            return self.__video_core_pyav(
                lq, "libx264", {"preset": "ultrafast", "crf": str(quality)}, "mp4"
            )
        output_args = ["-crf", str(quality)]
        return self.__video_core(lq, "h264", output_args)

    def __hevc(self, lq: np.ndarray, quality: int) -> np.ndarray:
        if _HAS_TORCHCODEC:
            return self.__video_core_torchcodec(lq, "hevc", quality)
        if _HAS_PYAV:
            return self.__video_core_pyav(
                lq, "libx265",
                {"preset": "ultrafast", "crf": str(quality), "x265-params": "log-level=0"},
                "mp4",
            )
        output_args = ["-crf", str(quality), "-x265-params", "log-level=0"]
        return self.__video_core(lq, "hevc", output_args)

    def __mpeg2(self, lq: np.ndarray, quality: int) -> np.ndarray:
        # TorchCodec can't decode MPEG2 — use PyAV
        if _HAS_PYAV:
            return self.__video_core_pyav(
                lq, "mpeg2video",
                {"qscale:v": str(quality), "qmax": str(quality), "qmin": str(quality)},
                "mpegts",
            )
        output_args = [
            "-qscale:v", str(quality), "-qmax", str(quality), "-qmin", str(quality),
        ]
        return self.__video_core(lq, "mpeg2video", output_args)

    def __mpeg4(self, lq: np.ndarray, quality: int) -> np.ndarray:
        if _HAS_TORCHCODEC:
            return self.__video_core_torchcodec(lq, "mpeg4", quality)
        if _HAS_PYAV:
            return self.__video_core_pyav(
                lq, "mpeg4",
                {"qscale:v": str(quality), "qmax": str(quality), "qmin": str(quality)},
                "mp4",
            )
        output_args = [
            "-qscale:v", str(quality), "-qmax", str(quality), "-qmin", str(quality),
        ]
        return self.__video_core(lq, "mpeg4", output_args)

    def __vp9(self, lq: np.ndarray, quality: int) -> np.ndarray:
        # TorchCodec can't pass VP9 speed option — use PyAV
        if _HAS_PYAV:
            return self.__video_core_pyav(
                lq, "libvpx-vp9",
                {"cpu-used": "8", "crf": str(quality), "b:v": "0", "row-mt": "1"},
                "webm",
            )
        output_args = ["-crf", str(quality), "-b:v", "0"]
        return self.__video_core(lq, "libvpx-vp9", output_args, "webm")

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
        COMPRESS_TYPE_MAP = {
            "jpeg": self.__jpeg,
            "webp": self.__webp,
            "h264": self.__h264,
            "hevc": self.__hevc,
            "mpeg2": self.__mpeg2,
            "mpeg4": self.__mpeg4,
            "vp9": self.__vp9,
        }
        try:
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
            lq = COMPRESS_TYPE_MAP[algorithm](lq, random_comp)

            if gray:
                lq = cv.cvtColor(lq, cv.COLOR_BGR2GRAY)
            else:
                lq = cv.cvtColor(lq, cv.COLOR_BGR2RGB)
            return lq.astype(np.float32) / 255.0, hq
        except Exception as e:
            logging.error(f"Compress error: {e}")
