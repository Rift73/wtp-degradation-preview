import numpy as np
from .utils import probability
from ..utils.registry import register_class
import logging

try:
    import torch
    from optimized.gpu_degradations import ntsc_composite_pt

    _HAS_GPU_NTSC = True
except ImportError:
    _HAS_GPU_NTSC = False

# Presets: (enable_vhs, vhs_luma_bw_MHz, color_under_bw_kHz, noise, luma_noise, edge_ringing)
_PRESETS = {
    "broadcast": (False, 4.2, 500, 0.03, 0.0, 0.0),
    "vhs_sp": (True, 3.0, 500, 0.06, 0.03, 0.8),
    "vhs_ep": (True, 1.6, 300, 0.10, 0.06, 0.5),
}


@register_class("ntsc")
class NTSCComposite:
    """Full NTSC composite video simulation.

    Physically accurate encode/decode at real NTSC sample rate (754x480),
    with optional VHS tape-path effects, ghosting, jitter, and tape trailing.

    Args:
        config (dict): Configuration dictionary with keys:
            - "preset" (str): "broadcast", "vhs_sp", "vhs_ep", or "custom".
            - "comb_mode" (str): "2sample" or "1h". Default "2sample".
            - "noise" (list[float]): Gaussian noise amplitude range. Default from preset.
            - "luma_noise" (list[float]): Luma-dependent noise range. Default from preset.
            - "ghost_amplitude" (list[float]): Ghost strength range. Default [0, 0].
            - "ghost_delay_us" (list[float]): Ghost delay range in us. Default [1.5, 1.5].
            - "ghost_phase" (list[float]): Ghost phase range in degrees. Default [180, 180].
            - "jitter" (list[float]): Jitter strength range. Default [0, 0].
            - "edge_ringing" (list[float]): Edge ringing range. Default from preset.
            - "vhs_luma_bw" (list[float]): VHS luma BW range in MHz. Default from preset.
            - "color_under_bw" (list[float]): Color-under BW range in kHz. Default from preset.
            - "tape_trailing" (list[float]): Tape trailing range. Default [0, 0].
            - "intensity" (list[float]): Blend intensity range. Default [1, 1].
            - "probability" (float): Probability of applying. Default 1.0.
    """

    def __init__(self, config: dict):
        self.probability = config.get("probability", 1.0)
        self.preset = config.get("preset", "broadcast")
        self.comb_mode = config.get("comb_mode", "2sample")

        # Load preset defaults
        enable_vhs, vlbw, cubw, n, ln, er = _PRESETS.get(self.preset, _PRESETS["broadcast"])
        self.enable_vhs = config.get("enable_vhs", enable_vhs)

        self.noise = config.get("noise", [n, n])
        self.luma_noise = config.get("luma_noise", [ln, ln])
        self.ghost_amplitude = config.get("ghost_amplitude", [0.0, 0.0])
        self.ghost_delay_us = config.get("ghost_delay_us", [1.5, 1.5])
        self.ghost_phase = config.get("ghost_phase", [180.0, 180.0])
        self.jitter = config.get("jitter", [0.0, 0.0])
        self.edge_ringing = config.get("edge_ringing", [er, er])
        self.vhs_luma_bw = config.get("vhs_luma_bw", [vlbw, vlbw])
        self.color_under_bw = config.get("color_under_bw", [cubw, cubw])
        self.tape_trailing = config.get("tape_trailing", [0.0, 0.0])
        self.intensity = config.get("intensity", [1.0, 1.0])

    def run(self, lq: np.ndarray, hq: np.ndarray) -> tuple:
        if probability(self.probability):
            return lq, hq
        if lq.ndim == 2:
            return lq, hq

        if not (_HAS_GPU_NTSC and torch.cuda.is_available()):
            logging.warning("NTSC requires CUDA — skipping")
            return lq, hq

        # Sample parameters
        noise_val = float(np.random.uniform(*self.noise))
        luma_noise_val = float(np.random.uniform(*self.luma_noise))
        ghost_amp = float(np.random.uniform(*self.ghost_amplitude))
        ghost_del = float(np.random.uniform(*self.ghost_delay_us))
        ghost_ph = float(np.random.uniform(*self.ghost_phase))
        jitter_val = float(np.random.uniform(*self.jitter))
        ringing_val = float(np.random.uniform(*self.edge_ringing))
        vlbw_val = float(np.random.uniform(*self.vhs_luma_bw))
        cubw_val = float(np.random.uniform(*self.color_under_bw))
        trail_val = float(np.random.uniform(*self.tape_trailing))
        intensity_val = float(np.random.uniform(*self.intensity))

        logging.debug(
            "NTSC - preset: %s noise: %.3f luma_noise: %.3f ghost: %.2f "
            "ringing: %.1f vhs_bw: %.1f trail: %.2f",
            self.preset, noise_val, luma_noise_val, ghost_amp,
            ringing_val, vlbw_val, trail_val,
        )

        tensor = torch.from_numpy(lq.transpose(2, 0, 1)[None]).cuda()
        result = ntsc_composite_pt(
            tensor,
            noise=noise_val,
            luma_noise=luma_noise_val,
            ghost_amplitude=ghost_amp,
            ghost_delay_us=ghost_del,
            ghost_phase=ghost_ph,
            jitter=jitter_val,
            edge_ringing=ringing_val,
            vhs_luma_bw=vlbw_val,
            color_under_bw=cubw_val,
            tape_trailing=trail_val,
            intensity=intensity_val,
            comb_mode=self.comb_mode,
            enable_vhs=self.enable_vhs,
        )
        lq = result.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        return np.clip(lq, 0, 1).astype(np.float32), hq
