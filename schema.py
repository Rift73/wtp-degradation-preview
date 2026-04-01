"""
Degradation parameter schemas for the WTP Degradation Preview GUI.

Each schema defines the GUI-facing parameters for one degradation type,
plus a build function that converts GUI values into the config dict
expected by the degradation class.

Config type mapping (GUI value → pipeline config format):
  uniform_range → [v, v]   (for safe_uniform)
  randint_range → [v, v]   (for safe_randint)
  arange_single → [v]      (for safe_arange → np.random.choice)
  choice_list   → [v]      (for np.random.choice on a list)
  raw           → v         (pass through)
"""

SCHEMAS = {}

# Category colors for the left-border accent on degradation blocks
CATEGORY_COLORS = {
    "blur":         "#5B9DF5",  # blue - blur/smooth
    "noise":        "#D4A843",  # amber - noise/grain
    "hf_noise":     "#D4A843",  # amber - noise/grain
    "compress":     "#E05555",  # red - compression artifacts
    "resize":       "#8B6FC0",  # purple - geometric
    "color":        "#4CAF6A",  # green - color/levels
    "halo":         "#C77D4F",  # orange - sharpen/halo
    "dithering":    "#7AAFB7",  # teal - pattern
    "saturation":   "#4CAF6A",  # green - color
    "pixelate":     "#8B6FC0",  # purple - geometric
    "screentone":   "#7AAFB7",  # teal - pattern
    "subsampling":  "#7AAFB7",  # teal - pattern
    "shift":        "#5B9DF5",  # blue - channel
    "sin":          "#7AAFB7",  # teal - pattern
    "canny":        "#D4A843",  # amber - detection
    "rainbow":      "#E05555",  # red - video signal artifact
    "lowpass":      "#5B9DF5",  # blue - blur/filter
    "interlace":    "#8B6FC0",  # purple - geometric
    "overshoot":    "#C77D4F",  # orange - sharpen/edge
    "banding":      "#4CAF6A",  # green - color
    "filmgrain":    "#D4A843",  # amber - noise/grain
    "ghosting":     "#8B6FC0",  # purple - geometric
    "scanline":     "#7AAFB7",  # teal - pattern
    "ntsc":         "#E05555",  # red - video signal artifact
}


def _reg(key, label, params, build=None):
    SCHEMAS[key] = {"label": label, "params": params, "build": build}


def build_config(schema_key, gui_values):
    """Generic config builder. Converts GUI values to pipeline config dict."""
    schema = SCHEMAS[schema_key]
    if schema["build"] is not None:
        return schema["build"](gui_values)

    config = {"type": schema_key, "probability": 1.0}
    for p in schema["params"]:
        key = p.get("config_key", p["key"])
        val = gui_values[p["key"]]
        ct = p.get("config_type", _default_config_type(p["type"]))
        if ct == "uniform_range":
            config[key] = [val, val]
        elif ct == "randint_range":
            config[key] = [val, val]
        elif ct == "arange_single":
            config[key] = [val]
        elif ct == "choice_list":
            config[key] = [val]
        elif ct == "raw":
            config[key] = val
    return config


def _default_config_type(ptype):
    return {
        "float": "uniform_range",
        "int": "randint_range",
        "choice": "choice_list",
        "bool": "raw",
    }.get(ptype, "raw")


# ──────────────────────────────────────────────
# Blur
# ──────────────────────────────────────────────
_reg("blur", "Blur", [
    {"key": "filter", "label": "Filter", "type": "choice",
     "options": ["gauss", "box", "median", "lens", "motion", "random"],
     "default": "gauss"},
    {"key": "kernel", "label": "Kernel / Sigma", "type": "float",
     "min": 0.0, "max": 20.0, "step": 0.05, "default": 1.0, "decimals": 2},
    {"key": "motion_size", "label": "Motion Size", "type": "int",
     "min": 1, "max": 100, "default": 10},
    {"key": "motion_angle", "label": "Motion Angle", "type": "int",
     "min": 0, "max": 360, "default": 0},
])


# ──────────────────────────────────────────────
# Noise
# ──────────────────────────────────────────────
def _build_noise(p):
    config = {
        "type": "noise",
        "type_noise": [p["type_noise"]],
        "alpha": [p["alpha"]],
        "probability": 1.0,
        "y_noise": 0,
        "uv_noise": 0,
        "octaves": [p["octaves"]],
        "frequency": [p["frequency"]],
        "lacunarity": [p["lacunarity"]],
        "bias": [0, 0],
    }
    mode = p["color_mode"]
    if mode == "Y only":
        config["y_noise"] = 1.0
    elif mode == "UV only":
        config["uv_noise"] = 1.0
    return config


_reg("noise", "Noise", [
    {"key": "type_noise", "label": "Noise Type", "type": "choice",
     "options": ["uniform", "gauss", "perlin", "opensimplex", "simplex",
                 "supersimplex", "salt", "pepper", "salt_and_pepper"],
     "default": "gauss"},
    {"key": "alpha", "label": "Intensity", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.005, "default": 0.05, "decimals": 3},
    {"key": "color_mode", "label": "Color Mode", "type": "choice",
     "options": ["RGB", "Y only", "UV only"], "default": "RGB"},
    {"key": "octaves", "label": "Octaves (procedural)", "type": "int",
     "min": 1, "max": 8, "default": 1},
    {"key": "frequency", "label": "Frequency (procedural)", "type": "float",
     "min": 0.01, "max": 5.0, "step": 0.01, "default": 0.8, "decimals": 2},
    {"key": "lacunarity", "label": "Lacunarity (procedural)", "type": "float",
     "min": 0.01, "max": 5.0, "step": 0.01, "default": 0.4, "decimals": 2},
], build=_build_noise)


# ──────────────────────────────────────────────
# Compress
# ──────────────────────────────────────────────

# Per-codec quality profiles: label, min, max, default
# JPEG/WebP: direct quality (higher = better)
# H264/HEVC/VP9: CRF (lower = better), MPEG: qscale (lower = better)
CODEC_QUALITY_PROFILES = {
    "jpeg":  {"label": "Quality",  "min": 1,  "max": 100, "default": 80},
    "webp":  {"label": "Quality",  "min": 1,  "max": 100, "default": 80},
    "h264":  {"label": "CRF",      "min": 0,  "max": 51,  "default": 23},
    "hevc":  {"label": "CRF",      "min": 0,  "max": 51,  "default": 28},
    "vp9":   {"label": "CRF",      "min": 0,  "max": 63,  "default": 31},
    "mpeg2": {"label": "QScale",   "min": 1,  "max": 31,  "default": 4},
    "mpeg4": {"label": "QScale",   "min": 1,  "max": 31,  "default": 4},
}


def _build_compress(p):
    alg = p["algorithm"]
    return {
        "type": "compress",
        "algorithm": [alg],
        "compress": [p["quality"], p["quality"]],
        "probability": 1.0,
        "jpeg_sampling": [p["jpeg_sampling"]],
        "video_sampling": [p["video_sampling"]],
    }


_reg("compress", "Compression", [
    {"key": "algorithm", "label": "Algorithm", "type": "choice",
     "options": ["jpeg", "webp", "h264", "hevc", "mpeg2", "mpeg4", "vp9"],
     "default": "jpeg"},
    {"key": "quality", "label": "Quality", "type": "int",
     "min": 1, "max": 100, "default": 80,
     "profiles": {"source": "algorithm", "map": CODEC_QUALITY_PROFILES}},
    {"key": "jpeg_sampling", "label": "JPEG Sampling", "type": "choice",
     "options": ["4:4:4", "4:4:0", "4:2:2", "4:2:0", "4:1:1"],
     "default": "4:2:0"},
    {"key": "video_sampling", "label": "Video Sampling", "type": "choice",
     "options": ["444", "422", "420"], "default": "420"},
], build=_build_compress)


# ──────────────────────────────────────────────
# Resize
# ──────────────────────────────────────────────
_RESIZE_ALGS = [
    "nearest", "box", "hermite", "linear", "lagrange",
    "cubic_catrom", "cubic_mitchell", "cubic_bspline",
    "lanczos", "gauss", "mat_cubic",
]


def _build_resize(p):
    return {
        "type": "resize",
        "alg_lq": [p["alg_lq"]],
        "alg_hq": [p["alg_hq"]],
        "scale": p["scale"],
        "spread": [p["spread"]],
        "probability": 1.0,
        "color_fix": p.get("color_fix", False),
        "gamma_correction": p.get("gamma_correction", False),
    }


_reg("resize", "Resize", [
    {"key": "alg_lq", "label": "LQ Algorithm", "type": "choice",
     "options": _RESIZE_ALGS, "default": "lanczos"},
    {"key": "alg_hq", "label": "HQ Algorithm", "type": "choice",
     "options": _RESIZE_ALGS, "default": "lanczos"},
    {"key": "scale", "label": "Scale Factor", "type": "int",
     "min": 1, "max": 8, "default": 4, "config_type": "raw"},
    {"key": "spread", "label": "Spread", "type": "float",
     "min": 1.0, "max": 4.0, "step": 0.1, "default": 1.0, "decimals": 1},
    {"key": "color_fix", "label": "Color Fix", "type": "bool", "default": False},
    {"key": "gamma_correction", "label": "Gamma Correction", "type": "bool",
     "default": False},
], build=_build_resize)


# ──────────────────────────────────────────────
# Color Levels
# ──────────────────────────────────────────────
_reg("color", "Color Levels", [
    {"key": "high", "label": "Output High", "type": "int",
     "min": 0, "max": 255, "default": 255},
    {"key": "low", "label": "Output Low", "type": "int",
     "min": 0, "max": 255, "default": 0},
    {"key": "gamma", "label": "Gamma", "type": "float",
     "min": 0.1, "max": 5.0, "step": 0.01, "default": 1.0, "decimals": 2},
])


# ──────────────────────────────────────────────
# Halo (Unsharp Mask / Oversharpening)
# ──────────────────────────────────────────────
_reg("halo", "Halo / Sharpen", [
    {"key": "type_halo", "label": "Type", "type": "choice",
     "options": ["unsharp_mask", "unsharp_gray", "unsharp_halo"],
     "default": "unsharp_mask"},
    {"key": "kernel", "label": "Sigma", "type": "float",
     "min": 0.0, "max": 20.0, "step": 0.05, "default": 1.0, "decimals": 2},
    {"key": "amount", "label": "Amount", "type": "float",
     "min": 0.0, "max": 10.0, "step": 0.05, "default": 1.0, "decimals": 2},
    {"key": "threshold", "label": "Threshold (0-255)", "type": "float",
     "min": 0.0, "max": 255.0, "step": 1.0, "default": 0.0, "decimals": 0},
])


# ──────────────────────────────────────────────
# Dithering
# ──────────────────────────────────────────────
_reg("dithering", "Dithering", [
    {"key": "dithering_type", "label": "Algorithm", "type": "choice",
     "options": ["quantize", "floydsteinberg", "jarvisjudiceninke", "stucki",
                 "atkinson", "burkes", "sierra", "tworowsierra", "sierraLite",
                 "order", "riemersma"],
     "default": "floydsteinberg"},
    {"key": "color_ch", "label": "Color Levels", "type": "int",
     "min": 2, "max": 64, "default": 8},
    {"key": "map_size", "label": "Map Size (ordered)", "type": "int",
     "min": 2, "max": 16, "default": 4},
    {"key": "history", "label": "History (riemersma)", "type": "int",
     "min": 2, "max": 64, "default": 10},
    {"key": "ratio", "label": "Decay Ratio (riemersma)", "type": "float",
     "min": 0.01, "max": 0.99, "step": 0.01, "default": 0.5, "decimals": 2},
])


# ──────────────────────────────────────────────
# Saturation
# ──────────────────────────────────────────────
_reg("saturation", "Saturation", [
    {"key": "rand", "label": "Saturation Multiplier", "type": "float",
     "min": 0.0, "max": 2.0, "step": 0.01, "default": 0.5, "decimals": 2,
     "config_key": "rand"},
])


# ──────────────────────────────────────────────
# Pixelate
# ──────────────────────────────────────────────
_reg("pixelate", "Pixelate", [
    {"key": "size", "label": "Pixel Block Size", "type": "float",
     "min": 1.0, "max": 32.0, "step": 0.5, "default": 4.0, "decimals": 1},
])


# ──────────────────────────────────────────────
# Sin (Moiré Pattern)
# ──────────────────────────────────────────────
def _build_sin(p):
    wl = int(p["wavelength"])
    return {
        "type": "sin",
        "shape": [wl, wl + 1, 1],
        "alpha": [p["alpha"], p["alpha"]],
        "bias": [p["bias"], p["bias"]],
        "vertical": 1.0 if p["vertical"] else 0.0,
        "probability": 1.0,
    }


_reg("sin", "Sin Pattern", [
    {"key": "wavelength", "label": "Wavelength (px)", "type": "int",
     "min": 2, "max": 2000, "default": 200},
    {"key": "alpha", "label": "Amplitude", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.1, "decimals": 2},
    {"key": "bias", "label": "Bias", "type": "float",
     "min": 0.0, "max": 2.0, "step": 0.01, "default": 1.0, "decimals": 2},
    {"key": "vertical", "label": "Vertical", "type": "bool", "default": False},
], build=_build_sin)


# ──────────────────────────────────────────────
# Screentone / Halftone
# ──────────────────────────────────────────────
def _build_screentone(p):
    dt = p["dot_type"]
    angle_val = int(p["angle"])
    return {
        "type": "screentone",
        "dot_size": [p["dot_size"]],
        "dot_type": [dt],
        "angle": [angle_val],
        "probability": 1.0,
        "color": [{
            "type_halftone": [p["halftone_type"]],
            "dot": [
                {"type": [dt], "angle": [angle_val]},
                {"type": [dt], "angle": [angle_val]},
                {"type": [dt], "angle": [angle_val]},
                {"type": [dt], "angle": [angle_val]},
            ],
            "cmyk_alpha": [1, 1],
        }],
    }


_reg("screentone", "Screentone", [
    {"key": "halftone_type", "label": "Halftone Mode", "type": "choice",
     "options": ["cmyk", "rgb", "hsv", "not_rot", "gray"], "default": "rgb"},
    {"key": "dot_size", "label": "Dot Size", "type": "int",
     "min": 2, "max": 32, "default": 7},
    {"key": "dot_type", "label": "Dot Shape", "type": "choice",
     "options": ["circle", "line", "cross", "ellipse"], "default": "circle"},
    {"key": "angle", "label": "Angle", "type": "int",
     "min": 0, "max": 180, "default": 0},
], build=_build_screentone)


# ──────────────────────────────────────────────
# Subsampling
# ──────────────────────────────────────────────
_INTERP_ALGS = [
    "nearest", "box", "hermite", "linear", "lagrange",
    "cubic_catrom", "cubic_mitchell", "cubic_bspline",
    "lanczos", "gauss",
]


def _build_subsampling(p):
    config = {
        "type": "subsampling",
        "down": [p["down_alg"]],
        "up": [p["up_alg"]],
        "sampling": [p["sampling"]],
        "yuv": [p["yuv"]],
        "probability": 1.0,
    }
    blur_val = p.get("blur", 0.0)
    if blur_val > 0:
        config["blur"] = [blur_val, blur_val]
    return config


_reg("subsampling", "Chroma Subsampling", [
    {"key": "sampling", "label": "Format", "type": "choice",
     "options": ["4:4:4", "4:2:2", "4:2:0", "4:1:1", "4:1:0",
                 "4:4:0", "4:2:1", "4:1:2", "4:1:3"],
     "default": "4:2:0"},
    {"key": "down_alg", "label": "Down Algorithm", "type": "choice",
     "options": _INTERP_ALGS, "default": "linear"},
    {"key": "up_alg", "label": "Up Algorithm", "type": "choice",
     "options": _INTERP_ALGS, "default": "linear"},
    {"key": "yuv", "label": "YCbCr Standard", "type": "choice",
     "options": ["601", "709", "2020", "240"], "default": "709"},
    {"key": "blur", "label": "Chroma Blur Sigma", "type": "float",
     "min": 0.0, "max": 5.0, "step": 0.05, "default": 0.0, "decimals": 2},
], build=_build_subsampling)


# ──────────────────────────────────────────────
# Shift (Chromatic Aberration)
# ──────────────────────────────────────────────
def _build_shift(p):
    sx = p["shift_x"]
    sy = p["shift_y"]
    no = [[0, 0], [0, 0]]
    pos = [[sx, sx], [sy, sy]]
    neg = [[-sx, -sx], [-sy, -sy]]
    st = p["shift_type"]

    config = {
        "type": "shift",
        "shift_type": [st],
        "probability": 1.0,
    }

    if st == "rgb":
        config["rgb"] = {"r": pos, "g": no, "b": neg}
    elif st == "yuv":
        config["yuv"] = {"y": no, "u": pos, "v": neg}
    elif st == "cmyk":
        config["cmyk"] = {"c": pos, "m": no, "y": neg, "k": no}
    return config


_reg("shift", "Channel Shift", [
    {"key": "shift_type", "label": "Color Space", "type": "choice",
     "options": ["rgb", "yuv", "cmyk"], "default": "rgb"},
    {"key": "shift_x", "label": "Shift X (px)", "type": "int",
     "min": -50, "max": 50, "default": 2},
    {"key": "shift_y", "label": "Shift Y (px)", "type": "int",
     "min": -50, "max": 50, "default": 0},
], build=_build_shift)


# ──────────────────────────────────────────────
# Canny Edge Detection
# ──────────────────────────────────────────────
def _build_canny(p):
    return {
        "type": "canny",
        "thread1": [p["threshold1"]],
        "thread2": [p["threshold2_offset"]],
        "aperture_size": [int(p["aperture_size"])],
        "white": 1.0 if p["white_bg"] else 0.0,
        "probability": 1.0,
        "lq_hq": p.get("lq_hq", False),
    }


_reg("canny", "Canny Edge", [
    {"key": "threshold1", "label": "Threshold 1", "type": "int",
     "min": 1, "max": 255, "default": 50},
    {"key": "threshold2_offset", "label": "Threshold 2 Offset", "type": "int",
     "min": 0, "max": 200, "default": 50},
    {"key": "aperture_size", "label": "Aperture Size", "type": "choice",
     "options": ["3", "5", "7"], "default": "3"},
    {"key": "white_bg", "label": "White Background", "type": "bool",
     "default": False},
    {"key": "lq_hq", "label": "Replace HQ with LQ", "type": "bool",
     "default": False},
], build=_build_canny)


# ──────────────────────────────────────────────
# HF Noise (Beta-distributed texture noise)
# ──────────────────────────────────────────────
def _build_hf_noise(p):
    config = {
        "type": "hf_noise",
        "alpha": [p["alpha_min"], p["alpha_max"]],
        "beta_shape": [p["beta_shape_min"], p["beta_shape_max"]],
        "gray_prob": p["gray_prob"],
        "normalize": p["normalize"],
        "denoise": p["denoise"],
        "denoise_strength": p["denoise_strength"],
        "probability": 1.0,
    }
    if p["use_offset"]:
        config["beta_offset"] = [p["offset_min"], p["offset_max"]]
    return config


_reg("hf_noise", "HF Noise", [
    {"key": "alpha_min", "label": "Alpha Min", "type": "float",
     "min": 0.001, "max": 0.5, "step": 0.005, "default": 0.01, "decimals": 3},
    {"key": "alpha_max", "label": "Alpha Max", "type": "float",
     "min": 0.001, "max": 0.5, "step": 0.005, "default": 0.05, "decimals": 3},
    {"key": "beta_shape_min", "label": "Beta Shape Min", "type": "float",
     "min": 0.1, "max": 20.0, "step": 0.1, "default": 2.0, "decimals": 1},
    {"key": "beta_shape_max", "label": "Beta Shape Max", "type": "float",
     "min": 0.1, "max": 20.0, "step": 0.1, "default": 5.0, "decimals": 1},
    {"key": "gray_prob", "label": "Grayscale Probability", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.05, "default": 1.0, "decimals": 2},
    {"key": "normalize", "label": "Normalize", "type": "bool", "default": True},
    {"key": "use_offset", "label": "Use Beta Offset", "type": "bool", "default": False},
    {"key": "offset_min", "label": "Offset Min", "type": "float",
     "min": 0.0, "max": 20.0, "step": 0.1, "default": 1.0, "decimals": 1},
    {"key": "offset_max", "label": "Offset Max", "type": "float",
     "min": 0.0, "max": 20.0, "step": 0.1, "default": 5.0, "decimals": 1},
    {"key": "denoise", "label": "Denoise", "type": "bool", "default": False},
    {"key": "denoise_strength", "label": "Denoise Strength", "type": "float",
     "min": 1.0, "max": 150.0, "step": 1.0, "default": 30.0, "decimals": 0},
], build=_build_hf_noise)


# ──────────────────────────────────────────────
# Rainbow (Composite Video Artifact)
# ──────────────────────────────────────────────
def _build_rainbow(p):
    return {
        "type": "rainbow",
        "subcarrier_freq": [p["subcarrier_freq"], p["subcarrier_freq"]],
        "chroma_bandwidth": [p["chroma_bandwidth"], p["chroma_bandwidth"]],
        "intensity": [p["intensity"], p["intensity"]],
        "phase_alternation": p["phase_alternation"],
        "probability": 1.0,
    }


_reg("rainbow", "Rainbow (Composite)", [
    {"key": "subcarrier_freq", "label": "Subcarrier Freq (cyc/px)", "type": "float",
     "min": 0.05, "max": 0.50, "step": 0.01, "default": 0.25, "decimals": 2},
    {"key": "chroma_bandwidth", "label": "Chroma Bandwidth", "type": "float",
     "min": 0.01, "max": 0.25, "step": 0.005, "default": 0.08, "decimals": 3},
    {"key": "intensity", "label": "Intensity", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.05, "default": 1.0, "decimals": 2},
    {"key": "phase_alternation", "label": "Phase Alternation (NTSC)",
     "type": "bool", "default": True},
], build=_build_rainbow)


# ──────────────────────────────────────────────
# Lowpass (Frequency-domain Bandwidth Limit)
# ──────────────────────────────────────────────
def _build_lowpass(p):
    return {
        "type": "lowpass",
        "cutoff": [p["cutoff"], p["cutoff"]],
        "order": [p["order"], p["order"]],
        "detail_mask": p["detail_mask"],
        "mask_lines_brz": p["mask_lines_brz"],
        "probability": 1.0,
    }


_reg("lowpass", "Lowpass Filter", [
    {"key": "cutoff", "label": "Cutoff (fraction of Nyquist)", "type": "float",
     "min": 0.05, "max": 1.0, "step": 0.01, "default": 0.5, "decimals": 2},
    {"key": "order", "label": "Filter Order", "type": "int",
     "min": 1, "max": 10, "default": 2},
    {"key": "detail_mask", "label": "Detail Mask", "type": "bool", "default": False},
    {"key": "mask_lines_brz", "label": "Mask Lines Threshold", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.08, "decimals": 2},
], build=_build_lowpass)


# ──────────────────────────────────────────────
# NTSC Composite (Full Signal Simulation)
# ──────────────────────────────────────────────
def _build_ntsc(p):
    preset = p["preset"]
    # Preset determines enable_vhs and default values
    enable_vhs = preset in ("vhs_sp", "vhs_ep")
    return {
        "type": "ntsc",
        "preset": preset,
        "enable_vhs": enable_vhs,
        "comb_mode": p["comb_mode"],
        "noise": [p["noise"], p["noise"]],
        "luma_noise": [p["luma_noise"], p["luma_noise"]],
        "ghost_amplitude": [p["ghost_amplitude"], p["ghost_amplitude"]],
        "ghost_delay_us": [p["ghost_delay_us"], p["ghost_delay_us"]],
        "ghost_phase": [p["ghost_phase"], p["ghost_phase"]],
        "jitter": [p["jitter"], p["jitter"]],
        "edge_ringing": [p["edge_ringing"], p["edge_ringing"]],
        "vhs_luma_bw": [p["vhs_luma_bw"], p["vhs_luma_bw"]],
        "color_under_bw": [p["color_under_bw"], p["color_under_bw"]],
        "tape_trailing": [p["tape_trailing"], p["tape_trailing"]],
        "intensity": [p["intensity"], p["intensity"]],
        "probability": 1.0,
    }


_reg("ntsc", "NTSC Composite", [
    {"key": "preset", "label": "Preset", "type": "choice",
     "options": ["broadcast", "vhs_sp", "vhs_ep"], "default": "broadcast"},
    {"key": "comb_mode", "label": "Comb Filter", "type": "choice",
     "options": ["2sample", "1h"], "default": "2sample"},
    {"key": "noise", "label": "Noise", "type": "float",
     "min": 0.0, "max": 0.3, "step": 0.01, "default": 0.05, "decimals": 2},
    {"key": "luma_noise", "label": "Luma-Dependent Noise", "type": "float",
     "min": 0.0, "max": 0.15, "step": 0.01, "default": 0.0, "decimals": 2},
    {"key": "ghost_amplitude", "label": "Ghost Amplitude", "type": "float",
     "min": 0.0, "max": 0.5, "step": 0.01, "default": 0.0, "decimals": 2},
    {"key": "ghost_delay_us", "label": "Ghost Delay (us)", "type": "float",
     "min": 0.5, "max": 10.0, "step": 0.1, "default": 1.5, "decimals": 1},
    {"key": "ghost_phase", "label": "Ghost Phase (deg)", "type": "float",
     "min": 0.0, "max": 360.0, "step": 1.0, "default": 180.0, "decimals": 0},
    {"key": "jitter", "label": "Jitter", "type": "float",
     "min": 0.0, "max": 3.0, "step": 0.1, "default": 0.0, "decimals": 1},
    {"key": "edge_ringing", "label": "Edge Ringing", "type": "float",
     "min": 0.0, "max": 3.0, "step": 0.1, "default": 0.0, "decimals": 1},
    {"key": "vhs_luma_bw", "label": "VHS Luma BW (MHz)", "type": "float",
     "min": 1.5, "max": 4.2, "step": 0.1, "default": 4.2, "decimals": 1},
    {"key": "color_under_bw", "label": "Color-Under BW (kHz)", "type": "float",
     "min": 200.0, "max": 600.0, "step": 10.0, "default": 500.0, "decimals": 0},
    {"key": "tape_trailing", "label": "Tape Trailing", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.0, "decimals": 2},
    {"key": "intensity", "label": "Intensity", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.05, "default": 1.0, "decimals": 2},
], build=_build_ntsc)


# ──────────────────────────────────────────────
# Interlace (Combing Artifact)
# ──────────────────────────────────────────────
def _build_interlace(p):
    return {
        "type": "interlace",
        "field_shift": [p["field_shift"], p["field_shift"]],
        "dominant_field": [p["dominant_field"]],
        "probability": 1.0,
    }


_reg("interlace", "Interlace (Combing)", [
    {"key": "field_shift", "label": "Field Shift (px)", "type": "int",
     "min": 0, "max": 20, "default": 2},
    {"key": "dominant_field", "label": "Dominant Field", "type": "choice",
     "options": ["top", "bottom"], "default": "top"},
], build=_build_interlace)


# ──────────────────────────────────────────────
# Overshoot (Edge Ringing / Warp Sharp)
# ──────────────────────────────────────────────
def _build_overshoot(p):
    return {
        "type": "overshoot",
        "amount": [p["amount"], p["amount"]],
        "cutoff": [p["cutoff"], p["cutoff"]],
        "order": [p["order"], p["order"]],
        "probability": 1.0,
    }


_reg("overshoot", "Overshoot (Warp Sharp)", [
    {"key": "amount", "label": "Amount", "type": "float",
     "min": 0.0, "max": 5.0, "step": 0.1, "default": 1.5, "decimals": 1},
    {"key": "cutoff", "label": "Cutoff (fraction of Nyquist)", "type": "float",
     "min": 0.05, "max": 0.8, "step": 0.01, "default": 0.35, "decimals": 2},
    {"key": "order", "label": "Filter Order", "type": "int",
     "min": 1, "max": 5, "default": 2},
], build=_build_overshoot)


# ──────────────────────────────────────────────
# Color Banding (Bit Depth Reduction)
# ──────────────────────────────────────────────
def _build_banding(p):
    return {
        "type": "banding",
        "bits": [p["bits"], p["bits"]],
        "broadcast_range": p["broadcast_range"],
        "probability": 1.0,
    }


_reg("banding", "Color Banding", [
    {"key": "bits", "label": "Bit Depth", "type": "int",
     "min": 1, "max": 8, "default": 6},
    {"key": "broadcast_range", "label": "Broadcast Range (16-235)",
     "type": "bool", "default": False},
], build=_build_banding)


# ──────────────────────────────────────────────
# Film Grain
# ──────────────────────────────────────────────
def _build_filmgrain(p):
    return {
        "type": "filmgrain",
        "intensity": [p["intensity"], p["intensity"]],
        "grain_size": [p["grain_size"], p["grain_size"]],
        "midtone_bias": [p["midtone_bias"], p["midtone_bias"]],
        "probability": 1.0,
    }


_reg("filmgrain", "Film Grain", [
    {"key": "intensity", "label": "Intensity", "type": "float",
     "min": 0.0, "max": 0.3, "step": 0.005, "default": 0.05, "decimals": 3},
    {"key": "grain_size", "label": "Grain Size", "type": "float",
     "min": 0.5, "max": 5.0, "step": 0.1, "default": 1.5, "decimals": 1},
    {"key": "midtone_bias", "label": "Midtone Bias", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.8, "decimals": 2},
], build=_build_filmgrain)


# ──────────────────────────────────────────────
# Temporal Ghosting
# ──────────────────────────────────────────────
def _build_ghosting(p):
    return {
        "type": "ghosting",
        "shift_x": [p["shift_x"], p["shift_x"]],
        "shift_y": [p["shift_y"], p["shift_y"]],
        "opacity": [p["opacity"], p["opacity"]],
        "probability": 1.0,
    }


_reg("ghosting", "Temporal Ghosting", [
    {"key": "shift_x", "label": "Shift X (px)", "type": "int",
     "min": -20, "max": 20, "default": 4},
    {"key": "shift_y", "label": "Shift Y (px)", "type": "int",
     "min": -20, "max": 20, "default": 0},
    {"key": "opacity", "label": "Opacity", "type": "float",
     "min": 0.0, "max": 0.5, "step": 0.01, "default": 0.15, "decimals": 2},
], build=_build_ghosting)


# ──────────────────────────────────────────────
# Scanline (CRT Darkening)
# ──────────────────────────────────────────────
def _build_scanline(p):
    return {
        "type": "scanline",
        "strength": [p["strength"], p["strength"]],
        "even_lines": p["even_lines"],
        "probability": 1.0,
    }


_reg("scanline", "Scanline (CRT)", [
    {"key": "strength", "label": "Strength", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.3, "decimals": 2},
    {"key": "even_lines", "label": "Darken Even Lines",
     "type": "bool", "default": True},
], build=_build_scanline)


# ──────────────────────────────────────────────
# Ordered list for the Add menu
# ──────────────────────────────────────────────
SCHEMA_ORDER = [
    "blur", "lowpass", "noise", "hf_noise", "filmgrain", "compress", "resize",
    "color", "halo", "overshoot", "banding",
    "saturation", "pixelate", "dithering", "screentone",
    "subsampling", "shift", "rainbow", "ntsc", "interlace", "ghosting", "scanline",
    "sin", "canny",
]
