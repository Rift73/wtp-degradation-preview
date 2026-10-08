"""
Degradation parameter schemas for the WTP Degradation Preview GUI.

Each schema defines the GUI-facing parameters for one degradation type,
plus a build function that converts GUI values into the config dict
expected by the degradation class, and optionally a summary function
for the one-line card summary. Every param carries a "help" sentence.

Config type mapping (GUI value → pipeline config format):
  uniform_range → [v, v]   (for safe_uniform)
  randint_range → [v, v]   (for safe_randint)
  arange_single → [v]      (for safe_arange → np.random.choice)
  choice_list   → [v]      (for np.random.choice on a list)
  raw           → v         (pass through)
"""

SCHEMAS = {}

# Category accent colors, in Add-menu order
CATEGORY_COLORS = {
    "Blur / Filter":  "#5B9DF5",  # blue
    "Noise / Grain":  "#D4A843",  # amber
    "Compression":    "#E05555",  # red
    "Color":          "#4CAF6A",  # green
    "Pattern":        "#7AAFB7",  # teal
    "Edge / Sharpen": "#C77D4F",  # orange
    "Geometric":      "#8B6FC0",  # purple
    "Video signal":   "#E05555",  # red
}

CATEGORY_OF = {
    "blur":        "Blur / Filter",
    "lowpass":     "Blur / Filter",
    "noise":       "Noise / Grain",
    "hf_noise":    "Noise / Grain",
    "filmgrain":   "Noise / Grain",
    "compress":    "Compression",
    "subsampling": "Compression",
    "color":       "Color",
    "saturation":  "Color",
    "banding":     "Color",
    "dithering":   "Pattern",
    "screentone":  "Pattern",
    "sin":         "Pattern",
    "scanline":    "Pattern",
    "halo":        "Edge / Sharpen",
    "overshoot":   "Edge / Sharpen",
    "canny":       "Edge / Sharpen",
    "resize":      "Geometric",
    "pixelate":    "Geometric",
    "shift":       "Geometric",
    "rainbow":     "Video signal",
    "ntsc":        "Video signal",
    "interlace":   "Video signal",
    "ghosting":    "Video signal",
}

_SUMMARY_LEN = 60


def _reg(key, label, params, build=None, summary=None):
    SCHEMAS[key] = {"label": label, "params": params, "build": build,
                    "summary": summary}


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


def _format_value(p, val):
    if p["type"] == "float":
        return f"{val:.{p.get('decimals', 2)}f}"
    if p["type"] == "int":
        return str(int(val))
    if p["type"] == "bool":
        return "on" if val else "off"
    return str(val)


def summarize(schema_key, values):
    """One-line (at most 60 chars) summary of a step's settings for its card.

    Missing values fall back to the param defaults. Schemas without their own
    summary show the first two params that differ from their defaults (or the
    first two params when nothing was changed).
    """
    schema = SCHEMAS[schema_key]
    full = {p["key"]: values.get(p["key"], p["default"]) for p in schema["params"]}
    if schema["summary"] is not None:
        text = schema["summary"](full)
    else:
        params = schema["params"]
        changed = [p for p in params if full[p["key"]] != p["default"]]
        text = " · ".join(
            f"{p['label'].split(' (')[0]} {_format_value(p, full[p['key']])}"
            for p in (changed or params)[:2]
        )
    if len(text) > _SUMMARY_LEN:
        return text[:_SUMMARY_LEN - 1] + "…"
    return text


# ──────────────────────────────────────────────
# Blur
# ──────────────────────────────────────────────
def _build_blur(p):
    median = int(p["median_size"])
    median += 1 - median % 2  # medianBlur needs an odd kernel
    return {
        "type": "blur",
        "probability": 1.0,
        "filter": [p["filter"]],
        "kernel": [p["kernel"], p["kernel"]],
        "motion_size": [p["motion_size"], p["motion_size"]],
        "motion_angle": [p["motion_angle"], p["motion_angle"]],
        "target_kernel": {"median": [median, median]},
    }


def _summary_blur(p):
    f = p["filter"]
    if f == "median":
        return f"median · {int(p['median_size']) | 1} px"
    if f == "motion":
        return f"motion · {p['motion_size']} px {p['motion_angle']}°"
    return f"{f} · σ {p['kernel']:.2f}"


_reg("blur", "Blur", [
    {"key": "filter", "label": "Filter", "type": "choice",
     "options": ["gauss", "box", "median", "lens", "motion", "random"],
     "default": "gauss",
     "help": "Blur kernel shape: Gaussian, box, median, lens (disc), motion "
             "streak, or a random anisotropic kernel."},
    {"key": "kernel", "label": "Kernel / Sigma", "type": "float",
     "min": 0.0, "max": 20.0, "step": 0.05, "default": 1.0, "decimals": 2,
     "help": "Blur radius for gauss, box, lens and random; bigger values blur "
             "more, 0 turns the blur off."},
    {"key": "median_size", "label": "Median Size (px)", "type": "int",
     "min": 3, "max": 31, "default": 5,
     "help": "Window size of the median filter (rounded up to odd); bigger "
             "values flatten more detail into paint-like patches."},
    {"key": "motion_size", "label": "Motion Size", "type": "int",
     "min": 1, "max": 100, "default": 10,
     "help": "Length in pixels of the motion-blur streak; bigger values smear "
             "further."},
    {"key": "motion_angle", "label": "Motion Angle", "type": "int",
     "min": 0, "max": 360, "default": 0,
     "help": "Direction of the motion-blur streak in degrees (0 = horizontal)."},
], build=_build_blur, summary=_summary_blur)


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


def _summary_noise(p):
    return f"{p['type_noise']} · {p['alpha']:.3f} {p['color_mode']}"


_reg("noise", "Noise", [
    {"key": "type_noise", "label": "Noise Type", "type": "choice",
     "options": ["uniform", "gauss", "perlin", "simplex", "opensimplex",
                 "supersimplex", "salt", "pepper", "salt_and_pepper"],
     "default": "gauss",
     "help": "Noise distribution: per-pixel (uniform, gauss), procedural "
             "gradient noise (perlin, simplex; opensimplex and supersimplex "
             "are rendered with the simplex generator in the preview), or "
             "impulse (salt/pepper)."},
    {"key": "alpha", "label": "Intensity", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.005, "default": 0.05, "decimals": 3,
     "help": "Noise strength; bigger values add more noise (salt/pepper "
             "ignore it and hit a random 0-50 % of pixels)."},
    {"key": "color_mode", "label": "Color Mode", "type": "choice",
     "options": ["RGB", "Y only", "UV only"], "default": "RGB",
     "help": "Add noise to all RGB channels, only to brightness (Y), or only "
             "to color (UV)."},
    {"key": "octaves", "label": "Octaves (procedural)", "type": "int",
     "min": 1, "max": 8, "default": 1,
     "help": "Procedural noise only: number of layers summed; each layer has "
             "half the strength of the last and its frequency times the "
             "lacunarity, so it is coarser when lacunarity is below 1 and "
             "finer when above 1."},
    {"key": "frequency", "label": "Frequency (procedural)", "type": "float",
     "min": 0.01, "max": 5.0, "step": 0.01, "default": 0.8, "decimals": 2,
     "help": "Procedural noise only: base frequency in cycles per pixel, as "
             "in the destroyer; 0.02-0.1 gives soft blobs, 0.5 and up "
             "(default 0.8) gives per-pixel grain."},
    {"key": "lacunarity", "label": "Lacunarity (procedural)", "type": "float",
     "min": 0.01, "max": 5.0, "step": 0.01, "default": 0.4, "decimals": 2,
     "help": "Procedural noise only: frequency ratio from one octave to the "
             "next; below 1 (default 0.4) each octave is coarser than the "
             "last, above 1 finer."},
], build=_build_noise, summary=_summary_noise)


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


def _summary_compress(p):
    alg = p["algorithm"]
    if alg == "jpeg":
        return f"jpeg q{p['quality']} {p['jpeg_sampling']}"
    if alg == "webp":
        return f"webp q{p['quality']}"
    label = CODEC_QUALITY_PROFILES[alg]["label"].lower()
    return f"{alg} {label}{p['quality']} {p['video_sampling']}"


_reg("compress", "Compression", [
    {"key": "algorithm", "label": "Algorithm", "type": "choice",
     "options": ["jpeg", "webp", "h264", "hevc", "mpeg2", "mpeg4", "vp9"],
     "default": "jpeg",
     "help": "Codec to round-trip the image through: still-image (JPEG, WebP) "
             "or a single intra-coded video frame."},
    {"key": "quality", "label": "Quality", "type": "int",
     "min": 1, "max": 100, "default": 80,
     "profiles": {"source": "algorithm", "map": CODEC_QUALITY_PROFILES},
     "help": "Codec quality: for JPEG/WebP bigger is cleaner; for CRF and "
             "QScale (video codecs) bigger means stronger artifacts."},
    {"key": "jpeg_sampling", "label": "JPEG Sampling", "type": "choice",
     "options": ["4:4:4", "4:4:0", "4:2:2", "4:2:0", "4:1:1"],
     "default": "4:2:0",
     "help": "JPEG chroma subsampling; lower chroma resolution (4:2:0, 4:1:1) "
             "bleeds and blocks the colors more."},
    {"key": "video_sampling", "label": "Video Sampling", "type": "choice",
     "options": ["444", "422", "420"], "default": "420",
     "help": "Video codec chroma format; 420 halves color resolution both "
             "ways, 444 keeps full color."},
], build=_build_compress, summary=_summary_compress)


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


def _summary_resize(p):
    return f"{p['alg_lq']} ×{p['scale']}"


_reg("resize", "Resize", [
    {"key": "alg_lq", "label": "LQ Algorithm", "type": "choice",
     "options": _RESIZE_ALGS, "default": "lanczos",
     "help": "Filter used to downscale the LQ image; sharper filters "
             "(lanczos, catrom) ring, softer ones (bspline, gauss) blur."},
    {"key": "alg_hq", "label": "HQ Algorithm", "type": "choice",
     "options": _RESIZE_ALGS, "default": "lanczos",
     "help": "Filter used to resize HQ to its target size; soft filters "
             "(mitchell, bspline, gauss) blur HQ even at the same size."},
    {"key": "scale", "label": "Scale Factor", "type": "int",
     "min": 1, "max": 8, "default": 4, "config_type": "raw",
     "help": "Downscale factor from HQ to LQ; bigger values give a smaller, "
             "blockier LQ."},
    {"key": "spread", "label": "Spread", "type": "float",
     "min": 1.0, "max": 4.0, "step": 0.1, "default": 1.0, "decimals": 1,
     "help": "Extra divisor applied to both HQ and LQ sizes; bigger values "
             "shrink both images further."},
    {"key": "color_fix", "label": "Color Fix", "type": "bool", "default": False,
     "help": "Stretch input levels 0-254 to full range on both images after "
             "resizing (slightly brighter, near-white clips)."},
    {"key": "gamma_correction", "label": "Gamma Correction", "type": "bool",
     "default": False,
     "help": "Resize in linear light instead of gamma space, which keeps "
             "bright and dark detail balanced."},
], build=_build_resize, summary=_summary_resize)


# ──────────────────────────────────────────────
# Color Levels
# ──────────────────────────────────────────────
def _summary_color(p):
    return f"{p['low']}–{p['high']} · γ {p['gamma']:.2f}"


_reg("color", "Color Levels", [
    {"key": "high", "label": "Output High", "type": "int",
     "min": 0, "max": 255, "default": 255,
     "help": "Brightest output level; lower values dim highlights and reduce "
             "contrast."},
    {"key": "low", "label": "Output Low", "type": "int",
     "min": 0, "max": 255, "default": 0,
     "help": "Darkest output level; bigger values lift blacks to gray."},
    {"key": "gamma", "label": "Gamma", "type": "float",
     "min": 0.1, "max": 5.0, "step": 0.01, "default": 1.0, "decimals": 2,
     "help": "Midtone curve; values away from 1 brighten or darken the "
             "midtones."},
], summary=_summary_color)


# ──────────────────────────────────────────────
# Halo (Unsharp Mask / Oversharpening)
# ──────────────────────────────────────────────
def _summary_halo(p):
    return f"{p['type_halo']} · σ {p['kernel']:.2f} ×{p['amount']:.2f}"


_reg("halo", "Halo / Sharpen", [
    {"key": "type_halo", "label": "Type", "type": "choice",
     "options": ["unsharp_mask", "unsharp_gray", "unsharp_halo"],
     "default": "unsharp_mask",
     "help": "Sharpening variant: plain unsharp mask, mask computed on "
             "luminance, or pure-white halos only where edges are strongest."},
    {"key": "kernel", "label": "Sigma", "type": "float",
     "min": 0.1, "max": 20.0, "step": 0.05, "default": 1.0, "decimals": 2,
     "help": "Blur radius of the unsharp mask; bigger values give wider "
             "halos around edges."},
    {"key": "amount", "label": "Amount", "type": "float",
     "min": 0.0, "max": 10.0, "step": 0.05, "default": 1.0, "decimals": 2,
     "help": "Sharpening strength; bigger values give brighter, harsher "
             "halos."},
    {"key": "threshold", "label": "Threshold (0-255)", "type": "float",
     "min": 0.0, "max": 255.0, "step": 1.0, "default": 0.0, "decimals": 0,
     "help": "Minimum edge contrast that gets sharpened; bigger values leave "
             "flat and low-contrast areas alone."},
], summary=_summary_halo)


# ──────────────────────────────────────────────
# Dithering
# ──────────────────────────────────────────────
def _summary_dithering(p):
    return f"{p['dithering_type']} · {p['color_ch']} levels"


_reg("dithering", "Dithering", [
    {"key": "dithering_type", "label": "Algorithm", "type": "choice",
     "options": ["quantize", "floydsteinberg", "jarvisjudiceninke", "stucki",
                 "atkinson", "burkes", "sierra", "tworowsierra", "sierraLite",
                 "order", "riemersma"],
     "default": "floydsteinberg",
     "help": "Plain quantization, an error-diffusion kernel, ordered (Bayer) "
             "dithering, or Riemersma's space-filling curve."},
    {"key": "color_ch", "label": "Color Levels", "type": "int",
     "min": 2, "max": 64, "default": 8,
     "help": "Levels kept per channel; smaller values give coarser colors and "
             "more visible dither."},
    {"key": "map_size", "label": "Map Size (ordered)", "type": "int",
     "min": 2, "max": 16, "default": 4,
     "help": "Ordered dithering only: Bayer matrix size; bigger values give a "
             "larger, finer repeating pattern."},
    {"key": "history", "label": "History (riemersma)", "type": "int",
     "min": 2, "max": 64, "default": 10,
     "help": "Riemersma only: number of past pixels whose error is carried "
             "along the curve."},
    {"key": "ratio", "label": "Decay Ratio (riemersma)", "type": "float",
     "min": 0.01, "max": 0.99, "step": 0.01, "default": 0.5, "decimals": 2,
     "help": "Riemersma only: how fast old errors fade; bigger values keep "
             "older errors longer."},
], summary=_summary_dithering)


# ──────────────────────────────────────────────
# Saturation
# ──────────────────────────────────────────────
_reg("saturation", "Saturation", [
    {"key": "rand", "label": "Saturation Multiplier", "type": "float",
     "min": 0.0, "max": 2.0, "step": 0.01, "default": 0.5, "decimals": 2,
     "config_key": "rand",
     "help": "Multiplies color saturation; below 1 washes colors out (0 = "
             "gray), above 1 makes them more vivid."},
])


# ──────────────────────────────────────────────
# Pixelate
# ──────────────────────────────────────────────
_reg("pixelate", "Pixelate", [
    {"key": "size", "label": "Pixel Block Size", "type": "float",
     "min": 1.0, "max": 32.0, "step": 0.5, "default": 4.0, "decimals": 1,
     "help": "Size of the square blocks in pixels; bigger values give "
             "coarser mosaics, 1 does nothing."},
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
     "min": 2, "max": 2000, "default": 200,
     "help": "Size of the sine pattern in pixels; bigger values give wider, "
             "slower brightness waves."},
    {"key": "alpha", "label": "Amplitude", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.1, "decimals": 2,
     "help": "Strength of the brightness stripes; bigger values make them "
             "more visible."},
    {"key": "bias", "label": "Bias", "type": "float",
     "min": 0.0, "max": 2.0, "step": 0.01, "default": 1.0, "decimals": 2,
     "help": "Overall brightness multiplier under the stripes; 1 keeps "
             "brightness, bigger values brighten."},
    {"key": "vertical", "label": "Vertical", "type": "bool", "default": False,
     "help": "Run the stripes vertically instead of horizontally."},
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


def _summary_screentone(p):
    return f"{p['halftone_type']} · {p['dot_type']} {p['dot_size']} px"


_reg("screentone", "Screentone", [
    {"key": "halftone_type", "label": "Halftone Mode", "type": "choice",
     "options": ["cmyk", "rgb", "hsv", "not_rot", "gray"], "default": "rgb",
     "help": "Color model the halftone screens are built in; gray outputs a "
             "single-channel image."},
    {"key": "dot_size", "label": "Dot Size", "type": "int",
     "min": 2, "max": 32, "default": 7,
     "help": "Halftone cell size in pixels; bigger values give larger, more "
             "visible dots."},
    {"key": "dot_type", "label": "Dot Shape", "type": "choice",
     "options": ["circle", "line", "cross", "ellipse"], "default": "circle",
     "help": "Shape of each halftone dot."},
    {"key": "angle", "label": "Angle", "type": "int",
     "min": 0, "max": 180, "default": 0,
     "help": "Rotation of the dot grid in degrees."},
], build=_build_screentone, summary=_summary_screentone)


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


def _summary_subsampling(p):
    return f"{p['sampling']} · {p['down_alg']}/{p['up_alg']}"


_reg("subsampling", "Chroma Subsampling", [
    {"key": "sampling", "label": "Format", "type": "choice",
     "options": ["4:4:4", "4:2:2", "4:2:0", "4:1:1", "4:1:0",
                 "4:4:0", "4:2:1", "4:1:2", "4:1:3"],
     "default": "4:2:0",
     "help": "Chroma subsampling pattern; lower numbers keep less color "
             "resolution (4:4:4 changes nothing)."},
    {"key": "down_alg", "label": "Down Algorithm", "type": "choice",
     "options": _INTERP_ALGS, "default": "linear",
     "help": "Filter used to shrink the color channels."},
    {"key": "up_alg", "label": "Up Algorithm", "type": "choice",
     "options": _INTERP_ALGS, "default": "linear",
     "help": "Filter used to scale the color channels back up; nearest gives "
             "blocky color edges."},
    {"key": "yuv", "label": "YCbCr Standard", "type": "choice",
     "options": ["601", "709", "2020", "240"], "default": "709",
     "help": "RGB-to-YCbCr matrix (BT.601, BT.709, BT.2020, SMPTE 240M)."},
    {"key": "blur", "label": "Chroma Blur Sigma", "type": "float",
     "min": 0.0, "max": 5.0, "step": 0.05, "default": 0.0, "decimals": 2,
     "help": "Extra Gaussian blur on the color channels; bigger values bleed "
             "colors further, 0 turns it off."},
], build=_build_subsampling, summary=_summary_subsampling)


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


def _summary_shift(p):
    return f"{p['shift_type']} · {p['shift_x']}, {p['shift_y']} px"


_reg("shift", "Channel Shift", [
    {"key": "shift_type", "label": "Color Space", "type": "choice",
     "options": ["rgb", "yuv", "cmyk"], "default": "rgb",
     "help": "Which channels move: R/B in RGB, U/V in YUV, or C/Y in CMYK "
             "(in opposite directions)."},
    {"key": "shift_x", "label": "Shift X (px)", "type": "int",
     "min": -50, "max": 50, "default": 2,
     "help": "Horizontal channel offset in pixels; bigger values give wider "
             "color fringes."},
    {"key": "shift_y", "label": "Shift Y (px)", "type": "int",
     "min": -50, "max": 50, "default": 0,
     "help": "Vertical channel offset in pixels; bigger values give taller "
             "color fringes."},
], build=_build_shift, summary=_summary_shift)


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
     "min": 1, "max": 255, "default": 50,
     "help": "Lower Canny gradient threshold; bigger values detect fewer, "
             "stronger edges."},
    {"key": "threshold2_offset", "label": "Threshold 2 Offset", "type": "int",
     "min": 0, "max": 200, "default": 50,
     "help": "Upper threshold = Threshold 1 + this; bigger values keep only "
             "edges that start from very strong gradients."},
    {"key": "aperture_size", "label": "Aperture Size", "type": "choice",
     "options": ["3", "5", "7"], "default": "3",
     "help": "Sobel kernel size; bigger apertures find many more (and "
             "noisier) edges."},
    {"key": "white_bg", "label": "White Background", "type": "bool",
     "default": False,
     "help": "Paint detected edges white instead of black."},
    {"key": "lq_hq", "label": "Replace HQ with LQ", "type": "bool",
     "default": False,
     "help": "Also use the edge-marked result as the HQ image."},
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
     "min": 0.001, "max": 0.5, "step": 0.005, "default": 0.01, "decimals": 3,
     "help": "Lower bound of the texture-noise strength added to HQ; bigger "
             "values add more texture."},
    {"key": "alpha_max", "label": "Alpha Max", "type": "float",
     "min": 0.001, "max": 0.5, "step": 0.005, "default": 0.05, "decimals": 3,
     "help": "Upper bound of the texture-noise strength added to HQ; bigger "
             "values add more texture."},
    {"key": "beta_shape_min", "label": "Beta Shape Min", "type": "float",
     "min": 0.1, "max": 20.0, "step": 0.1, "default": 2.0, "decimals": 1,
     "help": "Lower bound of the Beta distribution shape; bigger shapes give "
             "a tighter, more Gaussian-like noise."},
    {"key": "beta_shape_max", "label": "Beta Shape Max", "type": "float",
     "min": 0.1, "max": 20.0, "step": 0.1, "default": 5.0, "decimals": 1,
     "help": "Upper bound of the Beta distribution shape; bigger shapes give "
             "a tighter, more Gaussian-like noise."},
    {"key": "gray_prob", "label": "Grayscale Probability", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.05, "default": 1.0, "decimals": 2,
     "help": "Chance that the noise is the same on all channels (gray) "
             "instead of colored."},
    {"key": "normalize", "label": "Normalize", "type": "bool", "default": True,
     "help": "Zero-center and scale the noise to unit variance before "
             "applying the strength."},
    {"key": "use_offset", "label": "Use Beta Offset", "type": "bool",
     "default": False,
     "help": "Derive the second Beta shape as the first plus an offset, "
             "which skews the noise to one side."},
    {"key": "offset_min", "label": "Offset Min", "type": "float",
     "min": 0.0, "max": 20.0, "step": 0.1, "default": 1.0, "decimals": 1,
     "help": "Lower bound of the Beta offset; bigger offsets make the noise "
             "more lopsided (mostly dark, rare bright specks)."},
    {"key": "offset_max", "label": "Offset Max", "type": "float",
     "min": 0.0, "max": 20.0, "step": 0.1, "default": 5.0, "decimals": 1,
     "help": "Upper bound of the Beta offset; bigger offsets make the noise "
             "more lopsided (mostly dark, rare bright specks)."},
    {"key": "denoise", "label": "Denoise", "type": "bool", "default": False,
     "help": "Clean the LQ image with Non-Local Means (needs CUDA)."},
    {"key": "denoise_strength", "label": "Denoise Strength", "type": "float",
     "min": 1.0, "max": 150.0, "step": 1.0, "default": 30.0, "decimals": 0,
     "help": "Non-Local Means strength; bigger values smooth the LQ more and "
             "erase finer detail."},
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
     "min": 0.05, "max": 0.50, "step": 0.01, "default": 0.25, "decimals": 2,
     "help": "Color subcarrier frequency in cycles per pixel; bigger values "
             "give finer rainbow stripes."},
    {"key": "chroma_bandwidth", "label": "Chroma Bandwidth", "type": "float",
     "min": 0.01, "max": 0.25, "step": 0.005, "default": 0.08, "decimals": 3,
     "help": "Bandwidth of the chroma decoder; bigger values let more luma "
             "detail leak into color as rainbows."},
    {"key": "intensity", "label": "Intensity", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.05, "default": 1.0, "decimals": 2,
     "help": "Blend between the original and the composite-decoded image."},
    {"key": "phase_alternation", "label": "Phase Alternation (NTSC)",
     "type": "bool", "default": True,
     "help": "Flip the subcarrier phase every line like NTSC, giving "
             "checkerboard dot crawl instead of diagonal stripes."},
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
     "min": 0.05, "max": 1.0, "step": 0.01, "default": 0.5, "decimals": 2,
     "help": "Highest frequency kept; smaller values remove more detail, "
             "bigger values keep more."},
    {"key": "order", "label": "Filter Order", "type": "int",
     "min": 1, "max": 10, "default": 2,
     "help": "Butterworth order; bigger values give a sharper cutoff with "
             "more ringing at edges."},
    {"key": "detail_mask", "label": "Detail Mask", "type": "bool", "default": False,
     "help": "Protect detected line art from the filter so only flat areas "
             "lose detail."},
    {"key": "mask_lines_brz", "label": "Mask Lines Threshold", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.08, "decimals": 2,
     "help": "Detail mask only: line-detection threshold; bigger values "
             "protect fewer lines."},
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
     "options": ["broadcast", "vhs_sp", "vhs_ep"], "default": "broadcast",
     "help": "Signal path: clean broadcast, or VHS tape at SP or (blurrier) "
             "EP speed."},
    {"key": "comb_mode", "label": "Comb Filter", "type": "choice",
     "options": ["2sample", "1h"], "default": "2sample",
     "help": "Luma/chroma separation filter; 1h (line comb) trades dot crawl "
             "for vertical color smearing."},
    {"key": "noise", "label": "Noise", "type": "float",
     "min": 0.0, "max": 0.3, "step": 0.01, "default": 0.05, "decimals": 2,
     "help": "Signal noise level; bigger values give more snow."},
    {"key": "luma_noise", "label": "Luma-Dependent Noise", "type": "float",
     "min": 0.0, "max": 0.15, "step": 0.01, "default": 0.0, "decimals": 2,
     "help": "Extra noise that grows with brightness; bigger values make "
             "bright areas noisier."},
    {"key": "ghost_amplitude", "label": "Ghost Amplitude", "type": "float",
     "min": 0.0, "max": 0.5, "step": 0.01, "default": 0.0, "decimals": 2,
     "help": "Strength of the multipath echo; bigger values give a more "
             "visible ghost image."},
    {"key": "ghost_delay_us", "label": "Ghost Delay (us)", "type": "float",
     "min": 0.5, "max": 10.0, "step": 0.1, "default": 1.5, "decimals": 1,
     "help": "Echo delay in microseconds; bigger values put the ghost "
             "further to the right."},
    {"key": "ghost_phase", "label": "Ghost Phase (deg)", "type": "float",
     "min": 0.0, "max": 360.0, "step": 1.0, "default": 180.0, "decimals": 0,
     "help": "Phase of the echo; 180 gives a dark (inverted) ghost, 0 a "
             "bright one."},
    {"key": "jitter", "label": "Jitter", "type": "float",
     "min": 0.0, "max": 3.0, "step": 0.1, "default": 0.0, "decimals": 1,
     "help": "Random horizontal line wobble in pixels; bigger values make "
             "edges more ragged."},
    {"key": "edge_ringing", "label": "Edge Ringing", "type": "float",
     "min": 0.0, "max": 3.0, "step": 0.1, "default": 0.0, "decimals": 1,
     "help": "Sharpening overshoot along edges; bigger values give stronger "
             "ringing."},
    {"key": "vhs_luma_bw", "label": "VHS Luma BW (MHz)", "type": "float",
     "min": 1.5, "max": 4.2, "step": 0.1, "default": 4.2, "decimals": 1,
     "help": "VHS presets only: luma bandwidth; smaller values blur detail "
             "more."},
    {"key": "color_under_bw", "label": "Color-Under BW (kHz)", "type": "float",
     "min": 200.0, "max": 600.0, "step": 10.0, "default": 500.0, "decimals": 0,
     "help": "VHS presets only: chroma bandwidth; smaller values smear color "
             "further."},
    {"key": "tape_trailing", "label": "Tape Trailing", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.0, "decimals": 2,
     "help": "VHS presets only: streaks trailing to the right of bright "
             "edges; bigger values give longer streaks."},
    {"key": "intensity", "label": "Intensity", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.05, "default": 1.0, "decimals": 2,
     "help": "Blend between the original and the simulated signal."},
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
     "min": 0, "max": 20, "default": 2,
     "help": "Horizontal offset between the two fields; bigger values give "
             "wider combing teeth, 0 does nothing."},
    {"key": "dominant_field", "label": "Dominant Field", "type": "choice",
     "options": ["top", "bottom"], "default": "top",
     "help": "Which field (even or odd lines) stays in place."},
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
     "min": 0.0, "max": 5.0, "step": 0.1, "default": 1.5, "decimals": 1,
     "help": "Sharpening strength; bigger values give brighter overshoot and "
             "darker undershoot at edges."},
    {"key": "cutoff", "label": "Cutoff (fraction of Nyquist)", "type": "float",
     "min": 0.05, "max": 0.8, "step": 0.01, "default": 0.35, "decimals": 2,
     "help": "Frequency where boosting starts; smaller values give wider "
             "halos, bigger values thinner ones."},
    {"key": "order", "label": "Filter Order", "type": "int",
     "min": 1, "max": 5, "default": 2,
     "help": "Filter steepness; bigger values add more ringing ripples."},
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
     "min": 1, "max": 8, "default": 6,
     "help": "Bits kept per channel; smaller values give fewer levels and "
             "wider bands in gradients."},
    {"key": "broadcast_range", "label": "Broadcast Range (16-235)",
     "type": "bool", "default": False,
     "help": "Also squeeze levels into TV range 16-235, which lifts blacks "
             "and dims whites."},
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
     "min": 0.0, "max": 0.3, "step": 0.005, "default": 0.05, "decimals": 3,
     "help": "Grain strength; bigger values give heavier grain (needs CUDA)."},
    {"key": "grain_size", "label": "Grain Size", "type": "float",
     "min": 0.5, "max": 5.0, "step": 0.1, "default": 1.5, "decimals": 1,
     "help": "Spatial scale of the grain; bigger values give coarser, softer "
             "clumps."},
    {"key": "midtone_bias", "label": "Midtone Bias", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.8, "decimals": 2,
     "help": "How much grain concentrates in midtones; 0 is uniform, 1 "
             "spares shadows and highlights."},
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
     "min": -20, "max": 20, "default": 4,
     "help": "Horizontal offset of the ghost copy; bigger values move it "
             "further away."},
    {"key": "shift_y", "label": "Shift Y (px)", "type": "int",
     "min": -20, "max": 20, "default": 0,
     "help": "Vertical offset of the ghost copy; bigger values move it "
             "further away."},
    {"key": "opacity", "label": "Opacity", "type": "float",
     "min": 0.0, "max": 0.5, "step": 0.01, "default": 0.15, "decimals": 2,
     "help": "Blend strength of the ghost; bigger values make the double "
             "image more visible."},
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
     "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.3, "decimals": 2,
     "help": "How much every other line is darkened; 1 makes those lines "
             "black."},
    {"key": "even_lines", "label": "Darken Even Lines",
     "type": "bool", "default": True,
     "help": "Darken the even lines instead of the odd ones."},
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
