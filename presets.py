"""
Pipeline presets for the WTP Degradation Preview GUI.

- save_preset / load_preset: the panel state as JSON, {"version": 1, "steps": [...]}
- to_hcl: pipeline configs as wtp_dataset_destroyer `degradation { ... }` blocks
"""

import json

PRESET_VERSION = 1


def save_preset(path, state):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": PRESET_VERSION, "steps": state}, f, indent=2)
        f.write("\n")


def load_preset(path):
    """Return the step list of a preset file; raises ValueError if it is not one."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        raise ValueError("not a pipeline preset (expected a \"steps\" list)")
    if data.get("version") != PRESET_VERSION:
        raise ValueError(f"unsupported preset version {data.get('version')!r} "
                         f"(this build reads version {PRESET_VERSION})")
    if not all(isinstance(step, dict) for step in data["steps"]):
        raise ValueError("every entry in \"steps\" must be an object")
    return data["steps"]


# ──────────────────────────────────────────────
# HCL export (syntax of wtp_dataset_destroyer/configs/full.hcl)
#   dict            → key = { ... }        (e.g. shift `rgb`)
#   list of dicts   → repeated key { ... }  (e.g. screentone `color`, its `dot`s)
# ──────────────────────────────────────────────

def to_hcl(configs):
    return "".join(_block("degradation", config, 0) + "\n\n" for config in configs)


def _block(name, mapping, depth):
    pad = "  " * depth
    return "\n".join([f"{pad}{name} {{", *_body(mapping, depth + 1), f"{pad}}}"])


def _body(mapping, depth):
    pad = "  " * depth
    lines = []
    for key, value in mapping.items():
        if isinstance(value, dict):
            lines += [f"{pad}{key} = {{", *_body(value, depth + 1), f"{pad}}}"]
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            lines += [_block(key, item, depth) for item in value]
        else:
            lines.append(f"{pad}{key} = {_literal(value)}")
    return lines


def _literal(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} value {value!r} as HCL")
