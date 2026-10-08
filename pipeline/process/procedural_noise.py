"""Seeded fractal procedural noise (perlin and simplex) in numpy.

pepeline 1.x's noise() takes no seed, so procedural noise re-rolled on every run. Here every
random choice comes from a numpy Generator built from the caller's seed.

The simplex generator is a port of SimplexNoise from chaiNNer
(backend/src/nodes/impl/noise_functions/simplex.py, GPL-3.0,
https://github.com/chaiNNer-org/chaiNNer), taken from the owner's chaiNNer-C without its
native fast path.
"""

import numpy as np

from ..constants import NOISE_MAP

# Permutation-table size (a power of two); the lattice repeats every _TABLE_SIZE cells
_TABLE_SIZE = 1024
# Gradients as chaiNNer's simplex picks them, one per column: 16 unit vectors evenly spaced
# on the circle (2-D) and the 12 midpoints of a cube's edges (3-D)
_ANGLES = 2 * np.pi * np.arange(16) / 16
_GRADIENTS = {
    2: np.stack((np.cos(_ANGLES), np.sin(_ANGLES))).astype(np.float32),
    3: np.array(
        [
            (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
            (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1),
            (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0),
        ],
        np.float32,
    ).T.copy(),
}
# Output scales by dimension, set to the spread of pepeline 0.3's perlin and simplex (the
# destroyer's); NOISE_MAP adds a per-name amplitude. chaiNNer's own simplex scales
# {2: 50, 3: 39} fill its [0, 1] range ({2: 100, 3: 78} for [-1, 1]).
_PERLIN_SCALE = {2: 2.0, 3: 1.12}
_SIMPLEX_SCALE = {2: 70.0, 3: 57.0}
# Squared radius of the simplex kernel
_RADIUS2 = 0.5


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)


def _perlin(
    x: np.ndarray, y: np.ndarray, perm: np.ndarray, z: np.ndarray | None = None
) -> np.ndarray:
    """Classic gradient noise with the quintic fade, sampled on the grid of column
    coordinates x by row coordinates y: 2-D when z is None, else 3-D with one slice per
    height in z, stacked on the last axis. Corner (i, j, k) takes gradient
    perm[perm[perm[i] + j] + k] (perm[perm[i] + j] in 2-D).

    Blending the two corners of a lattice row along x gives P(x) + Q(x) * fy (+ R(x) * fz in
    3-D), so hashes and x blends are computed once per lattice row in use, not per pixel,
    and each lattice plane in z once for all the slices next to it."""
    mask = perm.size - 1
    dims = 2 if z is None else 3
    gradients = _GRADIENTS[dims][:, perm % _GRADIENTS[dims].shape[1]]
    xi = np.floor(x)
    fx = (x - xi).astype(np.float32)
    xi = xi.astype(np.intp)
    yi = np.floor(y)
    fy = (y - yi).astype(np.float32)[:, None]
    yi = yi.astype(np.intp)
    u = _fade(fx)
    v = _fade(fy)
    rows, row_index = np.unique(np.concatenate((yi, yi + 1)), return_inverse=True)
    rows = (rows & mask)[:, None]
    top, bottom = row_index[: y.size], row_index[y.size :]
    left = (perm[xi & mask] + rows) & mask
    right = (perm[(xi + 1) & mask] + rows) & mask

    def plane(left: np.ndarray, right: np.ndarray) -> tuple:
        """The corners' contributions blended over every pixel: the value on the lattice
        plane and, in 3-D, its change per unit height above the plane."""
        p = ((1 - u) * fx) * gradients[0][left] + (u * (fx - 1)) * gradients[0][right]
        q = (1 - u) * gradients[1][left] + u * gradients[1][right]
        value = p[top] * (1 - v)
        value += q[top] * ((1 - v) * fy)
        value += p[bottom] * v
        value += q[bottom] * (v * (fy - 1))
        if dims == 2:
            return value, None
        r = (1 - u) * gradients[2][left] + u * gradients[2][right]
        return value, r[top] * (1 - v) + r[bottom] * v

    if dims == 2:
        return plane(left, right)[0] * _PERLIN_SCALE[2]
    left = perm[left]
    right = perm[right]
    zi = np.floor(z)
    fz = (z - zi).astype(np.float32)
    zi = zi.astype(np.intp)
    planes = {
        k: plane((left + k) & mask, (right + k) & mask) for k in np.union1d(zi, zi + 1)
    }
    slices = []
    for k, dz in zip(zi, fz):
        (lower, lower_slope), (upper, upper_slope) = planes[k], planes[k + 1]
        w = _fade(dz)
        slices.append(
            (1 - w) * (lower + dz * lower_slope) + w * (upper + (dz - 1) * upper_slope)
        )
    return np.stack(slices, axis=-1) * _PERLIN_SCALE[3]


def _simplex_grid(coords: tuple, perm: np.ndarray) -> np.ndarray:
    """Unscaled simplex noise at the points spanned by broadcasting the per-axis coordinates.

    The n-D path of chaiNNer's SimplexNoise by Alex Dodge (2023), after Stefan Gustavson,
    "Simplex noise demystified" (2005),
    http://staffwww.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf: same skew, gradients,
    vertex order (axes by falling fractional part, the lower axis first on a tie, as argmax),
    hash perm[perm[perm[i] + j] + k] and (r^2 - d^2)^4 kernel. Each skewed coordinate is a
    sum of per-axis terms, each reduced modulo the table size in float64 first, which keeps
    the hash and float32 sub-cell precision at any frequency."""
    dims = len(coords)
    mask = perm.size - 1
    skew = ((dims + 1) ** 0.5 - 1) / dims
    unskew = (1 - (dims + 1) ** -0.5) / dims
    gradients = _GRADIENTS[dims][:, perm % _GRADIENTS[dims].shape[1]]
    base, frac = [], []
    for axis in range(dims):
        skewed = sum(
            ((c * (skew + (axis == other))) % perm.size).astype(np.float32)
            for other, c in enumerate(coords)
        )
        cell = np.floor(skewed)
        frac.append(skewed - cell)
        base.append(cell.astype(np.intp))
    # Offsets from the first vertex: unskewing is linear, so unskew the fractional part
    total = sum(frac) * unskew
    offsets = [f - total for f in frac]
    # How many axes come before each axis by falling fractional part, ties to the lower axis
    rank = [
        sum(
            frac[other] >= frac[axis] if other < axis else frac[other] > frac[axis]
            for other in range(dims)
            if other != axis
        )
        for axis in range(dims)
    ]
    contributions = []
    for vertex in range(dims + 1):
        # Vertex v is one step along each of the v axes that come first
        step = [r < vertex for r in rank] if 0 < vertex < dims else [vertex // dims] * dims
        index = (base[0] + step[0]) & mask
        for axis in range(1, dims):
            index = (perm[index] + base[axis] + step[axis]) & mask
        d = [offsets[axis] - step[axis] + vertex * unskew for axis in range(dims)]
        falloff = np.maximum(_RADIUS2 - sum(c * c for c in d), 0)
        falloff *= falloff
        falloff *= falloff
        contributions.append(
            falloff * sum(gradients[axis][index] * d[axis] for axis in range(dims))
        )
    return sum(contributions)


def _simplex(
    x: np.ndarray, y: np.ndarray, perm: np.ndarray, z: np.ndarray | None = None
) -> np.ndarray:
    """Simplex noise on the grid of column coordinates x by row coordinates y: 2-D when z
    is None, else 3-D with one slice per height in z, stacked on the last axis."""
    if z is None:
        return _simplex_grid((x, y[:, None]), perm) * _SIMPLEX_SCALE[2]
    slices = [_simplex_grid((x, y[:, None], height), perm) for height in z]
    return np.stack(slices, axis=-1) * _SIMPLEX_SCALE[3]


_GENERATORS = {"perlin": _perlin, "simplex": _simplex}


def fractal_noise(
    shape: tuple, noise_type: str, octaves: int, frequency: float, lacunarity: float,
    seed: int,
) -> np.ndarray:
    """Fractal noise in [-1, 1] as float32; the same seed gives the same output.

    Octave i samples at frequency * lacunarity**i cycles per pixel (the destroyer's
    pepeline 0.3 scale: 0.8 is per-pixel grain, 0.02-0.1 gives blobs) with weight 0.5**i,
    normalised by the weight sum. A 3-D shape samples one 3-D field, channel c at height
    c * frequency, as pepeline 0.3 does: channels are near copies at low frequencies and
    independent at high ones. Every octave has its own permutation table and a sub-cell
    offset, so integer frequencies do not sample only lattice points (where gradient noise
    is 0)."""
    generator, stream, amplitude = NOISE_MAP[noise_type]
    generate = _GENERATORS[generator]
    rng = np.random.default_rng((seed, stream))
    noise = np.zeros(shape, np.float32)
    for i in range(octaves):
        step = frequency * lacunarity**i
        offset_x, offset_y, offset_z = rng.random(3)
        perm = rng.permutation(_TABLE_SIZE)
        z = None if len(shape) == 2 else np.arange(shape[2]) * step + offset_z
        noise += 0.5**i * generate(
            np.arange(shape[1]) * step + offset_x, np.arange(shape[0]) * step + offset_y,
            perm, z,
        )
    noise *= amplitude / sum(0.5**i for i in range(octaves))
    return np.clip(noise, -1, 1, out=noise)
