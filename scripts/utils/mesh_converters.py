"""
Mesh converters.

Conversion chain:  OBJ  ->  OFF  ->  TIFF (float32 heightmap)

The TIFF step assumes that the terrain is a regular XY grid displaced along Z,
which is what a Geometry Nodes "Grid + Set Position" setup produces.
"""

import os

import numpy as np


# ----------------------------------------------------------------------------
# OBJ -> OFF
# ----------------------------------------------------------------------------
def obj_to_off(obj_path, off_path):
    """
    Convert a Wavefront OBJ file into an OFF (Object File Format) file.

    Only vertex positions and faces are kept. Texture coordinates, normals,
    materials and groups are ignored because the OFF format does not store
    them. Polygons with any number of sides are preserved, since OFF supports
    them natively.

    Parameters
    ----------
    obj_path : str
        Path of the input ``.obj`` file.

    off_path : str
        Path of the output ``.off`` file. Missing parent folders are created.

    Returns
    -------
    tuple[int, int]
        Number of vertices and number of faces written to the OFF file.

    Raises
    ------
    FileNotFoundError
        If ``obj_path`` does not exist.

    ValueError
        If a vertex or face line contains values that cannot be parsed.

    Notes
    -----
    - OBJ face indices are 1-based; they are converted to 0-based indices.
    - Negative OBJ indices (relative to the end of the vertex list) are
      resolved correctly.
    - Face tokens such as ``v/vt/vn`` are reduced to the vertex index ``v``.
    """
    vertices = []
    faces = []

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue

            if parts[0] == "v":
                vertices.append(tuple(float(c) for c in parts[1:4]))

            elif parts[0] == "f":
                indices = []
                for token in parts[1:]:
                    idx = int(token.split("/")[0])
                    # OBJ indices are 1-based; negative ones count from the end.
                    idx = idx - 1 if idx > 0 else len(vertices) + idx
                    indices.append(idx)
                if len(indices) >= 3:
                    faces.append(indices)

    os.makedirs(os.path.dirname(os.path.abspath(off_path)), exist_ok=True)
    with open(off_path, "w", encoding="utf-8") as f:
        f.write("OFF\n")
        f.write(f"{len(vertices)} {len(faces)} 0\n")
        for x, y, z in vertices:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for face in faces:
            f.write(f"{len(face)} " + " ".join(str(i) for i in face) + "\n")

    return len(vertices), len(faces)


# ----------------------------------------------------------------------------
# OFF -> TIFF
# ----------------------------------------------------------------------------
def read_off_vertices(off_path):
    """
    Read the vertex positions of an OFF file.

    Parameters
    ----------
    off_path : str
        Path of the input ``.off`` file.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) with the X, Y, Z coordinates (float64).

    Raises
    ------
    ValueError
        If the file does not start with the ``OFF`` header.

    Notes
    -----
    - Only the vertex block is read; the face block is ignored.
    """
    with open(off_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        if header != "OFF":
            raise ValueError(f"{off_path}: not an OFF file (header: {header!r})")
        n_vertices = int(f.readline().split()[0])
        data = np.loadtxt(f, max_rows=n_vertices, dtype=np.float64)
    return data.reshape(n_vertices, 3)


def vertices_to_height_grid(vertices, decimals=5):
    """
    Arrange the vertices of a regular XY grid into a 2D height array.

    The unique X and Y coordinates define the columns and rows of the grid,
    and the Z coordinate of each vertex becomes the pixel value.

    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (N, 3) with the X, Y, Z coordinates.

    decimals : int, optional
        Number of decimals used to round X and Y before matching them to grid
        columns and rows, to absorb floating-point noise. Default is 5.

    Returns
    -------
    grid : np.ndarray
        Float32 array of shape (ny, nx) with the Z values.
    extent : dict
        Bounding values of the terrain: ``x_min``, ``x_max``, ``y_min``,
        ``y_max``, ``z_min`` and ``z_max``.

    Raises
    ------
    ValueError
        If the vertices do not form a complete regular XY grid.

    Notes
    -----
    - Row 0 corresponds to the maximum Y and column 0 to the minimum X, so the
      array has the same orientation as the source PNG image.
    """
    xs = np.unique(np.round(vertices[:, 0], decimals))
    ys = np.unique(np.round(vertices[:, 1], decimals))

    if len(xs) * len(ys) != len(vertices):
        raise ValueError(
            f"The mesh is not a regular XY grid ({len(xs)} x {len(ys)} unique "
            f"coordinates for {len(vertices)} vertices)."
        )

    ix = np.searchsorted(xs, np.round(vertices[:, 0], decimals))
    iy = np.searchsorted(ys, np.round(vertices[:, 1], decimals))

    grid = np.zeros((len(ys), len(xs)), dtype=np.float32)
    grid[iy, ix] = vertices[:, 2]
    grid = grid[::-1]  # row 0 = max Y, like an image

    extent = {
        "x_min": float(xs[0]), "x_max": float(xs[-1]),
        "y_min": float(ys[0]), "y_max": float(ys[-1]),
        "z_min": float(vertices[:, 2].min()), "z_max": float(vertices[:, 2].max()),
    }
    return np.ascontiguousarray(grid), extent


def resample_bilinear(grid, res_x, res_y):
    """
    Resample a 2D array to a new size using bilinear interpolation.

    Parameters
    ----------
    grid : np.ndarray
        Input array of shape (ny, nx).

    res_x : int
        Width of the output array.

    res_y : int
        Height of the output array.

    Returns
    -------
    np.ndarray
        Float32 array of shape (res_y, res_x). If the requested size equals
        the input size, the input array is returned unchanged.

    Notes
    -----
    - Output pixel centers are mapped onto the vertex grid, so the first and
      last output samples fall half a pixel inside the terrain edges.
    """
    ny, nx = grid.shape
    if (nx, ny) == (res_x, res_y):
        return grid

    # Pixel centers mapped onto the vertex grid [0, n-1].
    u = (np.arange(res_x) + 0.5) / res_x * (nx - 1)
    v = (np.arange(res_y) + 0.5) / res_y * (ny - 1)

    u0 = np.clip(np.floor(u).astype(int), 0, nx - 2)
    v0 = np.clip(np.floor(v).astype(int), 0, ny - 2)
    fu = (u - u0)[None, :]
    fv = (v - v0)[:, None]

    g00 = grid[np.ix_(v0, u0)]
    g01 = grid[np.ix_(v0, u0 + 1)]
    g10 = grid[np.ix_(v0 + 1, u0)]
    g11 = grid[np.ix_(v0 + 1, u0 + 1)]

    out = (g00 * (1 - fu) * (1 - fv) + g01 * fu * (1 - fv)
           + g10 * (1 - fu) * fv + g11 * fu * fv)
    return out.astype(np.float32)


def off_to_tiff(off_path, tiff_path, res=None, normalize=False):
    """
    Convert an OFF terrain into a single-channel float32 TIFF heightmap.

    The vertices are arranged into a 2D grid (see ``vertices_to_height_grid``),
    optionally resampled and normalized, and written with the ``tifffile``
    library.

    Parameters
    ----------
    off_path : str
        Path of the input ``.off`` file (regular XY grid displaced along Z).

    tiff_path : str
        Path of the output ``.tiff`` file. Missing parent folders are created.

    res : int or None, optional
        Side length of the output image (square). If None, the native vertex
        grid size is kept. Default is None.

    normalize : bool, optional
        If True, heights are scaled to the [0, 1] range. If False, the real Z
        values of the terrain are stored. Default is False.

    Returns
    -------
    tuple[int, int]
        Shape (height, width) of the saved image.

    Raises
    ------
    ImportError
        If the ``tifffile`` package is not installed.

    ValueError
        If the OFF file is invalid or not a regular XY grid.

    Notes
    -----
    - The TIFF is 32-bit float, so no height precision is lost (unlike 8/16-bit
      images).
    - The terrain extent (``x_min`` ... ``z_max``) is stored in the TIFF
      metadata (ImageDescription tag).
    - A flat terrain produces an all-zero image when ``normalize`` is True.
    """
    import tifffile as tiff

    vertices = read_off_vertices(off_path)
    grid, extent = vertices_to_height_grid(vertices)

    if res:
        grid = resample_bilinear(grid, res, res)

    if normalize:
        z_min, z_max = float(grid.min()), float(grid.max())
        denom = (z_max - z_min) if z_max != z_min else 1.0
        grid = ((grid - z_min) / denom).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(tiff_path)), exist_ok=True)
    tiff.imwrite(tiff_path, grid.astype(np.float32), metadata=extent)
    return grid.shape