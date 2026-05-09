import os
from pathlib import Path
import numpy as np
import tifffile as tiff

def obj_to_off(obj_path: str, off_path: str) -> bool:

    """
    Converts a 3D geometry file from OBJ format to OFF format.

    Reads vertices and faces from the source file, remaps indices 
    (from 1-based to 0-based), and writes the structure according to the OFF standard.

    Args:
        obj_path (str): The full or relative path to the source .obj file.
        off_path (str): The path where the generated .off file will be saved.

    Returns:
        bool: True if the conversion was successful, False otherwise.

    Raises:
        FileNotFoundError: If the input file does not exist.
        IOError: If an error occurs during file writing.
    """

    vertices = []
    faces = []

    try:
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append(line.strip()[2:])
                elif line.startswith('f '):
                    face = [str(int(x.split('/')[0]) - 1) for x in line.strip().split()[1:]]
                    faces.append(f"{len(face)} " + " ".join(face))

        with open(off_path, 'w') as f:
            f.write("OFF\n")
            f.write(f"{len(vertices)} {len(faces)} 0\n")
            for v in vertices:
                f.write(f"{v}\n")
            for face in faces:
                f.write(f"{face}\n")
        
        return True
    
    except (FileNotFoundError, IOError) as e:
        print(f"Error during conversion: {e}")
        return False


def geotiff_to_off(tif_path, out_path, downsample=1, xy_resolution_meters=10.0, z_scale=1.0) -> bool:
    """
    Reads a GeoTIFF file and generates a 3D mesh exported in OFF format.

    :param tif_path: Path to the input GeoTIFF file.
    :param obj_path: Path to the output OBJ file.
    :param downsample: Sampling factor used to reduce mesh resolution 
                       (e.g., 2 halves the number of sampled pixels). 
                       Useful for processing very large datasets.
    :param xy_resolution_meters: Spatial resolution of each pixel in meters 
                                 (e.g., 1/3 arc-second DEM ≈ 10 meters per pixel), 
                                 used to scale the X and Y coordinates.
    :param z_scale: Vertical scaling factor applied to elevation values 
                    in order to exaggerate or reduce terrain height.

    Returns:
        bool: True if the conversion was successful, False otherwise.

    Raises:
        FileNotFoundError: If the input file does not exist.
        IOError: If an error occurs during file writing.
    """

    if not path_exists(tif_path):
        print(f"Error: File '{tif_path}' does not exist.")
        return False

    if downsample < 1:
        print("Error: downsample must be >= 1.")
        return False

    print(f"Reading file {tif_path}...")

    try:
        elevation = tiff.imread(tif_path)[::downsample, ::downsample]

        h, w = elevation.shape
        num_vertices = h * w
        print(f"Generation mesh from Grid: {w}x{h} ({w * h} vertices)...")

        # 1. GENERATION VERTICES (X, Y, Z)
        x = np.arange(w) * (xy_resolution_meters * downsample)
        y = np.arange(h) * (xy_resolution_meters * downsample)
        
        y = y[::-1] 
        
        xx, yy = np.meshgrid(x, y)

        vertices = np.column_stack((
            xx.ravel(), 
            yy.ravel(), 
            elevation.ravel() * z_scale
        ))

        # 2. TRIANGLES GENERATION (FACES)
        r, c = np.mgrid[0:h-1, 0:w-1]
        
        top_left = r * w + c
        top_right = top_left + 1
        bottom_left = (r + 1) * w + c
        bottom_right = bottom_left + 1

        # Triangle 1: Top-Left, Bottom-Left, Bottom-Right
        tri1 = np.stack((top_left, bottom_left, bottom_right), axis=-1).reshape(-1, 3)
        # Triangle 2: Top-Left, Bottom-Right, Top-Right
        tri2 = np.stack((top_left, bottom_right, top_right), axis=-1).reshape(-1, 3)
        
        faces = np.vstack((tri1, tri2))
        num_faces = len(faces)

        # 3. SAVE IN FORMAT OFF
    
        print(f"Saved in: {out_path} ...")

        with open(out_path, 'w') as f:

            f.write("OFF\n")
            f.write(f"{num_vertices} {num_faces} 0\n")
            
            for v in vertices:
                f.write(f"{v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
            
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

        print("Conversion completed successfully.")
        return True

    except (FileNotFoundError, IOError) as e:
        print(f"Error during conversion: {e}")
        return False
    

def geotiff_to_obj(tif_path, out_path, grid_size=10.0, downsample=1, xy_resolution_meters=10.0, z_scale=1.0) -> bool:
    """
    Reads a GeoTIFF file and generates a 3D mesh exported in OBJ format.

    :param tif_path: Path to the input GeoTIFF file.
    :param obj_path: Path to the output OBJ file.
    :param grid_size: Target size for the largest dimension of the mesh (e.g., 10 units).
    :param downsample: Sampling factor used to reduce mesh resolution 
                       (e.g., 2 halves the number of sampled pixels). 
                       Useful for processing very large datasets.
    :param xy_resolution_meters: Spatial resolution of each pixel in meters 
                                 (e.g., 1/3 arc-second DEM ≈ 10 meters per pixel), 
                                 used to scale the X and Y coordinates.
    :param z_scale: Vertical scaling factor applied to elevation values 
                    in order to exaggerate or reduce terrain height.

    Returns:
        bool: True if the conversion was successful, False otherwise.

    Raises:
        FileNotFoundError: If the input file does not exist.
        IOError: If an error occurs during file writing.
    """

    if not path_exists(tif_path):
        print(f"Error: File '{tif_path}' does not exist.")
        return False

    if downsample < 1:
        print("Error: downsample must be >= 1.")
        return False

    print(f"Reading file {tif_path}...")

    try:
        elevation = tiff.imread(tif_path)[::downsample, ::downsample]

        h, w = elevation.shape
        print(f"Generation mesh from Grid: {w}x{h} ({w * h} vertices)...")

         # NORMALIZATION AND CENTERING
        real_width = (w - 1) * (xy_resolution_meters * downsample)
        real_height = (h - 1) * (xy_resolution_meters * downsample)


        max_real_dim = max(real_width, real_height)
        global_scale = grid_size / max_real_dim

        target_width = real_width * global_scale
        target_height = real_height * global_scale

        x = np.linspace(-target_width / 2, target_width / 2, w)
        y = np.linspace(target_height / 2, -target_height / 2, h)

        xx, yy = np.meshgrid(x, y)

        min_elevation = np.nanmin(elevation)
        normalized_elevation = (elevation - min_elevation) * global_scale * z_scale

        # 1. GENERATION VERTICES (X, Y, Z)
        vertices = np.column_stack((
            xx.ravel(), 
            yy.ravel(), 
            normalized_elevation.ravel()
        ))

        # 2. TRIANGLES GENERATION (FACES)
        r, c = np.mgrid[0:h-1, 0:w-1]
        
        top_left = r * w + c
        top_right = top_left + 1
        bottom_left = (r + 1) * w + c
        bottom_right = bottom_left + 1

        # Triangle 1: Top-Left, Bottom-Left, Bottom-Right
        tri1 = np.stack((top_left, bottom_left, bottom_right), axis=-1).reshape(-1, 3)
        # Triangle 2: Top-Left, Bottom-Right, Top-Right
        tri2 = np.stack((top_left, bottom_right, top_right), axis=-1).reshape(-1, 3)
        
        faces = np.vstack((tri1, tri2)) + 1

        # 3. SAVE IN FORMAT OBJ
    
        print(f"Saved in: {out_path} ...")

        with open(out_path, 'w') as f:

            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")

        print("Conversion completed successfully.")
        return True

    except (FileNotFoundError, IOError) as e:
        print(f"Error during conversion: {e}")
        return False




def get_project_root() -> Path:
    """
    Returns the project root directory by searching for a marker file.

    Returns:
        Path: The absolute path to the project root.
    """
    current_path = Path(__file__).resolve()

    for parent in [current_path] + list(current_path.parents):
        if (parent / '.gitignore').exists():
            return parent
    
    return current_path.parent


def get_full_path(*path_parts: str) -> str:
    """
    Joins the project root with the provided folders and/or file.

    Args:
        *path_parts (str): One or more strings representing folders or a filename.
                           Example: "data", "models", "cube.obj"

    Returns:
        str: The absolute path as a string.
    """

    root = get_project_root()
    
    full_path = root.joinpath(*path_parts)

    return str(full_path)


def path_exists(path: str) -> bool:
    """
    Checks if a given path exists on the file system.

    Args:
        path (str): The full path to the file or directory.

    Returns:
        bool: True if the path exists, False otherwise.
    """
    return Path(path).exists()


def is_valid_file(path: str) -> bool:
    """
    Checks if a path exists and points specifically to a file (not a folder).

    Args:
        path (str): The path to check.

    Returns:
        bool: True if it is an existing file.
    """
    p = Path(path)
    return p.exists() and p.is_file()


def mesh_from_asc(path_asc):
    header = {}

    with open(path_asc, 'r') as f:
        for _ in range(6):
            line = f.readline().split()
            header[line[0].lower()] = float(line[1])
    
    cols = int(header['ncols'])
    rows = int(header['nrows'])
    x_ll = header['xllcorner']
    y_ll = header['yllcorner']
    cell_size = header['cellsize']
    nodata = header.get('nodata_value', -9999)

    z_grid = np.loadtxt(path_asc, skiprows=6)

    x = x_ll + np.arange(cols) * cell_size

    y = y_ll + (rows - 1 - np.arange(rows)) * cell_size

    x_grid, y_grid = np.meshgrid(x,y)

    print(x_grid)
    print(y_grid)

    vertices = np.stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten()), axis=1)

    faces = []

    for r in range(rows - 1):
        for c in range(cols - 1):
            v_top_left = r * cols + c
            v_top_right = v_top_left + 1
            v_bottom_left = (r + 1) * cols + c
            v_bottom_right = v_bottom_left + 1

            faces.append([v_top_left, v_bottom_left, v_top_right])
            faces.append([v_top_right, v_bottom_left, v_bottom_right])

    return vertices, np.array(faces)


def save_off(vertices, faces, filename):
    with open(filename, 'w') as f:
        f.write("OFF\n")

        f.write(f"{len(vertices)} {len(faces)} 0\n")

        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")