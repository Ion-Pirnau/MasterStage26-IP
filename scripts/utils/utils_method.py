import os
from pathlib import Path

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
            f.write(f"{len(vertices)} len({faces}) 0\n")
            for v in vertices:
                f.write(f"{v}\n")
            for face in faces:
                f.write(f"{face}\n")
        
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
        if (parent / 'README.md').exists():
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
