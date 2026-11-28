"""
Entry point of the PNG -> 3D terrain pipeline.

    PNG heightmaps -> Blender Geometry Nodes -> OBJ -> OFF -> TIFF

1. Launches Blender in background with your .blend file; the Blender script
   loads each PNG into the Geometry Nodes and exports the terrain as OBJ.
2. Converts every OBJ file into an OFF file.
3. Converts every OFF file into a float32 TIFF heightmap (using tifffile).

Output layout:
    OUTPUT_DIR/obj/*.obj
    OUTPUT_DIR/off/*.off
    OUTPUT_DIR/tiff/*.tiff
"""

import os
import sys
import glob
import subprocess

from tqdm import tqdm

from utils.mesh_converters import obj_to_off, off_to_tiff
from utils.constants import (
    BLENDER_SCRIPT_CONVERTER, 
    BLEND_FILE_VIEWER, 
    BLENDER_EXECUTABLE,
    VIEWER_BLEND_IMG_IN,
    VIEWER_BLEND_IMG_OUT
    )

# ----------------------------------------------------------------------------
# SETTINGS - adapt these to your project
# ----------------------------------------------------------------------------

TIFF_RES = None                  # None = native vertex grid; or an int (e.g. 512)
TIFF_NORMALIZE = True           # False = real Z values, True = scaled to [0, 1]


def run_blender(total_images, obj_dir):
    """
    Launch Blender in background mode and export the OBJ files.

    Blender runs ``BLENDER_SCRIPT`` on ``BLEND_FILE``. Its output is streamed
    line by line: every ``BLENDER_IMAGE_DONE`` message advances the progress
    bar, while other relevant lines are printed for debugging and noisy ones
    (render progress, timing, etc.) are filtered out.

    Parameters
    ----------
    total_images : int
        Number of PNG files to process (used as the progress bar total).

    obj_dir : str
        Folder where Blender writes the OBJ files.

    Returns
    -------
    int
        The exit code of the Blender process (0 means success).

    Raises
    ------
    FileNotFoundError
        If the Blender executable cannot be found.

    Notes
    -----
    - Blender is executed with ``-b`` (background) and ``-P`` (Python script).
    - ``INPUT_DIR`` and ``obj_dir`` are passed to the script after ``--``.
    - Standard error is merged into standard output.
    """
    command = [
        BLENDER_EXECUTABLE,
        "-b", BLEND_FILE_VIEWER,
        "-P", BLENDER_SCRIPT_CONVERTER,
        "--", VIEWER_BLEND_IMG_IN, obj_dir,
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )

    with tqdm(total=total_images, desc="PNG -> OBJ", unit="img") as pbar:
        for line in process.stdout:
            line_clean = line.strip()

            if "BLENDER_IMAGE_DONE" in line_clean:
                pbar.update(1)

            elif line_clean:
                spam_keywords = ["Fra:", "Time:", "Saved:", "Blender quit", "Read blend:", "Sample"]
                if any(spam in line_clean for spam in spam_keywords):
                    continue
                tqdm.write(f"[BLENDER] {line_clean}")

    process.wait()
    return process.returncode


def main():
    """
    Run the whole PNG -> OBJ -> OFF -> TIFF pipeline.

    The function validates the required paths, creates the output folders,
    runs Blender to produce the OBJ files, and then converts them to OFF and
    TIFF with progress bars.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If a required file or folder is missing, no PNG is found, Blender is
        not installed, Blender fails, or no OBJ file is produced.

    Notes
    -----
    - Errors in a single OFF -> TIFF conversion are printed and do not stop
      the remaining files.
    - Output folders: ``OUTPUT_DIR/obj``, ``OUTPUT_DIR/off`` and
      ``OUTPUT_DIR/tiff``.
    """
    print("Start PNG -> OBJ -> OFF -> TIFF pipeline...")

    for label, path in (("Blender .blend file", BLEND_FILE_VIEWER),
                        ("Blender script", BLENDER_SCRIPT_CONVERTER),
                        ("Input folder", VIEWER_BLEND_IMG_IN)):
        if not os.path.exists(path):
            print(f"Error: {label} not found: {path}")
            sys.exit(1)

    pngs = sorted(glob.glob(os.path.join(VIEWER_BLEND_IMG_IN, "*.png")))
    if not pngs:
        print(f"Error: no PNG files found in {VIEWER_BLEND_IMG_IN}")
        sys.exit(1)

    obj_dir = os.path.join(VIEWER_BLEND_IMG_OUT, "obj")
    off_dir = os.path.join(VIEWER_BLEND_IMG_OUT, "off")
    tiff_dir = os.path.join(VIEWER_BLEND_IMG_OUT, "tiff")
    for d in (obj_dir, off_dir, tiff_dir):
        os.makedirs(d, exist_ok=True)

    # --- Step 1: Blender (PNG -> OBJ) ---
    try:
        return_code = run_blender(len(pngs), obj_dir)
    except FileNotFoundError:
        print("Error: Blender not found. Check BLENDER_EXECUTABLE or add Blender to the system PATH.")
        sys.exit(1)

    if return_code != 0:
        print(f"\nError: Blender exited with code {return_code}")
        sys.exit(return_code)

    obj_files = sorted(glob.glob(os.path.join(obj_dir, "*.obj")))
    if not obj_files:
        print("Error: Blender finished but no OBJ files were produced.")
        sys.exit(1)

    # --- Step 2: OBJ -> OFF ---
    off_files = []
    for obj_path in tqdm(obj_files, desc="OBJ -> OFF", unit="mesh"):
        stem = os.path.splitext(os.path.basename(obj_path))[0]
        off_path = os.path.join(off_dir, f"{stem}.off")
        obj_to_off(obj_path, off_path)
        off_files.append(off_path)

    # --- Step 3: OFF -> TIFF ---
    for off_path in tqdm(off_files, desc="OFF -> TIFF", unit="mesh"):
        stem = os.path.splitext(os.path.basename(off_path))[0]
        tiff_path = os.path.join(tiff_dir, f"{stem}.tiff")
        try:
            off_to_tiff(off_path, tiff_path, res=TIFF_RES, normalize=TIFF_NORMALIZE)
        except Exception as e:
            tqdm.write(f"[ERROR] {stem}: {e}")

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()