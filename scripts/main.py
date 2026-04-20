import sys, subprocess, os
from utils.utils_method import path_exists
from utils.constants import BLENDER_SCRIPT, BLEND_FILE, BLENDER_EXECUTABLE

def main():
    """
    Entry point of the dataset generation pipeline.

    This function validates required file paths, constructs the Blender execution
    command, and launches Blender in background mode to generate terrain data
    according to the provided script.

    The function handles execution results, printing standard output, errors,
    and return codes. It also manages exceptions related to subprocess execution
    and missing executables.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If required files (Blender executable, .blend file, or script) are not found.

    subprocess.CalledProcessError
        If Blender execution fails (non-zero exit code).

    FileNotFoundError
        If the Blender executable is not found in the system PATH.

    Notes
    -----
    - Blender is executed in background mode using the '-b' flag.
    - The Python script is passed using the '-P' flag.
    - Standard output and error streams are captured and printed for debugging.
    """

    print("Start generation dataset pipeline...")

    if not path_exists(BLEND_FILE):
        print(f"Error: File Blender not found {BLEND_FILE}")
        sys.exit(1)
    
    if not path_exists(BLENDER_SCRIPT):
        print(f"Error: File Blender not found {BLENDER_SCRIPT}")
        sys.exit(1)

    command = [
        BLENDER_EXECUTABLE,
        "-b", BLEND_FILE,
        "-P", BLENDER_SCRIPT
    ]

    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Generation completed successfully!")

        for line in result.stdout:
            print(f"[BLENDER]: {line}", end="")

        if not result.stdout == '':
            print("Output:")
            print(result.stdout)
        if not result.stderr == '':
            print("Errors:")
            print(result.stderr)

        print("Return Code:")
        print(result.returncode)

    except subprocess.CalledProcessError as e:
        print(f"Error during Blender execution. Exit Code: {e.returncode}")

        print("\n--- Blender Detail Error ---")
        print(e.stderr) 
    
        print("\n--- OUTPUT ---")
        print(e.stdout)
    except FileNotFoundError:
        print("Error: Blender not found. Make sure you have it installed and added in the system PATH")


if __name__ == "__main__":
    main()