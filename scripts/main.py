import sys, subprocess
from utils.utils_method import path_exists, load_config
from utils.constants import BLENDER_SCRIPT, BLEND_FILE, BLENDER_EXECUTABLE, GENERATION_MODE_RANDOM
from tqdm import tqdm

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

    config = load_config()
    settings = config["dataset_settings"]
    mode = settings.get("generation_mode", "fixed")
    TOTAL_IMAGES = settings["num_terrains_to_generate"] if mode == GENERATION_MODE_RANDOM else 1

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
        process = subprocess.Popen(command, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
            )

        with tqdm(total=TOTAL_IMAGES, desc="Generating Maps", unit="img") as pbar:
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

        if process.returncode == 0:
            print("\nGeneration completed successfully!")
        else:
            print(f"\nError: Blender exited with code {process.returncode}")

    except FileNotFoundError:
        print("Error: Blender not found. Make sure you have it installed and added in the system PATH")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()