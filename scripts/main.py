import sys, subprocess
from utils.utils_method import get_full_path, path_exists
from utils.constants import BLENDER_SCRIPT, BLEND_FILE, BLENDER_EXECUTABLE

def main():
    print("Start generation dataset pipeline...")

    if not path_exists(BLEND_FILE):
        print(f"Error: File Blender not found {BLEND_FILE}")
        sys.exit(1)
    
    if not path_exists(BLENDER_SCRIPT):
        print(f"Error: File Blender not found {BLEND_FILE}")
        sys.exit(1)

    command = [
        BLENDER_EXECUTABLE,
        "-b", BLEND_FILE    ,
        "-p", BLENDER_SCRIPT
    ]

    try:
        result = subprocess.run(command, check=True, text=True)
        print("Generation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during Blender execution. Exit Code: {e.returncode}")
    except FileNotFoundError:
        print("Error: Blender not found. Make sure you have it installed and added in the system PATH")



if __name__ == "__main__":
    main()