import sys, subprocess, os
from utils.utils_method import get_full_path, path_exists
from utils.constants import BLENDER_SCRIPT, BLEND_FILE, BLENDER_EXECUTABLE

def main():
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
            print("Errori:")
            print(result.stderr)

        print("Return Code:")
        print(result.returncode)

    except subprocess.CalledProcessError as e:
        print(f"Error during Blender execution. Exit Code: {e.returncode}")

        print("\n--- ERRORE DETTAGLIATO DI BLENDER ---")
        print(e.stderr) 
        
        # Stampa anche l'output normale in caso l'errore sia finito lì
        print("\n--- OUTPUT NORMALE ---")
        print(e.stdout)
    except FileNotFoundError:
        print("Error: Blender not found. Make sure you have it installed and added in the system PATH")



if __name__ == "__main__":
    main()