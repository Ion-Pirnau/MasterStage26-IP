from pathlib import Path
from safetensors.torch import load_file
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from utils.utils_method import get_full_path
from utils.constants import PATH_FOLDER_SDXL, PATH_MODEL_SINGLE_EPOCH_WEIGHTS


lora_folder = Path(
    get_full_path(
        PATH_MODEL_SINGLE_EPOCH_WEIGHTS,
        "epoch_6"
    )
)

safetensor_file = next(lora_folder.glob("*.safetensors"))

print("=" * 80)
print("FILE")
print("=" * 80)
print(safetensor_file)

weights = load_file(str(safetensor_file), device="cpu")

print("\n" + "=" * 80)
print("CHIAVI DEL CHECKPOINT")
print("=" * 80)

print(f"Numero totale: {len(weights)}\n")

for i, key in enumerate(weights.keys()):
    if i < 30:
        print(f"{i+1:4d}: {key}")
print("\nPrime 30 chiavi mostrate.")