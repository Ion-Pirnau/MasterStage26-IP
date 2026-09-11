from pathlib import Path
import torch
from safetensors.torch import load_file
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from utils.utils_method import get_full_path
from utils.constants import PATH_FOLDER_SDXL, PATH_MODEL_SINGLE_EPOCH_WEIGHTS

lora_folder_path = Path(
    get_full_path(
        PATH_MODEL_SINGLE_EPOCH_WEIGHTS,
        "epoch_6"
    )
)

print(f"Cartella: {lora_folder_path}")
print(f"Esiste: {lora_folder_path.exists()}\n")

print(f"Analisi della cartella: {lora_folder_path}\n")

# Cerchiamo il file safetensors
safetensor_files = list(lora_folder_path.glob("*.safetensors"))

if not safetensor_files:
    raise FileNotFoundError(
        f"Nessun file .safetensors trovato in {lora_folder_path}"
    )

if len(safetensor_files) > 1:
    print("ATTENZIONE: trovati più file .safetensors:")
    for f in safetensor_files:
        print(f"  - {f}")

# Prendiamo il primo
safetensor_path = safetensor_files[0]

print(f"File dei pesi: {safetensor_path}\n")

# Carichiamo i pesi
weights = load_file(str(safetensor_path), device="cpu")

# Lista delle matrici LoRA
chiavi = list(weights.keys())

print(f"Numero totale di tensori nel LoRA: {len(chiavi)}")
print("-" * 70)

# Analizziamo tutti i tensori
for i, chiave in enumerate(chiavi):

    tensore = weights[chiave].float()

    valore_massimo = tensore.max().item()
    valore_minimo = tensore.min().item()
    valore_medio = tensore.mean().item()
    valore_assoluto_medio = tensore.abs().mean().item()

    elementi_zero = (tensore == 0.0).sum().item()
    totale_elementi = tensore.numel()

    percentuale_zeri = (elementi_zero / totale_elementi) * 100

    # Controllo se l'intero tensore è zero
    tutto_zero = torch.all(tensore == 0).item()

    print(f"Tensore {i+1}: {chiave}")
    print(f"  -> Shape:               {list(tensore.shape)}")
    print(f"  -> Max:                 {valore_massimo:.8f}")
    print(f"  -> Min:                 {valore_minimo:.8f}")
    print(f"  -> Mean:                {valore_medio:.8f}")
    print(f"  -> Mean assoluto:       {valore_assoluto_medio:.8f}")
    print(f"  -> Zeri esatti:         {elementi_zero}/{totale_elementi}")
    print(f"  -> Percentuale zeri:    {percentuale_zeri:.4f}%")
    print(f"  -> TUTTO ZERO:           {tutto_zero}")
    print("-" * 70)

    totale_parametri = 0
parametri_non_zero = 0
parametri_zero = 0

for chiave, tensore in weights.items():

    tensore = tensore.float()

    totale_parametri += tensore.numel()
    parametri_non_zero += (tensore != 0).sum().item()
    parametri_zero += (tensore == 0).sum().item()

print("\n" + "=" * 70)
print("RIEPILOGO COMPLESSIVO LoRA")
print("=" * 70)

print(f"Parametri totali:       {totale_parametri:,}")
print(f"Parametri non-zero:     {parametri_non_zero:,}")
print(f"Parametri zero:         {parametri_zero:,}")

print(
    f"Percentuale non-zero:   "
    f"{parametri_non_zero / totale_parametri * 100:.6f}%"
)

print(
    f"Percentuale zero:       "
    f"{parametri_zero / totale_parametri * 100:.6f}%"
)