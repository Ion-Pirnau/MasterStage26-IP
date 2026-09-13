import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, AutoencoderKL
import sys, os
from pathlib import Path
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from utils.utils_method import get_full_path
from utils.constants import PATH_FOLDER_SDXL, PATH_MODEL_SINGLE_EPOCH_WEIGHTS, PATH_MODEL_WEIGHTS

def generate_without_lora():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    print("Caricamento SDXL BASE senza LoRA...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )

    pipe = pipe.to("cuda")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True
    )

    generator = torch.Generator(device="cuda").manual_seed(42)

    print("Generazione senza LoRA...")

    image = pipe(
        prompt="",
        negative_prompt="",
        num_inference_steps=30,
        guidance_scale=1.0,
        original_size=(1024, 1024),
        target_size=(1024, 1024),
        crops_coords_top_left=(0, 0),
        width=1024,
        height=1024,
        generator=generator,
    ).images[0]

    image = image.convert("L")

    output_path = get_full_path(
        PATH_FOLDER_SDXL,
        "test_SDXL_senza_LoRA.png"
    )

    image.save(output_path)

    print(f"Immagine salvata: {output_path}")


def generate_landscape(num_epoch):
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    print(f"Caricamento del modello {model_id} in corso...")

    # FIX VAE: Caricamento del VAE corretto per evitare la grana/marmo in fp16
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    )

    # Inizializzazione Pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    pipe = pipe.to("cuda")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True
    )

    # CARICAMENTO LORA
    nome_cartella_epoca = num_epoch
    lora_weights = 1.0
    lora_folder_path = Path(
        get_full_path(
            PATH_MODEL_SINGLE_EPOCH_WEIGHTS,
            nome_cartella_epoca
        )
    ) 

    print("Cartella LoRA:", lora_folder_path)
    print("Esiste:", lora_folder_path.exists())

    if lora_folder_path.exists():
        print("Contenuto:")
        for file in lora_folder_path.iterdir():
            print("  -", file.name)

    print(f"Caricamento LoRA PEFT dalla cartella: {lora_folder_path}")

    if not os.path.exists(lora_folder_path):
        raise FileNotFoundError(
            f"Cartella LoRA non trovata: {lora_folder_path}"
        )

    # La UNet originale è già presente nella pipeline
    unet_base = pipe.unet

    # Carica l'adapter PEFT sopra la UNet
    unet_lora = PeftModel.from_pretrained(
        unet_base,
        lora_folder_path,
        is_trainable=False
    )
    pipe.unet = unet_lora.merge_and_unload()

    # Sostituisce la UNet della pipeline con quella dotata dell'adapter
    #pipe.unet = unet_lora

    print("\n===== PEFT LOAD CHECK =====")
    print("Tipo UNet:", type(pipe.unet))

    #print("Adapter names:", pipe.unet.peft_config.keys())

    print("Trainable parameters:")
    #pipe.unet.print_trainable_parameters()

    print("===== END PEFT LOAD CHECK =====\n")

    print("LoRA PEFT caricata correttamente.")
    
    # GENERAZIONE UNCONDITIONED (Senza Testo)
    print("Generazione dell'immagine in corso...")
    generator = torch.Generator(device="cuda").manual_seed(42)

    image = pipe(
        prompt = "smooth grayscale depth map, soft terrain heightmap, top view",
        negative_prompt= "noise, grainy, marble pattern, high frequency, sharp edges, artifacts, detailed texture",
        num_inference_steps=50,
        guidance_scale=1.8,
        original_size=(1024, 1024),
        target_size=(1024, 1024),
        crops_coords_top_left=(0, 0),
        width=1024,
        height=1024,
        generator=generator
    ).images[0]

    #image = image.convert("L")
    # Salvataggio
    output_path = get_full_path(PATH_FOLDER_SDXL, f"paesaggio_generato_{nome_cartella_epoca}_LoRaAttiva.png")
    image.save(output_path)
    print(f"Immagine salvata con successo in: {output_path}")

if __name__ == "__main__":
    generate_landscape("epoch_25")
    #generate_without_lora()