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


def generate_landscape(num_epoch, num_samples=4, base_seed=42):
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

    print("\n===== PEFT LOAD CHECK =====")
    print("Tipo UNet:", type(pipe.unet))

    print("Trainable parameters:")

    print("===== END PEFT LOAD CHECK =====\n")

    print("LoRA PEFT caricata correttamente.")

    prompt_addestrato = (
        "heightmap terrain, high base noise, low secondary noise, craters, low crater spacing, very low crater density, high crater height, "
        "low inner radius, very high outer radius, mountains, high mountain scale, very low force field, medium mountain height, medium mountain "
        "steepness, very low mountain width, alluvial fan, low fan length, medium fan angle, very high fan height"
    )

    negative_prompt_addestrato = (
        "color, photorealistic, 3d render, shadows, lighting, "
        "water, trees, text, watermark, artifacts, noisy, flat, blurry"
    )

    print(f"\nInizio generazione di {num_samples} campioni...")

    for i in range(num_samples):
        current_seed = base_seed + i
        generator = torch.Generator(device="cuda").manual_seed(current_seed)
        
        print(f"-> Generando sample {i+1}/{num_samples} (Seed: {current_seed})...")

        image = pipe(
            prompt = prompt_addestrato,
            negative_prompt= negative_prompt_addestrato,
            num_inference_steps=30,
            guidance_scale=6.0,
            original_size=(1024, 1024),
            target_size=(1024, 1024),
            crops_coords_top_left=(0, 0),
            width=1024,
            height=1024,
            generator=generator
        ).images[0]

        file_name = f"paesaggio_{nome_cartella_epoca}_seed{current_seed}.png"
        output_path = get_full_path(PATH_FOLDER_SDXL, file_name)
        image.save(output_path)
        print(f"Immagine salvata con successo in: {output_path}")

if __name__ == "__main__":
    generate_landscape("epoch_15", num_samples=4, base_seed=42)
    #generate_without_lora()