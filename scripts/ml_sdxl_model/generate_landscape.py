import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from utils.utils_method import get_full_path
from utils.constants import PATH_FOLDER_SDXL, PATH_MODEL_SINGLE_EPOCH_WEIGHTS

def generate_landscape():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    print(f"Caricamento del modello {model_id} in corso...")

    # Inizializzazione Pipeline
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


    # CARICAMENTO LORA
    nome_cartella_epoca = "epoch_2"
    lora_weights = 1.0
    lora_folder_path = get_full_path(PATH_MODEL_SINGLE_EPOCH_WEIGHTS, nome_cartella_epoca)
    file_pesi = get_full_path(PATH_FOLDER_SDXL, nome_cartella_epoca, "adapter_model.safetensors")

    if os.path.exists(lora_folder_path):
        print(f"Caricamento del LoRA custom dalla cartella: {lora_folder_path}")
        pipe.load_lora_weights(lora_folder_path)
        print("LoRA caricato e fuso con successo sulla UNet!")
    else:
        print(f"ATTENZIONE: Cartella {lora_folder_path} NON TROVATA! Generazione base.")

    
    # GENERAZIONE UNCONDITIONED (Senza Testo)
    print("Generazione dell'immagine in corso...")
    generator = torch.Generator(device="cuda").manual_seed(42)

    image = pipe(
        prompt = "",
        negative_prompt= "",
        num_inference_steps=30,
        guidance_scale=0.0,
        original_size=(1024, 1024),
        target_size=(1024, 1024),
        crops_coords_top_left=(0, 0),
        width=1024,
        height=1024,
        generator=generator,
        cross_attention_kwargs={"scale": lora_weights}
    ).images[0]

    image = image.convert("L")
    # Salvataggio
    output_path = get_full_path(PATH_FOLDER_SDXL, f"paesaggio_generato_{nome_cartella_epoca}_LoRaAttiva.png")
    image.save(output_path)
    print(f"Immagine salvata con successo in: {output_path}")

if __name__ == "__main__":
    generate_landscape()