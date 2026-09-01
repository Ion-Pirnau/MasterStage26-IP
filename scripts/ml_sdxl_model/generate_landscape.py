import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.utils import convert_unet_state_dict_to_peft
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from utils.utils_method import get_full_path
from utils.constants import PATH_FOLDER_SDXL, PATH_MODEL_SINGLE_EPOCH_WEIGHTS

def generate_landscape():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    print(f"Caricamento del modello {model_id} in corso...")

    # 1. Inizializzazione Pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    pipe = pipe.to("cuda")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # ======================================================
    # 3. CARICAMENTO LORA
    # ======================================================
    nome_cartella_epoca = "epoch_1"
    lora_weights = 1.0
    lora_folder_path = get_full_path(PATH_MODEL_SINGLE_EPOCH_WEIGHTS, nome_cartella_epoca, "pytorch_lora_weights.safetensors")
    file_pesi = get_full_path(PATH_FOLDER_SDXL, nome_cartella_epoca, "adapter_model.safetensors")

    if os.path.exists(lora_folder_path):
        print(f"Caricamento del LoRA custom dalla cartella: {lora_folder_path}")
        
        # Estrarre lo state dict usando il loader ufficiale di Diffusers
        # Questo metodo legge correttamente il file 'pytorch_lora_weights.safetensors' senza richiedere JSON
        lora_state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(lora_folder_path)
        
        # Isolare e pulire le chiavi destinate alla UNet (esattamente come nel file di training)
        unet_state_dict = {
            f"{k.replace('unet.', '')}": v 
            for k, v in lora_state_dict.items() if k.startswith("unet.") or "base_model.model." in k
        }
        
        # Rimuovere eventuali residui di testo manuali prima della conversione ufficiale
        cleaned_unet_dict = {}
        for k, v in unet_state_dict.items():
            cleaned_key = k.replace("base_model.model.", "")
            cleaned_unet_dict[cleaned_key] = v

        peft_unet_state_dict = convert_unet_state_dict_to_peft(cleaned_unet_dict)
        
        # Caricare lo state dict convertito e validato nella pipeline
        pipe.load_lora_weights(peft_unet_state_dict)
        
        # Fondere i pesi LoRA direttamente nella UNet
        pipe.fuse_lora(lora_scale=3.0)
        
        
        print("LoRA caricato e fuso con successo sulla UNet!")
    else:
        print(f"ATTENZIONE: Cartella {lora_folder_path} NON TROVATA! Generazione base.")

    # ======================================================
    # 4. GENERAZIONE UNCONDITIONED (Senza Testo)
    # ======================================================
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
        width=768,
        height=768,
        generator=generator
    ).images[0]

    image = image.convert("L")
    # 5. Salvataggio
    output_path = get_full_path(PATH_FOLDER_SDXL, f"paesaggio_generato_{nome_cartella_epoca}_LoRaAtiva.png")
    image.save(output_path)
    print(f"Immagine salvata con successo in: {output_path}")

if __name__ == "__main__":
    generate_landscape()