import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from safetensors.torch import save_file, load_file
from peft import PeftModel, LoraConfig, get_peft_model, set_peft_model_state_dict
import sys, os
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from utils.utils_method import get_full_path
from utils.constants import PATH_FOLDER_SDXL, PATH_MODEL_SINGLE_EPOCH_WEIGHTS, PATH_MODEL_WEIGHTS

def generate_landscape():
    # 1. Definizione dell'ID del modello
    # Usiamo il modello base di SDXL
    # Da sostituire con il modello addestrato (DOPO)
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    print(f"Caricamento del modello {model_id} in corso...")

    # 2. Inizializzazione della Pipeline
    # Usiamo torch.float16 per dimezzare l'uso della VRAM della GPU (sto usando un LAPTOP)
    # use_safetensors=True garantisce caricamenti più sicuri e veloci
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )

    # 3. Spostramento su CUDA
    pipe = pipe.to("cuda")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True
    )

    # Se si ha poca VRAM usare questo comando
    # pipe.enable_model_cpu_offload()

    # Definiamo il percorso del file .safetensors che hai generato con l'addestramento
    # Puoi cambiare il nome del file per testare le varie epoche (es. "pytorch_lora_weights_epoch_5.safetensors")
    vero_file_safetensors = "pytorch_lora_weights_epoch_1_VERO.safetensors"
    cartella_pesi = get_full_path(PATH_MODEL_SINGLE_EPOCH_WEIGHTS)
    lora_path_new = get_full_path(PATH_MODEL_SINGLE_EPOCH_WEIGHTS, vero_file_safetensors)

    if os.path.exists(lora_path_new):
        print(f"Caricamento del LoRA custom da: {lora_path_new}")

        state_dict_peft = load_file(lora_path_new, device="cpu")
        state_dict_diffusers = {}

        # 2. Correggi i nomi delle chiavi al volo per adattarli a Diffusers
        for key, value in state_dict_peft.items():
            # Rimuovi il prefisso di PEFT
            new_key = key.replace("base_model.model.", "")
            # Aggiungi il prefisso unet. richiesto da Diffusers
            if not new_key.startswith("unet.") and not new_key.startswith("text_encoder"):
                new_key = "unet." + new_key
            state_dict_diffusers[new_key] = value

        # 3. Passa il dizionario corretto invece del percorso del file
        info = pipe.load_lora_weights(state_dict_diffusers, adapter_name="default")

        print(info)

        # 4. Ora l'adapter esiste e puoi impostare il peso a 1.5 per l'addestramento senza testo
        pipe.set_adapters(["default"], adapter_weights=[1.0])
        
        print("LoRA iniettato e attivato con successo sulla UNet!")

        
    else:
        print(f"ATTENZIONE: File LoRA non trovato in {lora_path_new}. Generazione con modello base.")

    # 4. Definizione del Prompt
    # Dato che la tua tesi riguarda i paesaggi, usiamo un prompt a tema
    prompt = "A breathtaking procedural fantasy landscape, towering majestic mountains, lush green valley, volumetric lighting, photorealistic, 8k resolution, highly detailed"
    negative_prompt = "blurry, low quality, distorted, bad composition, watermark"

    print("Generazione dell'immagine in corso...")

    generator = torch.Generator(device="cuda").manual_seed(42)

    # 5. Generazione (Inference)
    # num_inference_steps: quanti passaggi di "denoising" fare (default - 50, 40 sono un buon compromeso)
    image = pipe(
        prompt="",
        negative_prompt="",
        num_inference_steps=30,
        guidance_scale=1.0, # Quanto il modello deve ascoltare il prompt (tipico valore tra 5 - 8)
        original_size=(1024,1024),
        target_size=(1024,1024),
        crops_coords_top_left=(0,0),
        width=1024,
        height=1024
    ).images[0]

    # 6. Salvataggio
    output_path = get_full_path(PATH_FOLDER_SDXL, "paesaggio_generato_dopo_allenamento1.png")
    image.save(output_path)
    print(f"Immagine salvata con successo in: {output_path}")


def convert_tensor():
    lora_path_old = get_full_path(PATH_MODEL_SINGLE_EPOCH_WEIGHTS, "pytorch_lora_weights_epoch_4_test.pt")
    vero_file_safetensors = "pytorch_lora_weights_epoch_1_VERO.safetensors"
    lora_path_new = get_full_path(PATH_MODEL_SINGLE_EPOCH_WEIGHTS, vero_file_safetensors)
    
    state_dict = torch.load(lora_path_old, map_location="cpu")

    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("base.model.model"):
            new_key = k.replace("base_model.model.", "")
        clean_state_dict[new_key] = v.contiguous()

    
    save_file(clean_state_dict, lora_path_new)
    


if __name__ == "__main__":
    #convert_tensor()
    generate_landscape()