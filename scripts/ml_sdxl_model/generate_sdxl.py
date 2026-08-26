import torch
from diffusers import StableDiffusionXLPipeline
from utils.utils_method import get_full_path
from utils.constants import PATH_FOLDER_SDXL

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

    # Se si ha poca VRAM usare questo comando
    # pipe.enable_model_cpu_offload()

    # 4. Definizione del Prompt
    # Dato che la tua tesi riguarda i paesaggi, usiamo un prompt a tema
    prompt = "A breathtaking procedural fantasy landscape, towering majestic mountains, lush green valley, volumetric lighting, photorealistic, 8k resolution, highly detailed"
    negative_prompt = "blurry, low quality, distorted, bad composition, watermark"

    print("Generazione dell'immagine in corso...")

    # 5. Generazione (Inference)
    # num_inference_steps: quanti passaggi di "denoising" fare (default - 50, 40 sono un buon compromeso)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=40,
        guidance_scale=7.5 # Quanto il modello deve ascoltare il prompt (tipico valore tra 5 - 8)
    ).images[0]

    # 6. Salvataggio
    output_path = get_full_path(PATH_FOLDER_SDXL, "paesaggio_generato.png")
    image.save(output_path)
    print(f"Immagine salvata con successo in: {output_path}")


if __name__ == "__name__":
    generate_landscape()