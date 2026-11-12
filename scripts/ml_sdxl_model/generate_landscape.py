import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler, AutoencoderKL
import sys, os
import numpy as np
from pathlib import Path
from peft import PeftModel
import copy
import gc
from pathlib import Path
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from utils.utils_method import get_full_path
from utils.constants import PATH_FOLDER_SDXL, PATH_MODEL_SINGLE_EPOCH_WEIGHTS, PATH_MODEL_WEIGHTS

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

PROMPTS = [
    "heightmap terrain, mountains, low height, low steepness, very low width, "
    "alluvial fan, high length, medium angle, low height",

    "heightmap terrain, erosion, medium frequency, very low depth, mountains, medium height, "
    "medium steepness, medium width, alluvial fan, high length, very low angle, low height",

    "heightmap terrain, craters, high spacing, very low density, high height, mountains, high "
    "height, very low steepness, very high width, alluvial fan, medium length, high angle, very high height",

    "heightmap terrain, craters, low spacing, high density, medium height, mountains, very low height, "
    "very high steepness, very low width, alluvial fan, low length, high angle, medium height",               
]

NEGATIVE_PROMPT = (
    "color, photorealistic, 3d render, shadows, lighting, "
    "water, trees, text, watermark, artifacts, noisy, flat, blurry"
)

SEEDS = [42, 123, 2024]

def latents_to_uint16(vae, latents):
    """
    Decodes SDXL latents into a single-channel 16-bit heightmap.
 
    Steps
    -----
    1. Undo the latent scaling used by SDXL (division by ``vae.config.scaling_factor``).
    2. Decode with the VAE in float32 (no fp16 artifacts). Tiling, if enabled on the VAE,
       keeps memory low at 1024x1024.
    3. Map the decoded values from [-1, 1] to [0, 1] and clamp them.
    4. Average the 3 decoded channels into one (the heightmap is a scalar field, so the three
       channels carry the same information and averaging also reduces decoding noise).
    5. Scale to [0, 65535], round and cast to uint16.
 
    Parameters
    ----------
    vae : diffusers.AutoencoderKL
        The original SDXL VAE (float32), already moved to the target device.
    latents : torch.Tensor
        Latents returned by the pipeline with ``output_type="latent"``, shape (1, 4, H/8, W/8).
 
    Returns
    -------
    numpy.ndarray
        Array of shape (H, W) and dtype uint16.
    """

    latents = latents.to(device=vae.device, dtype=torch.float32) / vae.config.scaling_factor
    with torch.no_grad():
        decoded = vae.decode(latents).sample                      
    decoded = (decoded / 2 + 0.5).clamp(0, 1)[0].float().cpu().numpy()
    gray = decoded.mean(axis=0)                                 
    return np.round(gray * 65535.0).astype(np.uint16)

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


def generate_landscape(epochs, prompts=PROMPTS, seeds=SEEDS, num_steps=30, guidance_scale=6.0,
                       tag="", save_grid=True):

    """
    Generates heightmaps from one or more LoRA checkpoints, for several prompts and seeds.
 
    Purpose
    -------
    Qualitative evaluation of the fine-tuning. For every selected epoch, every prompt is
    generated with every seed (len(prompts) x len(seeds) images per epoch). Seeds and prompts
    are the same for all epochs, so images of different epochs are directly comparable and
    show how the same prompt and the same initial noise change as training progresses.
 
    What it does
    ------------
    1. Loads the original SDXL VAE in float32 (needed for 16-bit output) and the SDXL base
       pipeline in float16, with a DPM-Solver multistep scheduler using Karras sigmas.
    2. For each epoch folder:
       a. makes a copy of the clean base UNet (``merge_and_unload`` modifies the weights in
          place, so without a copy every epoch would inherit the merge of the previous one);
       b. loads the PEFT LoRA adapter on the copy and merges it into the weights;
       c. for each prompt and seed, runs the pipeline with ``output_type="latent"``, decodes
          the latents with ``latents_to_uint16`` and saves a 16-bit PNG;
       d. optionally builds a preview grid (rows = prompts, columns = seeds);
       e. restores the base UNet and frees GPU memory.
 
    Parameters
    ----------
    epochs : str or list of str
        Name(s) of the checkpoint folder(s) inside ``PATH_MODEL_SINGLE_EPOCH_WEIGHTS``,
        e.g. ``["epoch_5", "epoch_10", "epoch_20"]``.
    prompts : list of str
        Prompts to generate. Defaults to the module-level ``PROMPTS``.
    seeds : list of int
        Seeds used for every prompt. Defaults to the module-level ``SEEDS``.
    num_steps : int
        Number of inference steps.
    guidance_scale : float
        Classifier-free guidance scale.
    tag : str
        Optional suffix appended to every output file name (e.g. ``"_v2"``), useful to keep
        different generation sessions apart.
    save_grid : bool
        If True, saves one preview grid per epoch.
 
    Output files (in ``PATH_FOLDER_SDXL``)
    --------------------------------------
    ``<epoch>_p<prompt_index>_s<seed><tag>.png``
        Heightmap, single channel, 16 bit. These are the files to use for analysis.
    ``<epoch>_grid<tag>.png``
        Preview grid, 8 bit grayscale, 256x256 thumbnails. Each 16-bit image is reduced to
        8 bit (``>> 8``) and downscaled only for visual inspection; do not use it for any
        quantitative analysis.
 
    Notes
    -----
    - Peak GPU memory is higher than a single-checkpoint script, because the UNet copy lives
      next to the original one.
    - Each epoch folder must contain a valid PEFT adapter (``adapter_config.json`` and weights).
    """

    if isinstance(epochs, str):
        epochs = [epochs]
    
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    vae.enable_tiling()
    vae.enable_slicing()

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    base_unet = pipe.unet.to("cpu")
    torch.cuda.empty_cache()

    total = len(epochs) * len(prompts) * len(seeds)
    done = 0
    print(f"Epochs: {epochs} | prompts: {len(prompts)} | seeds: {seeds} | "
          f"images to generate: {total}")

    for epoch_name in epochs:
        lora_path = Path(get_full_path(PATH_MODEL_SINGLE_EPOCH_WEIGHTS, epoch_name))
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA folder not found: {lora_path}")
        print(f"\n=== {epoch_name} ===")
        print(f"Loading PEFT LoRA adapter from: {lora_path}")

        unet = copy.deepcopy(base_unet)
        merged  = PeftModel.from_pretrained(unet, lora_path, is_trainable=False).merge_and_unload()
        pipe.unet = merged.to("cuda")
        print("LoRA adapter loaded and merged.")
 
        thumbs = []
        for p_idx, prompt in enumerate(prompts):
            row = []
            for seed in seeds:
                done += 1
                print(f"-> [{done}/{total}] {epoch_name} | prompt {p_idx} | seed {seed}...")
                generator = torch.Generator(device="cuda").manual_seed(seed)
                latents = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    original_size=(1024, 1024),
                    target_size=(1024, 1024),
                    crops_coords_top_left=(0, 0),
                    width=1024,
                    height=1024,
                    generator=generator,
                    output_type="latent",          
                ).images
                gray16 = latents_to_uint16(vae, latents)
    
                name = f"{epoch_name}_p{p_idx}_s{seed}{tag}.png"
                out_path = get_full_path(PATH_FOLDER_SDXL, name)
                Image.fromarray(gray16).save(out_path)
                print(f"   Image saved: {out_path}")
                row.append(Image.fromarray((gray16 >> 8).astype(np.uint8)).resize((256, 256)))
            thumbs.append(row)

        if save_grid:
            grid = Image.new("L", (256 * len(seeds), 256 * len(prompts)))
            for r, row in enumerate(thumbs):
                for c, im in enumerate(row):
                    grid.paste(im, (c * 256, r * 256))
            grid_path = get_full_path(PATH_FOLDER_SDXL, f"{epoch_name}_grid{tag}.png")
            grid.save(grid_path)
            print(f"Preview grid saved: {grid_path}")
    
        pipe.unet = base_unet
        del unet
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\nGeneration completed.")



if __name__ == "__main__":
    generate_landscape(["epoch_5", "epoch_10", "epoch_15", "epoch_20"])
    #generate_without_lora()