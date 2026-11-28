import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionXLPipeline,
                       UNet2DConditionModel)
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.utils_method import get_full_path
from utils.constants import PATH_FOLDER_SDXL, PATH_MODEL_SINGLE_EPOCH_WEIGHTS, CAPTIONS_ENABLED

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# ----------------------------------------------------------------------------------------
# MUST match the training run you are loading:
#   True  -> the LoRA was trained WITH captions (conditioned training)
#   False -> the LoRA was trained WITHOUT captions (unconditioned training, empty prompt)
# ----------------------------------------------------------------------------------------

# --- Conditioned case ------------------------------------------------------------------
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
SEEDS = [42, 123, 2024]                    # 4 prompts x 3 seeds = 12 images per epoch
GUIDANCE_COND = 6.0

# --- Unconditioned case ----------------------------------------------------------------
# No prompt exists, so the only thing that varies between images is the initial noise (seed):
# more seeds are used to see the diversity of the learned distribution.
SEEDS_UNCOND = [42, 123, 2024, 7, 99, 555, 1234, 31415]
GUIDANCE_UNCOND = 1.0                      


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
        Array of shape (H, W) and dtype uint16. Saved with ``Image.fromarray`` it becomes a
        16-bit PNG. Do NOT call ``.convert('L')`` on it: that would destroy the 16-bit data.
    """
    latents = latents.to(device=vae.device, dtype=torch.float32) / vae.config.scaling_factor
    with torch.no_grad():
        decoded = vae.decode(latents).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)[0].float().cpu().numpy()
    gray = decoded.mean(axis=0)
    return np.round(gray * 65535.0).astype(np.uint16)


def generate_landscape(epochs, captions_enable=None, prompts=None, seeds=None, num_steps=30,
                       guidance_scale=None, tag="", save_grid=True,
                       weights_dir=None, output_dir=None):
    """
    Generates heightmaps from one or more LoRA checkpoints, for several seeds (and prompts).

    Purpose
    -------
    Qualitative evaluation of the fine-tuning. Seeds (and prompts) are the same for every
    epoch, so images of different epochs are directly comparable: the same initial noise is
    denoised by models at different stages of training.

    Behaviour depends on ``captions_enable`` (default: the module constant CAPTIONS_ENABLE)
    ------------------------------------------------------------------------------------
    True  (conditioned training)
        Every prompt is generated with every seed: len(prompts) x len(seeds) images per epoch.
        Classifier-free guidance is on (guidance 6.0) and the negative prompt is used.
    False (unconditioned training)
        The model was trained with an empty prompt, so inference must use an empty prompt too:
        ``prompt=""``, no negative prompt and ``guidance_scale=1.0`` (CFG disabled, as in
        training). Prompts are ignored (the LoRA learned no text dependence), and only the
        seeds vary: len(seeds) images per epoch (default: SEEDS_UNCOND).
        Using a descriptive prompt with an unconditioned LoRA would not control the result.

    What it does
    ------------
    1. Loads the original SDXL VAE in float32 (needed for 16-bit output) and the SDXL base
       pipeline in float16, with a DPM-Solver multistep scheduler using Karras sigmas.
    2. For each epoch folder: loads a clean base UNet on the GPU (``merge_and_unload`` modifies
       weights in place, so reloading guarantees that no epoch inherits the previous merge),
       loads the PEFT LoRA adapter and merges it.
       The special name ``"base"`` skips the LoRA: it generates with plain SDXL, which is the
       reference to compare against.
    3. Generates each image with ``output_type="latent"``, decodes it with
       ``latents_to_uint16`` and saves a 16-bit PNG.
    4. Optionally builds a preview grid (rows = prompts, columns = seeds; one row if
       unconditioned).

    Parameters
    ----------
    epochs : str or list of str
        Checkpoint folder name(s) inside ``weights_dir``, e.g. ``["base", "epoch_5", "epoch_30"]``.
    captions_enable : bool or None
        True/False as above. None uses the module constant CAPTIONS_ENABLE.
    prompts : list of str or None
        Only used when captions are enabled. None uses PROMPTS.
    seeds : list of int or None
        None uses SEEDS (captions) or SEEDS_UNCOND (no captions).
    num_steps : int
        Number of inference steps.
    guidance_scale : float or None
        None uses GUIDANCE_COND (captions) or GUIDANCE_UNCOND (no captions).
    tag : str
        Optional suffix for every output file name, to keep sessions apart.
    save_grid : bool
        If True, saves one preview grid per epoch.
    weights_dir : str or None
        Folder with the epoch checkpoints of the run. None uses PATH_MODEL_SINGLE_EPOCH_WEIGHTS.
        Set it per run, so that conditioned and unconditioned runs never get mixed up.
    output_dir : str or None
        Folder where images are written. None uses PATH_FOLDER_SDXL.

    Output files (in ``output_dir``)
    --------------------------------
    Conditioned:   ``<epoch>_p<prompt_index>_s<seed><tag>.png``
    Unconditioned: ``<epoch>_uncond_s<seed><tag>.png``
        Heightmaps, single channel, 16 bit: use these for analysis.
    ``<epoch>_grid<tag>.png`` / ``<epoch>_uncond_grid<tag>.png``
        8-bit preview grid with 256x256 thumbnails, for visual inspection only.
    """
    if isinstance(epochs, str):
        epochs = [epochs]
    use_captions = CAPTIONS_ENABLED if captions_enable is None else captions_enable

    if use_captions:
        prompts = PROMPTS if prompts is None else prompts
        seeds = SEEDS if seeds is None else seeds
        guidance_scale = GUIDANCE_COND if guidance_scale is None else guidance_scale
        negative_prompt = NEGATIVE_PROMPT
        prefix = ""
    else:
        prompts = [""]                         # a single "empty prompt" row
        seeds = SEEDS_UNCOND if seeds is None else seeds
        guidance_scale = GUIDANCE_UNCOND if guidance_scale is None else guidance_scale
        negative_prompt = None                 # with guidance 1.0 CFG is off anyway
        prefix = "uncond_"
    weights_root = PATH_MODEL_SINGLE_EPOCH_WEIGHTS if weights_dir is None else weights_dir
    out_root = PATH_FOLDER_SDXL if output_dir is None else output_dir
    os.makedirs(get_full_path(out_root), exist_ok=True)

    print(f"Mode: {'CONDITIONED (captions)' if use_captions else 'UNCONDITIONED (no captions)'} "
          f"| guidance {guidance_scale}")
    print(f"Loading original SDXL VAE (float32) and pipeline from {MODEL_ID}...")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    vae.enable_tiling()
    vae.enable_slicing()

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    pipe.unet = None
    gc.collect()
    torch.cuda.empty_cache()

    total = len(epochs) * len(prompts) * len(seeds)
    done = 0
    print(f"Epochs: {epochs} | prompts: {len(prompts)} | seeds: {seeds} | images: {total}")

    for epoch_name in epochs:
        print(f"\n=== {epoch_name} ===")
        unet = UNet2DConditionModel.from_pretrained(
            MODEL_ID, subfolder="unet", torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True,
        ).to("cuda")

        if epoch_name == "base":
            merged = unet
            print("No LoRA: plain SDXL base model (reference).")
        else:
            lora_path = Path(get_full_path(weights_root, epoch_name))
            if not lora_path.exists():
                raise FileNotFoundError(f"LoRA folder not found: {lora_path}")
            print(f"Loading PEFT LoRA adapter from: {lora_path}")
            merged = PeftModel.from_pretrained(unet, lora_path, is_trainable=False).merge_and_unload()
            merged = merged.to("cuda", dtype=torch.float16)
            print("LoRA adapter loaded and merged.")
        pipe.unet = merged

        thumbs = []
        for p_idx, prompt in enumerate(prompts):
            row = []
            for seed in seeds:
                done += 1
                print(f"-> [{done}/{total}] {epoch_name} | "
                      f"{'prompt ' + str(p_idx) + ' | ' if use_captions else ''}seed {seed}...")
                generator = torch.Generator(device="cuda").manual_seed(seed)
                latents = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
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

                if use_captions:
                    name = f"{epoch_name}_p{p_idx}_s{seed}{tag}.png"
                else:
                    name = f"{epoch_name}_{prefix}s{seed}{tag}.png"
                out_path = get_full_path(out_root, name)
                Image.fromarray(gray16).save(out_path)
                print(f"   Image saved: {out_path}")
                row.append(Image.fromarray((gray16 >> 8).astype(np.uint8)).resize((256, 256)))
            thumbs.append(row)

        if save_grid:
            grid = Image.new("L", (256 * len(seeds), 256 * len(prompts)))
            for r, row in enumerate(thumbs):
                for c, im in enumerate(row):
                    grid.paste(im, (c * 256, r * 256))
            grid_path = get_full_path(out_root, f"{epoch_name}_{prefix}grid{tag}.png")
            grid.save(grid_path)
            print(f"Preview grid saved: {grid_path}")

        pipe.unet = None               
        del unet, merged
        gc.collect()
        torch.cuda.empty_cache()

    print("\nGeneration completed.")


if __name__ == "__main__":
    generate_landscape(["base", "epoch_15", "epoch_20", "epoch_25", "epoch_30"])