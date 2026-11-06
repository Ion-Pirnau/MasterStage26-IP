"""
Validation of LoRA checkpoints on heightmaps unseen during training.

Reuses existing project methods (TerrainDataset, LatentTerrainDataset,
precompute_latents_to_disk, encode_prompt_sdxl) and only introduces:
  1. PEFT-based loading of epoch_N checkpoints;
  2. Loss computation using FIXED timesteps and noise (epoch 0 = base SDXL, LoRA disabled);
  3. Writing/logging to validation_history.csv.
"""

import csv
import os, sys
import re

import torch
from tqdm import tqdm
import torch.nn.functional as F
from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from peft import PeftModel
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.utils_ml import TerrainDataset, LatentTerrainDataset

from train_lora_custom_sdxl import (
    encode_prompt_sdxl,
    precompute_latents_to_disk,
    get_full_path,
    PATH_DATASET_GENERATED,
    PATH_MODEL_SINGLE_EPOCH_WEIGHTS,
    PATH_MODEL_SDXL_ORI,
    CAPTIONS_ENABLED,
)

PATH_DATASET_VALIDATION = "data/heightmaps/heightmap_png_val"
PATH_CAPTIONS_VALIDATION = "data/heightmaps/captions_img_validation"
 
SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
 
TRAIN_DIR = get_full_path(PATH_DATASET_GENERATED)            
VAL_DIR = get_full_path(PATH_DATASET_VALIDATION)
CKPT_DIR = get_full_path(PATH_MODEL_SINGLE_EPOCH_WEIGHTS)    
OUT_DIR = get_full_path(PATH_MODEL_SDXL_ORI)                 
 
USE_CAPTIONS = CAPTIONS_ENABLED                               
VAL_CAPTIONS_DIR = get_full_path(PATH_CAPTIONS_VALIDATION) if USE_CAPTIONS else None
 
SIZE = 1024
BATCH_SIZE = 4
SEED = 42
NOISE_OFFSET = 0.05          
TIMESTEPS = [50, 150, 300, 500, 700, 900]
# ======================================================================
 
 
def find_checkpoints(ckpt_dir):
    found = []
    for name in os.listdir(ckpt_dir):
        m = re.fullmatch(r"epoch_(\d+)", name)
        full = os.path.join(ckpt_dir, name)
        if m and os.path.isdir(full) and os.path.exists(os.path.join(full, "adapter_config.json")):
            found.append((int(m.group(1)), full))
    return sorted(found)
 
 
@torch.no_grad()
def evaluate(model, batches, scheduler, timesteps, device, size, seed, noise_offset):
    """
    MSE on noise per timestep, using deterministic noise (identical across checkpoints).
    """
    model.eval()
    sums = {t: 0.0 for t in timesteps}
 
    for b, (lat, pe, pp) in enumerate(batches):
        lat = lat.to(device, dtype=torch.float32)
        pe = pe.to(device)
        pp = pp.to(device)
        bsz = lat.shape[0]
        ids = torch.tensor([[size, size, 0, 0, size, size]], dtype=torch.float16, device=device).repeat(bsz, 1)
 
        for t in timesteps:
            g = torch.Generator(device=device).manual_seed(seed * 1_000_003 + b * 1009 + t)
            noise = torch.randn(lat.shape, generator=g, device=device)
            if noise_offset > 0:
                off = torch.randn((bsz, lat.shape[1], 1, 1), generator=g, device=device)
                noise = noise + noise_offset * off
 
            ts = torch.full((bsz,), t, device=device, dtype=torch.long)
            noisy = scheduler.add_noise(lat, noise, ts)
 
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred = model(
                    noisy, ts,
                    encoder_hidden_states=pe,
                    added_cond_kwargs={"text_embeds": pp, "time_ids": ids},
                ).sample
            sums[t] += F.mse_loss(pred.float(), noise.float()).item()
 
    per_t = {t: sums[t] / len(batches) for t in timesteps}
    return sum(per_t.values()) / len(per_t), per_t
 
 
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Requires a CUDA-capable GPU")
    os.makedirs(OUT_DIR, exist_ok=True)
 
    # --- range globale del TRAINING, letto dal tuo TerrainDataset ---
    train_ds = TerrainDataset(TRAIN_DIR, None, size=SIZE, use_captions=False)
    gmin, gmax = train_ds.global_min, train_ds.global_max
    del train_ds
 
    print("Load SDXL components...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
 
    # --- latenti di validazione, con i tuoi metodi ---
    val_ds = TerrainDataset(VAL_DIR, None, size=SIZE, use_captions=False)
    if val_ds.global_min < gmin or val_ds.global_max > gmax:
        print(
            "WARNING: The validation set heights are out of the training range "
            f"(val: {val_ds.global_min}..{val_ds.global_max}, train: {gmin}..{gmax}). "
            "Out-of-range values will fall outside of [-1, 1]."
        )
    # normalizzazione identica al training
    val_ds.global_min, val_ds.global_max = gmin, gmax
 
    latents_dir = VAL_DIR.rstrip("/\\") + "_latents"
    vae = pipe.vae.to(device, dtype=torch.float32)
    vae.enable_tiling()
    precompute_latents_to_disk(val_ds, vae, device, latents_dir)
    del vae, val_ds
    pipe.vae = None
    torch.cuda.empty_cache()
 
    # --- latenti + caption + embedding, calcolati una volta sola ---
    latent_ds = LatentTerrainDataset(latents_dir, VAL_CAPTIONS_DIR, use_captions=USE_CAPTIONS)
    loader = DataLoader(latent_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Immagini di validazione: {len(latent_ds)} | caption: {USE_CAPTIONS}")
 
    te1 = pipe.text_encoder.to(device)
    te2 = pipe.text_encoder_2.to(device)
    tokenizers = (pipe.tokenizer, pipe.tokenizer_2)
 
    batches = []
    with torch.no_grad():
        if not USE_CAPTIONS:
            static_pe, static_pp = encode_prompt_sdxl([""], tokenizers, (te1, te2), device)
        
        for batch in tqdm(loader, desc="Encoding prompts and preparation"):
            lat = batch["latent"]
            bsz = lat.shape[0]
            if USE_CAPTIONS:
                pe, pp = encode_prompt_sdxl(list(batch["text"]), tokenizers, (te1, te2), device)
            else:
                pe, pp = static_pe.repeat(bsz, 1, 1), static_pp.repeat(bsz, 1)
            batches.append((lat.cpu(), pe.cpu(), pp.cpu()))
    if USE_CAPTIONS:
        print(f"Example caption: {latent_ds[0]['text']!r}")
 
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    del te1, te2
    torch.cuda.empty_cache()
 
    # --- checkpoint ---
    ckpts = find_checkpoints(CKPT_DIR)
    if not ckpts:
        raise RuntimeError(f"No epoch_N checkpoint found in {CKPT_DIR}")
    print(f"Checkpoint trovati: {[e for e, _ in ckpts]}")
 
    scheduler = DDPMScheduler.from_pretrained(SDXL_ID, subfolder="scheduler")
    unet = pipe.unet.to(device, dtype=torch.float16)
 
    first_epoch, first_path = ckpts[0]
    model = PeftModel.from_pretrained(unet, first_path, adapter_name=f"epoch_{first_epoch}")
 
    kw = dict(batches=batches, scheduler=scheduler, timesteps=TIMESTEPS, device=device,
              size=SIZE, seed=SEED, noise_offset=NOISE_OFFSET)
 
    rows = []
 
    # epoca 0: SDXL base (LoRA disattivato)
    with model.disable_adapter():
        v, per_t = evaluate(model, **kw)
    base = v
    rows.append({"epoch": 0, "val_loss": v, **{f"val_t{t}": x for t, x in per_t.items()}})
    print(f"Epoca 0 (SDXL base): val_loss = {v:.6f}")
 
    for epoch, path in tqdm(ckpts, desc="Evaluating LoRA Checkpoints"):
        name = f"epoch_{epoch}"
        if epoch != first_epoch:
            model.load_adapter(path, adapter_name=name)
        model.set_adapter(name)
        v, per_t = evaluate(model, **kw)
        rows.append({"epoch": epoch, "val_loss": v, **{f"val_t{t}": x for t, x in per_t.items()}})
        print(f"Epoca {epoch:>3}: val_loss = {v:.6f}  ({(v - base) / base * 100:+.2f}% vs base)")
 
    out_csv = os.path.join(OUT_DIR, "validation_history.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
 
    best = min(rows[1:], key=lambda r: r["val_loss"])
    print(f"\nMigliore: epoca {best['epoch']} con val_loss {best['val_loss']:.6f}")
    print(f"CSV salvato in: {out_csv}")
 
 
if __name__ == "__main__":
    main()