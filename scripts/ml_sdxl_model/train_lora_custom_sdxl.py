import os, sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import bitsandbytes as bnb
from typing import Optional
from dataclasses import dataclass
import diffusers
import csv

print(diffusers.__version__)

sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from utils.utils_method import get_full_path
from utils.constants import PATH_DATASET_GENERATED, PATH_MODEL_WEIGHTS, PATH_MODEL_SINGLE_EPOCH_WEIGHTS, PATH_MODEL_SDXL_ORI, PATH_CAPTIONS_IMG, CAPTIONS_ENABLED
from utils.utils_ml import test_sdxl_conditioning, test_sdxl_time_ids, test_sdxl_unet_forward, test_vae_reconstruction, TerrainDataset, LatentTerrainDataset

@dataclass
class TrainingConfig:
    """
    Configuration parameters for the custom Stable Diffusion XL LoRA fine-tuning.
    
    Attributes:
        batch_size (int): Number of training examples utilized in one iteration. 
            Default is 1.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients 
            before performing a backward/update pass. Used to simulate a larger batch size. 
            Default is 4.
        learning_rate (float): The initial learning rate for the optimizer. 
            Default is 5e-5.
        num_epochs (int): Total number of training epochs to perform. 
            Default is 27.
        num_img_to_use (int): The maximum number of images from the dataset 
            to be used during this training run. Default is 50.
    """
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    num_epochs: int = 20
    num_img_to_use: Optional[int] = None

def precompute_latents_to_disk(original_dataset, vae, device, output_dir):
    """
    Passa tutte le immagini nel VAE una sola volta e salva i latenti su disco.
    Se i file esistono già, salta l'operazione per risparmiare tempo!
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Controlliamo se abbiamo già fatto il caching
    existing_files = [f for f in os.listdir(output_dir) if f.endswith(".pt")]
    if len(existing_files) == len(original_dataset):
        print(f"I latenti sono già stati calcolati ({len(existing_files)} file). Salto il caching!")
        return

    print("Inizio il Latent Caching... (Verrà fatto una sola volta!)")
    vae.eval()
    
    for i in range(len(original_dataset)):
        # Estraiamo i dati dal vecchio dataset
        sample = original_dataset[i]
        img = sample["image"].unsqueeze(0).to(device, dtype=torch.float32) # Aggiunge la dimensione del batch (1, C, H, W)
        
        # Nome originale del file (senza estensione exr/png)
        img_path = original_dataset.image_path[i]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        with torch.no_grad():
            # Codifica VAE e moltiplicazione per scaling_factor (esattamente come facevi nel loop!)
            latent = vae.encode(img).latent_dist.sample()
            latent = latent * vae.config.scaling_factor
            
            # Lo riportiamo alla CPU e rimuoviamo la dimensione batch finta
            latent = latent.squeeze(0).cpu() 
        
        # Salviamo il tensore su disco
        save_path = os.path.join(output_dir, f"{base_name}.pt")
        torch.save(latent, save_path)
        
        if (i+1) % 10 == 0:
            print(f"Cached {i+1}/{len(original_dataset)} images...")

    print("Latent Caching completato con successo!")


def encode_prompt_sdxl(batch_texts, tokenizers, text_encoders, device):
    """
    Encodes a batch of text prompts using the dual tokenizers and text encoders of SDXL.

    This function tokenizes the input text using two separate CLIP tokenizers, 
    passes the token IDs through their respective text encoders, extracts the 
    penultimate hidden states, and concatenates them to form the final prompt 
    embeddings. It also extracts the pooled text embeddings from the second encoder.

    Parameters
    ----------
    batch_texts : list of str
        A list of text prompts to be encoded.
    tokenizers : tuple or list
        A collection containing the two SDXL tokenizers (typically CLIPTokenizer 
        and CLIPTokenizerFast/OpenCLIPTokenizer).
    text_encoders : tuple or list
        A collection containing the two SDXL text encoders (typically CLIPTextModel 
        and CLIPTextModelWithProjection).
    device : torch.device or str
        The target device (e.g., 'cuda' or 'cpu') where tensors should be allocated.

    Returns
    -------
    prompt_embeds : torch.Tensor
        The concatenated hidden states from both text encoders, extracted from 
        the penultimate layer (-2). Shape: (batch_size, sequence_length, hidden_dim_1 + hidden_dim_2).
    pooled_prompt_embeds : torch.Tensor
        The pooled text embeddings from the second text encoder, used for SDXL 
        time embeddings. Shape: (batch_size, pooled_dim).

    Notes
    -----
    - Input texts are automatically padded and truncated to a maximum length of 77 tokens.
    - Hidden states are extracted from the penultimate layer (`hidden_states[-2]`) 
      following standard Stable Diffusion XL practices.
    """
     
    tok1, tok2 = tokenizers
    te1, te2 = text_encoders

    txt_in_1 = tok1(batch_texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
    txt_in_2 = tok2(batch_texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
    
    out_1 = te1(txt_in_1.input_ids, output_hidden_states=True)
    out_2 = te2(txt_in_2.input_ids, output_hidden_states=True)
    
    static_prompt_embeds_1 = out_1.hidden_states[-2]
    static_prompt_embeds_2 = out_2.hidden_states[-2]
    prompt_embeds = torch.cat([static_prompt_embeds_1, static_prompt_embeds_2], dim=-1)
    
    pooled_prompt_embeds = out_2.text_embeds if hasattr(out_2, "text_embeds") else out_2.pooler_output

    return prompt_embeds, pooled_prompt_embeds


def train_lora_custom(config: TrainingConfig = TrainingConfig()):

    """
    Fine-tunes the UNet of Stable Diffusion XL using LoRA on the
    generated terrain dataset.

    The training procedure:
    - loads the SDXL base pipeline and its VAE, UNet, and text encoders;
    - freezes the original model parameters;
    - injects LoRA adapters into the UNet attention layers;
    - precomputes the text embeddings for the empty prompt;
    - encodes the training images into the latent space using the VAE;
    - adds random noise according to the diffusion noise scheduler;
    - trains only the LoRA parameters to predict the added noise;
    - applies gradient scaling, gradient clipping, and a cosine learning-rate
      scheduler;
    - saves the LoRA adapter weights after each epoch and at the end of training.

    The function uses the same preprocessing and latent representation
    employed during training and is designed to fine-tune the model while
    keeping the original SDXL weights frozen.

    Returns:
        None. The trained LoRA adapter weights are saved to the configured
        output directories.
    """

    dataset_folder = PATH_DATASET_GENERATED
    output_dir_model = PATH_MODEL_WEIGHTS
    output_dir_model_ori = PATH_MODEL_SDXL_ORI
    output_dir_weights_epoch = PATH_MODEL_SINGLE_EPOCH_WEIGHTS
    use_captions = CAPTIONS_ENABLED

    batch_size = config.batch_size
    gradient_accumulation_steps = config.gradient_accumulation_steps
    #1e-4
    learning_rate = config.learning_rate
    #27
    num_epochs = config.num_epochs
    num_img_to_use = config.num_img_to_use

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(get_full_path(output_dir_model), exist_ok=True)

    print("Load SDLX Components...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )

    #test_vae_reconstruction(pipe, get_full_path(dataset_folder), device=device)
    #inspect_dataset_image(get_full_path(PATH_DATASET_GENERATED))
    #test_sdxl_conditioning(pipe, device="cuda")
    #test_sdxl_time_ids(pipe, device="cuda")
    #test_sdxl_unet_forward(pipe, get_full_path(dataset_folder), device="cuda")

    user_command = ""
    while user_command.strip().lower() != "continue":
        user_command = input("\nType 'continue' to resume Method For Training or Press CTRL+C for exit: ")

    vae = pipe.vae.to(device, dtype=torch.float32)
    vae.enable_tiling()
    unet = pipe.unet.to(device, dtype=torch.float16)
    unet.enable_gradient_checkpointing()
    text_encoder_1 = pipe.text_encoder.to(device)
    text_encoder_2 = pipe.text_encoder_2.to(device)
    tokenizer_1 = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    unet.enable_gradient_checkpointing()

    print("\n===== DIFFUSION SCHEDULER CHECK =====")
    print("Scheduler:", type(noise_scheduler).__name__)
    print("Prediction type:", noise_scheduler.config.prediction_type)
    print("Num train timesteps:", noise_scheduler.config.num_train_timesteps)
    print("Beta start:", noise_scheduler.config.beta_start)
    print("Beta end:", noise_scheduler.config.beta_end)
    print("===== END DIFFUSION SCHEDULER CHECK =====")

    print("Injection LoRA modules in the UNet...")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)

    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)


    unet.print_trainable_parameters()

    initial_lora = {}

    for name, param in unet.named_parameters():
        if param.requires_grad and "lora_" in name:
            initial_lora[name] = param.detach().float().cpu().clone()

    print("\n===== PEFT CHECK =====")
    trainable_params = [
        (name, p)
        for name, p in unet.named_parameters()
        if p.requires_grad
    ]

    print(f"Trainable tensors: {len(trainable_params)}")

    for name, p in trainable_params[:10]:
        print(
            f"{name} | shape={tuple(p.shape)} | dtype={p.dtype}"
        )
    print("\n===== END PEFT CHECK =====")

    trainable_parameters = [p for p in unet.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(trainable_parameters, lr=learning_rate, weight_decay=1e-2)

    scaler = torch.amp.GradScaler('cuda')

    size_img_format = 1024
    latents_folder = get_full_path(dataset_folder + "_latents")
    original_dataset = TerrainDataset(
        get_full_path(dataset_folder), 
        get_full_path(PATH_CAPTIONS_IMG), 
        size=size_img_format, 
        use_captions=False)

    precompute_latents_to_disk(original_dataset, vae, device, latents_folder)
    
    original_dataset.image_path = original_dataset.image_path[:num_img_to_use]

    del vae
    del original_dataset
    pipe.vae = None
    torch.cuda.empty_cache()
    print("VAE rimosso dalla VRAM. Memoria liberata con successo!")

    dataset = LatentTerrainDataset(latents_folder, get_full_path(PATH_CAPTIONS_IMG), use_captions=use_captions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    max_train_steps = num_epochs * len(dataloader) // gradient_accumulation_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * 0.05),  
        num_training_steps=max_train_steps
    )

    if not use_captions:
        print("USE_CAPTIONS is False. Pre-calculating empty prompt embeddings...")
        with torch.no_grad():
            dummy_text = [""] # Testo vuoto
            static_prompt_embeds, static_pooled_embeds = encode_prompt_sdxl(
                dummy_text, 
                (tokenizer_1, tokenizer_2), 
                (text_encoder_1, text_encoder_2), 
                device
            )
            del text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2
            torch.cuda.empty_cache()
    else:
        print("USE_CAPTIONS is True. Text Encoders will remain in memory for dynamic encoding.")

    print("Start Training...")
    unet.train()
    loss_history = []
    loss_history_path = os.path.join(
        get_full_path(output_dir_model_ori),
        "loss_history.csv"
    )

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()  
        
        for step, batch in enumerate(progress_bar):
            latents = batch["latent"].to(device, dtype=torch.float32)
            #images = batch["image"].to(device, dtype=torch.float32)
            bsz = latents.shape[0]

            with torch.no_grad():

                if use_captions:
                    texts = batch["text"]
                    prompt_embeds, pooled_prompt_embeds = encode_prompt_sdxl(
                        texts, 
                        (tokenizer_1, tokenizer_2), 
                        (text_encoder_1, text_encoder_2), 
                        device
                    )
                else:
                    prompt_embeds = static_prompt_embeds.repeat(bsz, 1, 1)
                    pooled_prompt_embeds = static_pooled_embeds.repeat(bsz, 1)

            noise = torch.randn_like(latents)
            
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()  
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
            add_time_ids = torch.tensor([[size_img_format, size_img_format, 0, 0, size_img_format, size_img_format]], dtype=torch.float16, device=device)
            add_time_ids = add_time_ids.repeat(bsz, 1) 

            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

            if torch.isnan(latents).any():
                print("THE CULPRIT IS THE VAE: Generated latents contain NaNs!")
            if torch.isnan(prompt_embeds).any():
                print("THE CULPRIT IS THE TEXT ENCODERS: Text embeddings contain NaNs!")
            if torch.isnan(noise).any():
                print("THE CULPRIT IS THE NOISE GENERATOR!")
            if torch.isnan(noisy_latents).any():
                print("THE CULPRIT IS THE NOISE SCHEDULER: Added noise resulted in NaNs!")

            with torch.amp.autocast('cuda', dtype=torch.float16):
                noise_pred = unet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)

                trainable_params = [p for p in unet.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)  

                scaler.step(optimizer)
                scaler.update()
                
                lr_scheduler.step()
                optimizer.zero_grad()

                current_loss = loss.item() * gradient_accumulation_steps if not torch.isnan(loss) else 0.0
                progress_bar.set_postfix({
                    "loss": f"{current_loss:.6f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.1e}"
                })

        print("\n===== LORA WEIGHT CHECK =====")
        for name, param in unet.named_parameters():
            if param.requires_grad and "lora_" in name:
                print(
                    name,
                    "mean abs =",
                    param.detach().float().abs().mean().item()
                )
                break
        print("\n===== END LORA WEIGHT CHECK =====")


        print("\n===== LORA UPDATE CHECK =====")
        for name, param in unet.named_parameters():
            if name in initial_lora:

                diff = (
                    param.detach().float().cpu() - initial_lora[name]
                ).abs().mean().item()

                print(f"{name} -> MEAN CHANGE = {diff:.10e}")
        print("\n===== END LORA UPDATE CHECK =====")


        media_loss = epoch_loss / len(dataloader)
        current_lr = lr_scheduler.get_last_lr()[0]

        lora_mean_abs = 0.0
        lora_mean_change = 0.0
        count_lora_tensors = 0

        for name, param in unet.named_parameters():
            if param.requires_grad and "lora_" in name:
                lora_mean_abs += param.detach().float().abs().mean().item()

                if name in initial_lora:
                    diff = (param.detach().float().cpu() - initial_lora[name]).abs().mean().item()
                    lora_mean_change += diff
                    
                count_lora_tensors += 1

        if count_lora_tensors > 0:
            lora_mean_abs /= count_lora_tensors
            lora_mean_change /= count_lora_tensors

        print(f"\n-> End Epoch {epoch+1}/{num_epochs}")
        print(f"   Average Loss     : {media_loss:.6f}")
        print(f"   Learning Rate    : {current_lr:.2e}")
        print(f"   LoRA Mean Abs    : {lora_mean_abs:.6f}")
        print(f"   LoRA Mean Change : {lora_mean_change:.6f}\n")

        loss_history.append({
            "epoch": epoch + 1,
            "average_loss": media_loss,
            "learning_rate": current_lr,
            "lora_mean_abs": lora_mean_abs,
            "lora_mean_change": lora_mean_change
        })

        print(f"-> End Epoch {epoch+1} - Average Loss: {media_loss:.6f}\n")

        with open(loss_history_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "average_loss", "learning_rate", "lora_mean_abs", "lora_mean_change"]
            )
            writer.writeheader()
            writer.writerows(loss_history)

        print(f"Saving checkpoint Epoch {epoch+1}...")
        cartella_epoca = get_full_path(output_dir_weights_epoch, f"epoch_{epoch+1}")
        unet.save_pretrained(cartella_epoca)
        print(f"Checkpoint saved: {cartella_epoca}\n")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\n--- Validation Image Generation (Epoch {epoch+1}) ---")
            unet.eval()

            if use_captions:
                val_prompt = "heightmap terrain, low base noise, high secondary noise, mountains, low mountain scale, very high force field, " \
                "medium mountain height, medium mountain steepness, medium mountain width, alluvial fan, very low fan length, high fan angle, " \
                "very high fan height" 
                val_guidance = 6.0
            else:
                val_prompt = ""
                val_guidance = 1.0

            pipe.unet = unet
            val_generator = torch.Generator(device=device).manual_seed(42)

            print("Generating validation latents...")
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    output = pipe(
                        prompt=val_prompt,
                        negative_prompt="color, photorealistic, 3d render, water, trees",
                        num_inference_steps=30,
                        guidance_scale=val_guidance,
                        original_size=(1024, 1024),
                        target_size=(1024, 1024),
                        generator=val_generator,
                        output_type="latent"
                    )
                    val_latents = output.images[0] if isinstance(output.images, list) else output.images

            print("Temporarily loading VAE into VRAM for validation...")
            from diffusers import AutoencoderKL
            from PIL import Image

            temp_vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", 
                torch_dtype=torch.float32
            ).to(device)
            temp_vae.enable_tiling()

            print("Decoding latents to image...")
            with torch.no_grad():
                # I latenti generati dalla pipeline vanno riscalati prima di passarli al VAE
                scaled_latents = val_latents / temp_vae.config.scaling_factor
                
                # Il VAE richiede float32 per evitare artefatti neri
                scaled_latents = scaled_latents.to(device, dtype=torch.float32)
                
                # Decodifica dei latenti
                decoded_images = temp_vae.decode(scaled_latents).sample
                
                # Convertiamo il tensore in una normale immagine PIL
                decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)
                decoded_images = decoded_images.cpu().permute(0, 2, 3, 1).float().numpy()
                val_image = Image.fromarray((decoded_images[0] * 255).astype("uint8"))

            # Pulizia immediata della VRAM
            del temp_vae
            torch.cuda.empty_cache()
            print("VAE removed from VRAM. Memory cleared to resume training.")

            visual_output_dir = get_full_path(output_dir_weights_epoch, "training_progress_images")
            os.makedirs(visual_output_dir, exist_ok=True)

            val_image_gray = val_image.convert('L')
            val_path = get_full_path(visual_output_dir, f"sample_epoch_{epoch+1:03d}.png")
            val_image_gray.save(val_path)

            print(f"Image saved in: {val_path}")

            unet.train()
            print("--- Continue Training ---\n")



    print("Saving the final model...")
    cartella_model_final = get_full_path(output_dir_model)
    unet.save_pretrained(cartella_model_final)
    print(f"Training completed! Final weights saved in: {cartella_model_final}")

if __name__ == "__main__":
    train_lora_custom(TrainingConfig)