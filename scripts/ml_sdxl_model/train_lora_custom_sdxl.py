import os, sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from transformers import get_cosine_schedule_with_warmup
from diffusers.utils import convert_all_state_dict_to_peft
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from tqdm import tqdm
import bitsandbytes as bnb

sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from utils.utils_method import get_full_path
from utils.constants import PATH_FOLDER_SDXL, PATH_DATASET_GENERATED, PATH_MODEL_WEIGHTS, PATH_MODEL_SINGLE_EPOCH_WEIGHTS

class TerrainDataset(Dataset):

    image_path = None
    transform = None
    dummy_prompt = None

    def __init__(self, image_dir, size=1024):
        self.image_path = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                           if f.endswith(('.png', '.jpg'))]
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.dummy_prompt = ""

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        img = Image.open(self.image_path[index]).convert("RGB")
        return {"image": self.transform(img), "text": self.dummy_prompt}


def train_lora_custom():
    dataset_folder = PATH_DATASET_GENERATED
    output_dir_model = PATH_MODEL_WEIGHTS
    output_dir_weights_epoch = PATH_MODEL_SINGLE_EPOCH_WEIGHTS

    batch_size = 1
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    num_epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(get_full_path(output_dir_model), exist_ok=True)

    print("Load SDLX Components...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )

    vae = pipe.vae.to(device, dtype=torch.float32)
    unet = pipe.unet.to(device, dtype=torch.float8_e4m3fn)
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

    print("Injection LoRA modules in the UNet...")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
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

    #optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate, weight_decay=1e-2)
    trainable_parameters = [p for p in unet.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(trainable_parameters, lr=learning_rate, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda')
  
    dataset = TerrainDataset(get_full_path(dataset_folder))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    max_train_steps = num_epochs * len(dataloader) // gradient_accumulation_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * 0.05),  
        num_training_steps=max_train_steps
    )

    print("Pre-calculation of text's embedding...")
    with torch.no_grad():
        dummy_text = [""]
        txt_in_1 = tokenizer_1(dummy_text, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
        txt_in_2 = tokenizer_2(dummy_text, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
        
        out_1 = text_encoder_1(txt_in_1.input_ids, output_hidden_states=True)
        out_2 = text_encoder_2(txt_in_2.input_ids, output_hidden_states=True)
        
        static_prompt_embeds_1 = out_1.hidden_states[-2]
        static_prompt_embeds_2 = out_2.hidden_states[-2]
        static_prompt_embeds = torch.cat([static_prompt_embeds_1, static_prompt_embeds_2], dim=-1)
        
        static_pooled_embeds = out_2.text_embeds if hasattr(out_2, "text_embeds") else out_2.pooler_output


    print("Start Training...")
    unet.train()
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()  
        
        for step, batch in enumerate(progress_bar):
            images = batch["image"].to(device, dtype=torch.float32)
            bsz = images.shape[0]

            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = (latents * vae.config.scaling_factor).to(dtype=torch.float16)

            noise = torch.randn_like(latents) # Rumore casuale puro
            
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()  
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            prompt_embeds = static_prompt_embeds.repeat(bsz, 1, 1)
            pooled_prompt_embeds = static_pooled_embeds.repeat(bsz, 1)
                
            add_time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.float16, device=device)
            add_time_ids = add_time_ids.repeat(bsz, 1) 

            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}


            if torch.isnan(latents).any():
                print("IL COLPETEVOLE È IL VAE: I latenti generati contengono NaN!")
            if torch.isnan(prompt_embeds).any():
                print("IL COLPETEVOLE SONO I TEXT ENCODERS: Gli embedding del testo contengono NaN!")
            if torch.isnan(noise).any():
                print("IL COLPETEVOLE È IL GENERATORE DI RUMORE (noise)!")
            if torch.isnan(noisy_latents).any():
                print("IL COLPETEVOLE È IL NOISE SCHEDULER: Il rumore aggiunto ha generato NaN!")


            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                noise_pred = unet(
                    noisy_latents, 
                    timesteps, 
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs
                ).sample

            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

            print("\n===== GRADIENT CHECK =====")

            for name, param in unet.named_parameters():
                if param.requires_grad and "lora_" in name:
                    if param.grad is None:
                        print(name, "-> GRADIENT = NONE")
                    else:
                        print(
                            name,
                            "-> GRADIENT MEAN =",
                            param.grad.detach().float().abs().mean().item()
                        )
                    break

            print("\n===== END GRADIENT CHECK =====")

            epoch_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                # 1. Riporta i gradienti alla scala normale prima del clipping
                scaler.unscale_(optimizer) 
                
                # 2. Esegui il clip sui parametri attivi
                trainable_params = [p for p in unet.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)  
                
                # 3. Aggiorna i pesi usando lo scaler (Sostituisce il vecchio optimizer.step())
                scaler.step(optimizer)
                scaler.update() # Aggiorna i fattori di scala per il prossimo step
                
                # 4. Fai avanzare lo scheduler del Learning Rate
                lr_scheduler.step()
                
                # 5. Resetta per il prossimo giro
                optimizer.zero_grad()
                global_step += 1

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
        print(f"-> End Epoch {epoch+1} - Average Loss: {media_loss:.6f}\n")

        print(f"Saving checkpoint Epoch {epoch+1}...")
        cartella_epoca = get_full_path(output_dir_weights_epoch, f"epoch_{epoch+1}")
        unet.save_pretrained(cartella_epoca)
        print(f"Checkpoint saved: {cartella_epoca}\n")
        #print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/len(dataloader):.4f}")

    print("Saving the final model...")
    cartella_model_final = get_full_path(output_dir_model)
    unet.save_pretrained(cartella_model_final)
    print(f"Training completed! Final weights saved in: {cartella_model_final}")

if __name__ == "__main__":
    train_lora_custom()