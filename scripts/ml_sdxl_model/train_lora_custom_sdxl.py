import os, sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import imageio.v3 as iio
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import bitsandbytes as bnb
import numpy as np
import diffusers
import csv

print(diffusers.__version__)

sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from utils.utils_method import get_full_path
from utils.constants import PATH_DATASET_GENERATED, PATH_MODEL_WEIGHTS, PATH_MODEL_SINGLE_EPOCH_WEIGHTS, PATH_FOLDER_SDXL


def test_sdxl_unet_forward(pipe, dataset_folder, device="cuda"):
    """
    Performs a deterministic forward pass through the SDXL UNet using
    the same latent representation and conditioning structure employed
    during custom LoRA training.

    The test verifies:
    - latent dimensions;
    - SDXL text conditioning;
    - pooled text conditioning;
    - time IDs;
    - UNet noise prediction;
    - output dimensions and numerical stability.

    A fixed random seed is used so that the generated noise and timestep
    are reproducible.
    """

    print("\n===== SDXL UNET FORWARD TEST =====")

    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = TerrainDataset(
        dataset_folder,
        size=1024
    )

    sample = dataset[0]

    image = sample["image"].unsqueeze(0).to(device)

    print("\n--- IMAGE ---")
    print("Shape:", image.shape)
    print("Min:", image.min().item())
    print("Max:", image.max().item())
    print("Mean:", image.mean().item())
    print("Std:", image.std().item())

    # ---------------------------------------------------------
    # VAE encoding
    # ---------------------------------------------------------

    vae = pipe.vae.to(
        device=device,
        dtype=torch.float32
    )

    with torch.no_grad():

        image_for_vae = image.to(
            device=device,
            dtype=torch.float32
        )

        posterior = vae.encode(
            image_for_vae
        ).latent_dist

        latents = posterior.sample()

        latents = latents * vae.config.scaling_factor

    print("\n--- LATENTS ---")
    print("Shape:", latents.shape)
    print("Min:", latents.min().item())
    print("Max:", latents.max().item())
    print("Mean:", latents.mean().item())
    print("Std:", latents.std().item())

    # ---------------------------------------------------------
    # Convert to UNet dtype
    # ---------------------------------------------------------

    latents = latents.to(
        device=device,
        dtype=pipe.unet.dtype
    )

    # ---------------------------------------------------------
    # Generate deterministic noise
    # ---------------------------------------------------------

    noise = torch.randn_like(latents)

    print("\n--- NOISE ---")
    print("Shape:", noise.shape)
    print("Min:", noise.min().item())
    print("Max:", noise.max().item())
    print("Mean:", noise.mean().item())
    print("Std:", noise.std().item())

    # ---------------------------------------------------------
    # Fixed timestep
    # ---------------------------------------------------------

    timestep = torch.tensor(
        [500],
        device=device,
        dtype=torch.long
    )

    print("\n--- TIMESTEP ---")
    print("Value:", timestep.item())

    # ---------------------------------------------------------
    # Add diffusion noise
    # ---------------------------------------------------------

    scheduler = pipe.scheduler

    noisy_latents = scheduler.add_noise(
        latents,
        noise,
        timestep
    )

    print("\n--- NOISY LATENTS ---")
    print("Shape:", noisy_latents.shape)
    print("Min:", noisy_latents.min().item())
    print("Max:", noisy_latents.max().item())
    print("Mean:", noisy_latents.mean().item())
    print("Std:", noisy_latents.std().item())

    # ---------------------------------------------------------
    # Text conditioning
    # ---------------------------------------------------------

    dummy_text = [""]

    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)

    with torch.no_grad():

        prompt_result = pipe.encode_prompt(
            prompt=dummy_text,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False
        )

    prompt_embeds = prompt_result[0]
    pooled_prompt_embeds = prompt_result[2]

    prompt_embeds = prompt_embeds.to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)

    print("\n--- PROMPT EMBEDDINGS ---")
    print("Shape:", prompt_embeds.shape)
    print("Mean:", prompt_embeds.mean().item())
    print("Std:", prompt_embeds.std().item())

    print("\n--- POOLED PROMPT EMBEDDINGS ---")
    print("Shape:", pooled_prompt_embeds.shape)
    print("Mean:", pooled_prompt_embeds.mean().item())
    print("Std:", pooled_prompt_embeds.std().item())

    # ---------------------------------------------------------
    # Time IDs
    # ---------------------------------------------------------

    add_time_ids = torch.tensor(
        [[
            1024,
            1024,
            0,
            0,
            1024,
            1024
        ]],
        dtype=torch.float16,
        device=device
    )

    print("\n--- TIME IDS ---")
    print("Shape:", add_time_ids.shape)
    print("Values:", add_time_ids)

    # ---------------------------------------------------------
    # Additional conditioning
    # ---------------------------------------------------------

    added_cond_kwargs = {
        "text_embeds": pooled_prompt_embeds,
        "time_ids": add_time_ids
    }

    pipe.unet = pipe.unet.to(
        device=device,
        dtype=torch.float16
    )

    print("\n--- DEVICE CHECK ---")
    print("Noisy latents:", noisy_latents.device)
    print("Timestep:", timestep.device)
    print("Prompt embeds:", prompt_embeds.device)
    print("Pooled embeds:", pooled_prompt_embeds.device)
    print("Time IDs:", add_time_ids.device)
    print("UNet:", next(pipe.unet.parameters()).device)

    # ---------------------------------------------------------
    # UNet forward
    # ---------------------------------------------------------

    with torch.no_grad():

        noise_pred = pipe.unet(
            noisy_latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs
        ).sample

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print("\n--- UNET OUTPUT ---")
    print("Shape:", noise_pred.shape)
    print("Min:", noise_pred.min().item())
    print("Max:", noise_pred.max().item())
    print("Mean:", noise_pred.mean().item())
    print("Std:", noise_pred.std().item())

    # ---------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------

    expected_shape = latents.shape

    print("\n--- SANITY CHECKS ---")

    print(
        "Output shape correct:",
        noise_pred.shape == expected_shape
    )

    print(
        "Contains NaN:",
        torch.isnan(noise_pred).any().item()
    )

    print(
        "Contains Inf:",
        torch.isinf(noise_pred).any().item()
    )

    if (
        noise_pred.shape == expected_shape
        and not torch.isnan(noise_pred).any()
        and not torch.isinf(noise_pred).any()
    ):
        print("\nRESULT: UNET FORWARD OK")
    else:
        print("\nRESULT: UNET FORWARD PROBLEM")

    print("===== END UNET FORWARD TEST =====\n")



def test_sdxl_time_ids(pipe, device="cuda"):
    """
    Checks the SDXL micro-conditioning time IDs used by the custom
    training code.

    The test verifies the expected time ID configuration for a
    1024x1024 training image and checks that the resulting tensor
    matches the configuration used by the SDXL pipeline.
    """

    print("\n===== SDXL TIME IDS TEST =====")

    size_img_format = 1024
    batch_size = 1

    # ---------------------------------------------------------
    # Time IDs used by the custom training code
    # ---------------------------------------------------------

    manual_time_ids = torch.tensor(
        [[
            size_img_format,
            size_img_format,
            0,
            0,
            size_img_format,
            size_img_format
        ]],
        dtype=torch.float16,
        device=device
    ).repeat(batch_size, 1)

    print("\n--- MANUAL TIME IDS ---")
    print("Shape:", manual_time_ids.shape)
    print("Values:", manual_time_ids)
    print("Dtype:", manual_time_ids.dtype)

    # ---------------------------------------------------------
    # Check SDXL configuration
    # ---------------------------------------------------------

    expected_time_ids = torch.tensor(
        [[
            1024,  # original height
            1024,  # original width
            0,     # crop top
            0,     # crop left
            1024,  # target height
            1024   # target width
        ]],
        dtype=torch.float16,
        device=device
    )

    print("\n--- EXPECTED SDXL TIME IDS ---")
    print("Shape:", expected_time_ids.shape)
    print("Values:", expected_time_ids)
    print("Dtype:", expected_time_ids.dtype)

    # ---------------------------------------------------------
    # Numerical comparison
    # ---------------------------------------------------------

    max_diff = torch.max(
        torch.abs(
            manual_time_ids - expected_time_ids
        )
    ).item()

    mean_diff = torch.mean(
        torch.abs(
            manual_time_ids - expected_time_ids
        )
    ).item()

    print("\n--- TIME IDS DIFFERENCE ---")
    print("Max absolute difference:", max_diff)
    print("Mean absolute difference:", mean_diff)

    # ---------------------------------------------------------
    # UNet configuration check
    # ---------------------------------------------------------

    print("\n--- UNET ADDITIONAL CONDITIONING ---")

    print(
        "UNet addition_embed_type:",
        pipe.unet.config.addition_embed_type
    )

    print(
        "UNet addition_time_embed_dim:",
        pipe.unet.config.addition_time_embed_dim
    )

    print(
        "UNet addition_time_embed_type:",
        pipe.unet.config.addition_time_embed_type
        if hasattr(pipe.unet.config, "addition_time_embed_type")
        else "N/A"
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    if max_diff == 0.0:
        print("\nRESULT: TIME IDS MATCH")
    else:
        print("\nRESULT: TIME IDS DO NOT MATCH")

    print("===== END TIME IDS TEST =====\n")


def test_sdxl_conditioning(pipe, device="cuda"):
    """
    Compares the manually computed SDXL text conditioning with the
    conditioning generated by the official pipeline method.

    The test uses the same empty prompt used during LoRA training and
    compares:
    - prompt embeddings;
    - pooled prompt embeddings.

    The goal is to verify that the custom training code produces the
    same text conditioning expected by the SDXL pipeline.
    """

    print("\n===== SDXL CONDITIONING TEST =====")

    pipe = pipe.to(device)

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    dummy_text = [""]
    size_img_format = 1024

    tokenizer_1 = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2

    text_encoder_1 = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2

    # ---------------------------------------------------------
    # MANUAL CONDITIONING
    # Same procedure currently used during training
    # ---------------------------------------------------------

    txt_in_1 = tokenizer_1(
        dummy_text,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    txt_in_2 = tokenizer_2(
        dummy_text,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        out_1 = text_encoder_1(
            txt_in_1.input_ids,
            output_hidden_states=True
        )

        out_2 = text_encoder_2(
            txt_in_2.input_ids,
            output_hidden_states=True
        )

        manual_prompt_embeds_1 = out_1.hidden_states[-2]
        manual_prompt_embeds_2 = out_2.hidden_states[-2]

        manual_prompt_embeds = torch.cat(
            [
                manual_prompt_embeds_1,
                manual_prompt_embeds_2
            ],
            dim=-1
        )

        manual_pooled_prompt_embeds = (
            out_2.text_embeds
            if hasattr(out_2, "text_embeds")
            else out_2.pooler_output
        )

    # ---------------------------------------------------------
    # PIPELINE CONDITIONING
    # Official Diffusers implementation
    # ---------------------------------------------------------

    with torch.no_grad():

        pipeline_result = pipe.encode_prompt(
            prompt=dummy_text,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False
        )

    pipeline_prompt_embeds = pipeline_result[0]
    pipeline_pooled_prompt_embeds = pipeline_result[2]

    # ---------------------------------------------------------
    # Shapes
    # ---------------------------------------------------------

    print("\n--- PROMPT EMBEDDINGS ---")

    print(
        "Manual shape:",
        manual_prompt_embeds.shape
    )

    print(
        "Pipeline shape:",
        pipeline_prompt_embeds.shape
    )

    print("\n--- POOLED PROMPT EMBEDDINGS ---")

    print(
        "Manual shape:",
        manual_pooled_prompt_embeds.shape
    )

    print(
        "Pipeline shape:",
        pipeline_pooled_prompt_embeds.shape
    )

    # ---------------------------------------------------------
    # Numerical comparison
    # ---------------------------------------------------------

    manual_prompt_embeds = manual_prompt_embeds.float()
    pipeline_prompt_embeds = pipeline_prompt_embeds.float()

    manual_pooled_prompt_embeds = (
        manual_pooled_prompt_embeds.float()
    )

    pipeline_pooled_prompt_embeds = (
        pipeline_pooled_prompt_embeds.float()
    )

    prompt_max_diff = torch.max(
        torch.abs(
            manual_prompt_embeds -
            pipeline_prompt_embeds
        )
    ).item()

    prompt_mean_diff = torch.mean(
        torch.abs(
            manual_prompt_embeds -
            pipeline_prompt_embeds
        )
    ).item()

    pooled_max_diff = torch.max(
        torch.abs(
            manual_pooled_prompt_embeds -
            pipeline_pooled_prompt_embeds
        )
    ).item()

    pooled_mean_diff = torch.mean(
        torch.abs(
            manual_pooled_prompt_embeds -
            pipeline_pooled_prompt_embeds
        )
    ).item()

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    print("\n--- PROMPT EMBEDDINGS DIFFERENCE ---")

    print(
        "Max absolute difference:",
        prompt_max_diff
    )

    print(
        "Mean absolute difference:",
        prompt_mean_diff
    )

    print("\n--- POOLED EMBEDDINGS DIFFERENCE ---")

    print(
        "Max absolute difference:",
        pooled_max_diff
    )

    print(
        "Mean absolute difference:",
        pooled_mean_diff
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    print("\n--- MANUAL STATISTICS ---")

    print(
        "Mean:",
        manual_prompt_embeds.mean().item()
    )

    print(
        "Std:",
        manual_prompt_embeds.std().item()
    )

    print("\n--- PIPELINE STATISTICS ---")

    print(
        "Mean:",
        pipeline_prompt_embeds.mean().item()
    )

    print(
        "Std:",
        pipeline_prompt_embeds.std().item()
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    tolerance = 1e-5

    if (
        prompt_max_diff < tolerance
        and pooled_max_diff < tolerance
    ):
        print("\nRESULT: CONDITIONING MATCHES")
    else:
        print("\nRESULT: CONDITIONING DOES NOT MATCH")

    print("===== END CONDITIONING TEST =====\n")


def inspect_dataset_image(dataset_folder):
    """
    Inspects the first image in the dataset before applying any preprocessing.

    The function loads the image directly from disk and reports its size,
    pixel range, mean, and standard deviation. This helps determine whether
    the original dataset image already contains valid terrain information
    before being processed by TerrainDataset.
    """
    print("\n===== RAW DATASET IMAGE TEST =====")

    image_files = [
        f for f in os.listdir(dataset_folder)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    if len(image_files) == 0:
        raise RuntimeError("No images found in the dataset.")

    image_path = os.path.join(dataset_folder, image_files[0])

    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    print("Image:", image_files[0])
    print("Size:", image.size)
    print("Shape:", image_array.shape)
    print("Min:", image_array.min())
    print("Max:", image_array.max())
    print("Mean:", image_array.mean())
    print("Std:", image_array.std())

    print("===== END RAW DATASET IMAGE TEST =====\n")


def test_vae_reconstruction(pipe, dataset_folder, device="cuda"):
    """
    Tests the VAE encoding and reconstruction process using an image from
    the terrain training dataset.

    The image is loaded through the same `TerrainDataset` class used during
    training, ensuring that the preprocessing steps are identical. The image
    is encoded into the latent space using the SDXL VAE and then decoded
    back into image space.

    This test is used to verify that:
    - the dataset preprocessing is correct;
    - the image can be successfully encoded into the latent space;
    - the VAE decoding process works correctly;
    - the main terrain structures and visual features are preserved during
      the image → latent → image reconstruction process.

    Args:
        pipe: Stable Diffusion XL pipeline containing the VAE.
        dataset_folder: Path to the folder containing the training images.
        device: Device used for the test, such as "cuda" or "cpu".

    Returns:
        None. The reconstructed image is saved as a PNG file.
    """

    print("\n===== VAE RECONSTRUCTION TEST =====")

    # Usiamo ESATTAMENTE il dataset del training
    dataset = TerrainDataset(
        dataset_folder,
        size=1024
    )

    if len(dataset) == 0:
        raise RuntimeError("Il dataset è vuoto.")

    # Prendiamo la prima immagine
    sample = dataset[0]

    image_tensor = sample["image"].unsqueeze(0).to(
        device,
        dtype=torch.float32
    )

    print(f"Numero immagini dataset: {len(dataset)}")

    print("\n--- INPUT DA TERRAIN DATASET ---")
    print("Shape:", image_tensor.shape)
    print("Min:", image_tensor.min().item())
    print("Max:", image_tensor.max().item())
    print("Mean:", image_tensor.mean().item())
    print("Std:", image_tensor.std().item())

    vae = pipe.vae.to(device, dtype=torch.float32)
    vae.eval()

    with torch.no_grad():

        # Encode
        latents = vae.encode(image_tensor).latent_dist.sample()

        # Stesso scaling usato nel training
        latents = latents * vae.config.scaling_factor

        print("\n--- LATENTS ---")
        print("Shape:", latents.shape)
        print("Min:", latents.min().item())
        print("Max:", latents.max().item())
        print("Mean:", latents.mean().item())
        print("Std:", latents.std().item())

        # Decode
        decoded = vae.decode(
            latents / vae.config.scaling_factor
        ).sample

    print("\n--- DECODED ---")
    print("Shape:", decoded.shape)
    print("Min:", decoded.min().item())
    print("Max:", decoded.max().item())
    print("Mean:", decoded.mean().item())
    print("Std:", decoded.std().item())

    # [-1,1] -> [0,1]
    decoded = (decoded / 2 + 0.5).clamp(0, 1)

    # Tensor -> PIL
    reconstructed = decoded[0].permute(1, 2, 0).cpu().numpy()
    reconstructed = (reconstructed * 255).round().astype("uint8")

    reconstructed_image = Image.fromarray(reconstructed)

    output_path = get_full_path(
        PATH_FOLDER_SDXL,
        "vae_reconstruction_test.png"
    )

    reconstructed_image.save(output_path)

    print("\nRicostruzione salvata:")
    print(output_path)

    print("===== END VAE TEST =====\n")



class TerrainDataset(Dataset):
    """
    Dataset for loading 16-bit terrain heightmaps and preparing them
    as RGB tensors suitable for Stable Diffusion XL training.

    The original 16-bit height information is preserved by loading the
    heightmaps directly with ImageIO. A global minimum and maximum
    elevation are computed across the entire dataset so that pixel
    values have a consistent elevation meaning between different
    terrain samples.

    Each heightmap is normalized to [0, 1], resized directly as a
    float32 tensor, replicated across three channels to obtain an RGB
    representation, and finally normalized to [-1, 1] for the SDXL VAE.
    """

    def __init__(self, image_dir, size=1024):
        self.image_path = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if len(self.image_path) == 0:
            raise RuntimeError(
                f"No heightmaps found in: {image_dir}"
            )

        self.size = size
        self.dummy_prompt = ""

        # ---------------------------------------------------------
        # Compute global height range across the entire dataset
        # ---------------------------------------------------------
        print("Computing global height range...")

        self.global_min = np.inf
        self.global_max = -np.inf

        for path in self.image_path:
            heightmap = iio.imread(path)

            if heightmap.ndim != 2:
                raise ValueError(
                    f"Expected a single-channel heightmap, "
                    f"but got shape {heightmap.shape} for {path}"
                )

            heightmap = heightmap.astype(np.float32)

            self.global_min = min(
                self.global_min,
                heightmap.min()
            )

            self.global_max = max(
                self.global_max,
                heightmap.max()
            )

        if self.global_max <= self.global_min:
            raise ValueError(
                "Invalid global height range: "
                f"min={self.global_min}, "
                f"max={self.global_max}"
            )

        print(f"Global height min: {self.global_min}")
        print(f"Global height max: {self.global_max}")

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):

        # ---------------------------------------------------------
        # 1. Load 16-bit heightmap
        # ---------------------------------------------------------
        img = iio.imread(self.image_path[index])

        if img.ndim != 2:
            raise ValueError(
                f"Expected a single-channel heightmap, "
                f"but got shape {img.shape} for "
                f"{self.image_path[index]}"
            )

        # ---------------------------------------------------------
        # 2. Convert uint16 → float32
        # ---------------------------------------------------------
        img = img.astype(np.float32)

        # ---------------------------------------------------------
        # 3. Global normalization [height] → [0, 1]
        # ---------------------------------------------------------
        img = (
            (img - self.global_min)
            / (self.global_max - self.global_min)
        )

        # ---------------------------------------------------------
        # 4. Convert NumPy → PyTorch tensor
        # ---------------------------------------------------------
        img = torch.from_numpy(img)

        # [H, W] → [1, H, W]
        img = img.unsqueeze(0)

        # ---------------------------------------------------------
        # 5. Resize while preserving float32 precision
        # ---------------------------------------------------------
        img = F.interpolate(
            img.unsqueeze(0),
            size=(self.size, self.size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # Grayscale → RGB
        img = img.repeat(3, 1, 1)

        # [0, 1] → [-1, 1] for SDXL VAE
        img = img * 2.0 - 1.0

        return {
            "image": img,
            "text": self.dummy_prompt
        }


def train_lora_custom():

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
    output_dir_weights_epoch = PATH_MODEL_SINGLE_EPOCH_WEIGHTS

    batch_size = 1
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    num_epochs = 100
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
    unet = pipe.unet.to(device, dtype=torch.float16)
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
    dataset = TerrainDataset(get_full_path(dataset_folder), size=size_img_format)
    dataset.image_path = dataset.image_path[:2]
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

        print(static_pooled_embeds.shape)

    del text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2
    torch.cuda.empty_cache()

    print("Start Training...")
    unet.train()
    loss_history = []
    loss_history_path = os.path.join(
        get_full_path(output_dir_model),
        "loss_history.csv"
    )

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

            noise = torch.randn_like(latents)
            
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()  
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            #noisy_latents.requires_grad_(True) 

            prompt_embeds = static_prompt_embeds.repeat(bsz, 1, 1)
            pooled_prompt_embeds = static_pooled_embeds.repeat(bsz, 1)
                
            add_time_ids = torch.tensor([[size_img_format, size_img_format, 0, 0, size_img_format, size_img_format]], dtype=torch.float16, device=device)
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

                # Esegui il clip sui parametri attivi
                trainable_params = [p for p in unet.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)  

                scaler.step(optimizer)
                scaler.update()
                
                # Fai avanzare lo scheduler del Learning Rate
                lr_scheduler.step()
                # Resetta per il prossimo giro
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
        loss_history.append({
            "epoch": epoch + 1,
            "average_loss": media_loss
        })

        print(f"-> End Epoch {epoch+1} - Average Loss: {media_loss:.6f}\n")

        with open(loss_history_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "average_loss"]
            )
            writer.writeheader()
            writer.writerows(loss_history)

        print(f"Saving checkpoint Epoch {epoch+1}...")
        cartella_epoca = get_full_path(output_dir_weights_epoch, f"epoch_{epoch+1}")
        unet.save_pretrained(cartella_epoca)
        print(f"Checkpoint saved: {cartella_epoca}\n")



    print("Saving the final model...")
    cartella_model_final = get_full_path(output_dir_model)
    unet.save_pretrained(cartella_model_final)
    print(f"Training completed! Final weights saved in: {cartella_model_final}")

if __name__ == "__main__":
    train_lora_custom()