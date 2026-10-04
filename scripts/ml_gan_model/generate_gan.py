import os
import sys
import torch
import numpy as np
import imageio.v3 as iio
from tqdm import tqdm
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from utils.utils_method import get_full_path
from utils.utils_ml import Generator
from utils.constants import (
    PATH_MODEL_WEIGHTS,
    PATH_MODEL_SINGLE_EPOCH_WEIGHTS, 
    PATH_FOLDER_GAN
)

@dataclass
class GenerateConfig:
    """Configuration class for Heightmap Generation."""
    latent_dim: int = 100
    num_maps: int = 10
    model_filename: str = get_full_path(PATH_MODEL_SINGLE_EPOCH_WEIGHTS,"generator_final.pth")
    output_dir: str = PATH_FOLDER_GAN
    # Costanti per il 16-bit puro (non alterare con l'altezza in metri)
    pixel_min: float = 0.0
    pixel_max: float = 65535.0


def generate_heightmaps(
    generator: torch.nn.Module, 
    device: torch.device, 
    config: GenerateConfig
) -> None:
    """
    Generates 16-bit terrain heightmaps from random latent noise.

    Args:
        generator (torch.nn.Module): The trained generator model.
        device (torch.device): Compute device (CPU or CUDA).
        config (GenerateConfig): Configuration settings for generation.
    """
    
    # Ensure output directory exists
    os.makedirs(config.output_dir, exist_ok=True)
    
    print(f"Generating {config.num_maps} heightmaps in 16-bit format...")

    # We use tqdm to show a progress bar for generation
    with torch.no_grad():
        for i in tqdm(range(config.num_maps), desc="Generating maps", unit="map"):
            
            # 1. Generate random noise vector (Latent Space)
            z = torch.randn(1, config.latent_dim, device=device)
            
            # 2. Forward pass through the Generator -> Tensor shape [1, 1, 256, 256]
            gen_img = generator(z)
            
            # 3. Remove extra dimensions -> shape [256, 256] and move to CPU/NumPy
            gen_img = gen_img.squeeze().cpu().numpy()
            
            # 4. Denormalize from GAN range [-1, 1] to standard range [0, 1]
            gen_img = (gen_img + 1.0) / 2.0
            
            # 5. Denormalize from [0, 1] to absolute 16-bit pixel range [0, 65535]
            gen_img = gen_img * (config.pixel_max - config.pixel_min) + config.pixel_min
            
            # 6. Clip values to prevent overflow and cast to 16-bit unsigned integer
            gen_img = np.clip(gen_img, 0, 65535)
            gen_img_16bit = gen_img.astype(np.uint16)
            
            # 7. Save output as PNG (ImageIO natively supports 16-bit PNGs)
            filename = get_full_path(config.output_dir, f"generated_terrain_{i+1:03d}.png")
            iio.imwrite(filename, gen_img_16bit)


def main():
    """
    Main execution pipeline: setup configuration, load model, and trigger generation.
    """
    # 1. Configuration & Device Setup
    config = GenerateConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Allocated Device: {device}")

    # 2. Initialize and Load Generator Model
    generator = Generator(config.latent_dim).to(device)
    
    if not os.path.exists(config.model_filename):
        raise FileNotFoundError(f"Trained model not found at: {config.model_filename}")
        
    print(f"Loading weights from: {config.model_filename}")
    generator.load_state_dict(torch.load(config.model_filename, map_location=device))
    
    # Set generator to evaluation mode (disables Dropout/BatchNorm updates)
    generator.eval()

    # 3. Start Generation Process
    generate_heightmaps(
        generator=generator, 
        device=device, 
        config=config
    )
    
    print(f"\nGeneration complete! Files saved in: {os.path.abspath(config.output_dir)}")


if __name__ == "__main__":
    main()