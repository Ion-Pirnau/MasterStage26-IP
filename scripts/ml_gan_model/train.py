import torch
import torch.nn as nn
import torch.optim as optim
import os, sys
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from utils.utils_method import get_full_path
from utils.utils_ml import GANTerrainDataset, Generator, Discriminator
from utils.constants import (
    PATH_DATASET_GENERATED, 
    PATH_MODEL_WEIGHTS, 
    PATH_MODEL_SINGLE_EPOCH_WEIGHTS
    )



@dataclass
class GANConfig:
    """Configuration class for GAN hyperparameters."""
    latent_dim: int = 100
    batch_size: int = 16
    epochs: int = 200
    learning_rate: float = 0.0002
    image_size: int = 256
    beta1: float = 0.5
    beta2: float = 0.999
    save_image_freq: int = 5    
    save_weight_freq: int = 50


def train_gan(
    generator: nn.Module, 
    discriminator: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device, 
    config: GANConfig
) -> None:
    """
    Executes the training loop for the Generative Adversarial Network.

    Args:
        generator (nn.Module): The generator neural network.
        discriminator (nn.Module): The discriminator neural network.
        dataloader (DataLoader): PyTorch DataLoader providing real images.
        device (torch.device): Compute device (CPU or CUDA).
        config (GANConfig): Hyperparameters configuration object.
    """
    
    # Initialize Loss function and Optimizers
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))

    # Ensure output directories exist
    os.makedirs(get_full_path(PATH_MODEL_WEIGHTS), exist_ok=True)
    os.makedirs(get_full_path(PATH_MODEL_SINGLE_EPOCH_WEIGHTS), exist_ok=True)
    
    # Dedicated directory for visual output during training
    visual_output_dir = get_full_path(PATH_MODEL_SINGLE_EPOCH_WEIGHTS, "training_progress_images")
    os.makedirs(visual_output_dir, exist_ok=True)

    # Fixed noise to evaluate generator's progress on the exact same latent vectors
    fixed_noise = torch.randn(16, config.latent_dim, device=device)

    print(f"Starting Training Loop on {device} for {config.epochs} epochs...")

    for epoch in range(config.epochs):
        
        epoch_iterator = tqdm(
            dataloader, 
            desc=f"Epoch [{epoch + 1}/{config.epochs}]", 
            leave=True
        )
        
        for i, real_imgs in enumerate(epoch_iterator):
            real_imgs = real_imgs.to(device)
            current_batch_size = real_imgs.size(0)
            
            # Ground truth labels
            valid = torch.ones((current_batch_size, 1), device=device)
            fake = torch.zeros((current_batch_size, 1), device=device)

            optimizer_G.zero_grad()
            
            z = torch.randn(current_batch_size, config.latent_dim, device=device)
            gen_imgs = generator(z)
            
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()

            epoch_iterator.set_postfix({
                "D_Loss": f"{d_loss.item():.4f}", 
                "G_Loss": f"{g_loss.item():.4f}"
            })
        
        # ---------------------
        #  Artifacts Generation
        # ---------------------
        # Save evaluation images
        if (epoch + 1) % config.save_image_freq == 0 or epoch == 0:
            with torch.no_grad():
                test_images = generator(fixed_noise)
                image_path = get_full_path(visual_output_dir, f"epoch_{epoch + 1:03d}.png")
                # Normalize shifts domain from [-1, 1] to [0, 1] for saving
                vutils.save_image(test_images.data, image_path, normalize=True, value_range=(-1, 1))
                
        # Save model weights checkpoints (Single Epoch)
        if (epoch + 1) % config.save_weight_freq == 0:
            checkpoint_path = get_full_path(PATH_MODEL_SINGLE_EPOCH_WEIGHTS, f"generator_epoch_{epoch + 1}.pth")
            torch.save(generator.state_dict(), checkpoint_path)

    # ---------------------
    #  Final Model Save
    # ---------------------
    final_model_path = get_full_path(PATH_MODEL_WEIGHTS, "generator_final.pth")
    torch.save(generator.state_dict(), final_model_path)
    print(f"Training Complete. Final model saved to {final_model_path}")


def main():
    """
    Main execution pipeline: initialization, dataset loading, and triggering the training process.
    """
    config = GANConfig()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Allocated Device: {device}")

    print(f"Loading dataset from: {PATH_DATASET_GENERATED}")
    dataset = GANTerrainDataset(image_dir=PATH_DATASET_GENERATED, size=config.image_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    generator = Generator(config.latent_dim).to(device)
    discriminator = Discriminator().to(device)

    train_gan(
        generator=generator, 
        discriminator=discriminator, 
        dataloader=dataloader, 
        device=device, 
        config=config
    )

if __name__ == "__main__":
    main()