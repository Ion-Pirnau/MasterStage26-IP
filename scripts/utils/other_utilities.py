import tifffile as tiff
import os
from pathlib import Path
import re
from tqdm import tqdm
import numpy as np
from utils.utils_method import path_exists
import imageio.v3 as iio
import csv
import os
import matplotlib.pyplot as plt


def geotiff_to_off(tif_path, out_path, downsample=1, xy_resolution_meters=10.0, z_scale=1.0) -> bool:
    """
    Reads a GeoTIFF file and generates a 3D mesh exported in OFF format.

    :param tif_path: Path to the input GeoTIFF file.
    :param obj_path: Path to the output OBJ file.
    :param downsample: Sampling factor used to reduce mesh resolution 
                       (e.g., 2 halves the number of sampled pixels). 
                       Useful for processing very large datasets.
    :param xy_resolution_meters: Spatial resolution of each pixel in meters 
                                 (e.g., 1/3 arc-second DEM ≈ 10 meters per pixel), 
                                 used to scale the X and Y coordinates.
    :param z_scale: Vertical scaling factor applied to elevation values 
                    in order to exaggerate or reduce terrain height.

    Returns:
        bool: True if the conversion was successful, False otherwise.

    Raises:
        FileNotFoundError: If the input file does not exist.
        IOError: If an error occurs during file writing.
    """

    if not path_exists(tif_path):
        print(f"Error: File '{tif_path}' does not exist.")
        return False

    if downsample < 1:
        print("Error: downsample must be >= 1.")
        return False

    print(f"Reading file {tif_path}...")

    try:
        elevation = tiff.imread(tif_path)[::downsample, ::downsample]

        h, w = elevation.shape
        num_vertices = h * w
        print(f"Generation mesh from Grid: {w}x{h} ({w * h} vertices)...")

        # 1. GENERATION VERTICES (X, Y, Z)
        x = np.arange(w) * (xy_resolution_meters * downsample)
        y = np.arange(h) * (xy_resolution_meters * downsample)
        
        y = y[::-1] 
        
        xx, yy = np.meshgrid(x, y)

        vertices = np.column_stack((
            xx.ravel(), 
            yy.ravel(), 
            elevation.ravel() * z_scale
        ))

        # 2. TRIANGLES GENERATION (FACES)
        r, c = np.mgrid[0:h-1, 0:w-1]
        
        top_left = r * w + c
        top_right = top_left + 1
        bottom_left = (r + 1) * w + c
        bottom_right = bottom_left + 1

        # Triangle 1: Top-Left, Bottom-Left, Bottom-Right
        tri1 = np.stack((top_left, bottom_left, bottom_right), axis=-1).reshape(-1, 3)
        # Triangle 2: Top-Left, Bottom-Right, Top-Right
        tri2 = np.stack((top_left, bottom_right, top_right), axis=-1).reshape(-1, 3)
        
        faces = np.vstack((tri1, tri2))
        num_faces = len(faces)

        # 3. SAVE IN FORMAT OFF
    
        print(f"Saved in: {out_path} ...")

        with open(out_path, 'w') as f:

            f.write("OFF\n")
            f.write(f"{num_vertices} {num_faces} 0\n")
            
            for v in vertices:
                f.write(f"{v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
            
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

        print("Conversion completed successfully.")
        return True

    except (FileNotFoundError, IOError) as e:
        print(f"Error during conversion: {e}")
        return False
    

def geotiff_to_obj(tif_path, out_path, grid_size=10.0, downsample=1, xy_resolution_meters=10.0, z_scale=1.0) -> bool:
    """
    Reads a GeoTIFF file and generates a 3D mesh exported in OBJ format.

    :param tif_path: Path to the input GeoTIFF file.
    :param obj_path: Path to the output OBJ file.
    :param grid_size: Target size for the largest dimension of the mesh (e.g., 10 units).
    :param downsample: Sampling factor used to reduce mesh resolution 
                       (e.g., 2 halves the number of sampled pixels). 
                       Useful for processing very large datasets.
    :param xy_resolution_meters: Spatial resolution of each pixel in meters 
                                 (e.g., 1/3 arc-second DEM ≈ 10 meters per pixel), 
                                 used to scale the X and Y coordinates.
    :param z_scale: Vertical scaling factor applied to elevation values 
                    in order to exaggerate or reduce terrain height.

    Returns:
        bool: True if the conversion was successful, False otherwise.

    Raises:
        FileNotFoundError: If the input file does not exist.
        IOError: If an error occurs during file writing.
    """

    if not path_exists(tif_path):
        print(f"Error: File '{tif_path}' does not exist.")
        return False

    if downsample < 1:
        print("Error: downsample must be >= 1.")
        return False

    print(f"Reading file {tif_path}...")

    try:
        elevation = tiff.imread(tif_path)[::downsample, ::downsample]

        h, w = elevation.shape
        print(f"Generation mesh from Grid: {w}x{h} ({w * h} vertices)...")

         # NORMALIZATION AND CENTERING
        real_width = (w - 1) * (xy_resolution_meters * downsample)
        real_height = (h - 1) * (xy_resolution_meters * downsample)


        max_real_dim = max(real_width, real_height)
        global_scale = grid_size / max_real_dim

        target_width = real_width * global_scale
        target_height = real_height * global_scale

        x = np.linspace(-target_width / 2, target_width / 2, w)
        y = np.linspace(target_height / 2, -target_height / 2, h)

        xx, yy = np.meshgrid(x, y)

        min_elevation = np.nanmin(elevation)
        normalized_elevation = (elevation - min_elevation) * global_scale * z_scale

        # 1. GENERATION VERTICES (X, Y, Z)
        vertices = np.column_stack((
            xx.ravel(), 
            yy.ravel(), 
            normalized_elevation.ravel()
        ))

        # 2. TRIANGLES GENERATION (FACES)
        r, c = np.mgrid[0:h-1, 0:w-1]
        
        top_left = r * w + c
        top_right = top_left + 1
        bottom_left = (r + 1) * w + c
        bottom_right = bottom_left + 1

        # Triangle 1: Top-Left, Bottom-Left, Bottom-Right
        tri1 = np.stack((top_left, bottom_left, bottom_right), axis=-1).reshape(-1, 3)
        # Triangle 2: Top-Left, Bottom-Right, Top-Right
        tri2 = np.stack((top_left, bottom_right, top_right), axis=-1).reshape(-1, 3)
        
        faces = np.vstack((tri1, tri2)) + 1

        # 3. SAVE IN FORMAT OBJ
    
        print(f"Saved in: {out_path} ...")

        with open(out_path, 'w') as f:

            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")

        print("Conversion completed successfully.")
        return True

    except (FileNotFoundError, IOError) as e:
        print(f"Error during conversion: {e}")
        return False


def converti_exr_a_png_16bit(cartella_sorgente, cartella_destinazione):
    file_convertiti = 0
    print(f"Leggendo la cartella: {cartella_sorgente}")
    
    if not os.path.exists(cartella_destinazione):
        os.makedirs(cartella_destinazione)
        
    for nome_file in os.listdir(cartella_sorgente):
        if nome_file.lower().endswith('.exr'):
            percorso_exr = os.path.join(cartella_sorgente, nome_file)
            
            try:
                # Usa ImageIO per leggere l'EXR (Non fallisce mai)
                img_exr = iio.imread(percorso_exr)
                
                # Estrae il singolo canale (solitamente R) se è multicanale
                if len(img_exr.shape) == 3:
                    img_gray = img_exr[:, :, 0] 
                else:
                    img_gray = img_exr

                print(f"\nFile: {nome_file}")
                print("Shape:", img_gray.shape)
                print("Dtype:", img_gray.dtype)
                print("Min:", np.min(img_gray))
                print("Max:", np.max(img_gray))
                print("Mean:", np.mean(img_gray))
                print("Std:", np.std(img_gray))
                print("Percentile 1%:", np.percentile(img_gray, 1))
                print("Percentile 99%:", np.percentile(img_gray, 99))
                
                # Taglia i valori sotto 0 e sopra 1 per sicurezza
                img_clipped = np.clip(img_gray, 0.0, 1.0)
                
                # Moltiplica per il massimo del 16-bit intero (65535)
                img_16bit = (img_clipped * 65535.0).astype(np.uint16)
                
                # Salvataggio in PNG a 16-bit usando ImageIO
                nome_uscita = os.path.splitext(nome_file)[0] + ".png"
                percorso_uscita = os.path.join(cartella_destinazione, nome_uscita)
                
                iio.imwrite(percorso_uscita, img_16bit)
                
                print(f"Convertito in PNG 16-bit: {nome_uscita}")
                file_convertiti += 1
                
            except Exception as e:
                print(f"ERRORE GRAVE con il file {nome_file}: {e}")
                
    print(f"Finito! File convertiti: {file_convertiti}")

def inspect_png(path):
    """
    Inspects a PNG heightmap directly using ImageIO, without applying
    any preprocessing or conversion.

    This test verifies whether the 16-bit height values are preserved
    correctly during the EXR-to-PNG conversion.
    """
    print("\n===== PNG TEST =====")

    img = iio.imread(path)

    print("Shape:", img.shape)
    print("Dtype:", img.dtype)
    print("Min:", img.min())
    print("Max:", img.max())
    print("Mean:", img.mean())
    print("Std:", img.std())

    print("===== END PNG TEST =====\n")

def clean_caption_text(caption):
    """Cleans and compacts a single caption, removing redundant parameters
    and shortening descriptions to remain within the 77-token CLIP limit.
    """

    ELEMENTS_TO_REMOVE = [
        "base noise",
        "secondary noise",
        "force field",
        "inner radius",
        "outer radius",
        "mountain scale"
    ]

    tags = [t.strip() for t in caption.split(",") if t.strip()]
    cleaned_tags = []

    for tag in tags:
        tag_lower = tag.lower()

        if any(rem in tag_lower for rem in ELEMENTS_TO_REMOVE):
            continue

        tag_clean = tag
        tag_clean = re.sub(r'\b(mountain|crater|erosion|fan)\s+', '', tag_clean, flags=re.IGNORECASE)

        cleaned_tags.append(tag_clean)

    return ', '.join(cleaned_tags)


def optimize_caption_files(
    cartella_sorgente, cartella_destinazione):
    """Reads caption text files from a source directory, optimizes their content,

    and writes the processed text to a destination directory.

    The function scans all text files in the specified source folder. It filters
    out
    technical parameters (Grid Size, Resolution, Material activation, and
    Seeds),
    rounds long decimals to three decimal places, and strips redundant punctuation
    to optimize token count for CLIP/SDXL model inputs.

    Parameters
    ----------
    cartella_sorgente : str
        The path to the directory containing the original caption text files.
    cartella_destinazione : str
        The path to the directory where the optimized caption files will be
        saved.

    Returns
    -------
    None

    Notes
    -----
    - Only files with a '.txt' extension are processed.
    - Specifically removes technical attributes: Grid Size, XY Resolution,
    Resolution,
      Material Type activation, and any seed parameter.
    """
    if not os.path.exists(cartella_destinazione):
        os.makedirs(cartella_destinazione)

    print(f"Working on folder: {cartella_sorgente}")

    tutti_i_file = [
        f for f in os.listdir(cartella_sorgente) if f.endswith('.txt')
    ]

    if not tutti_i_file:
        print(f'No file .txt found in: {cartella_sorgente}')
        return 

    print(f"Found {len(tutti_i_file)} files to elaborate...")

    for nome_file in tqdm(
        tutti_i_file, desc="Optimize captions", unit='file'
    ):
        percorso_sorgente = os.path.join(cartella_sorgente, nome_file)
        percorso_destinazione = os.path.join(cartella_destinazione, nome_file)

        with open(percorso_sorgente, "r", encoding="utf-8") as f:
            caption = f.read()

        caption_ottimizzata = clean_caption_text(caption=caption)

        with open(percorso_destinazione, "w", encoding="utf-8") as f:
            f.write(caption_ottimizzata)

    print(f"Finished! Total captions optimized: {len(tutti_i_file)}")

def plot_training_metrics(csv_path):
    """
    Reads the Stable Diffusion training CSV file and generates 
    a visual dashboard with 4 plots to analyze the learning process.
    """

    
    if not os.path.exists(csv_path):
        print(f"Error: The file {csv_path} does not exist. Make sure that the training has started.")
        return

    epochs = []
    losses = []
    lrs = []
    mean_abs = []
    mean_change = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["average_loss"]))
            lrs.append(float(row["learning_rate"]))
            mean_abs.append(float(row["lora_mean_abs"]))
            mean_change.append(float(row["lora_mean_change"]))

    if len(epochs) == 0:
        print("Il file CSV è vuoto.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Analisi Addestramento LoRA SDXL", fontsize=18, fontweight='bold')

    # ==========================================
    # Graph 1: Average Loss
    # ==========================================
    axs[0, 0].plot(epochs, losses, marker='o', color='tab:red', linewidth=2)
    axs[0, 0].set_title("Average Loss", fontsize=14)
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("MSE Loss")
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)

    # ==========================================
    # Graph 2: Learning Rate
    # ==========================================
    axs[0, 1].plot(epochs, lrs, marker='s', color='tab:blue', linewidth=2)
    axs[0, 1].set_title("Learning Rate (Cosine Scheduler)", fontsize=14)
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("LR")
    # Usa la notazione scientifica (es. 5e-5) per l'asse Y
    axs[0, 1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    # ==========================================
    # Graph 3: LoRA Mean Abs
    # ==========================================
    axs[1, 0].plot(epochs, mean_abs, marker='^', color='tab:purple', linewidth=2)
    axs[1, 0].set_title("LoRA Mean Abs (Magnitudine Pesi)", fontsize=14)
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Valore Assoluto Medio")
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)

    # ==========================================
    # Graph 4: LoRA Mean Change
    # ==========================================
    axs[1, 1].plot(epochs, mean_change, marker='D', color='tab:green', linewidth=2)
    axs[1, 1].set_title("LoRA Mean Change (Apprendimento)", fontsize=14)
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Variazione Media da Zero")
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_png = os.path.splitext(csv_path)[0] + "_dashboard.png"
    plt.savefig(output_png, dpi=300)
    print(f"\nGraph saved: {output_png}")

    plt.show()