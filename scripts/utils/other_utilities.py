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
import shutil
from sklearn.cluster import KMeans


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
    print(f"Read from folder: {cartella_sorgente}")
    
    if not os.path.exists(cartella_destinazione):
        os.makedirs(cartella_destinazione)

    files_exr = [f for f in os.listdir(cartella_sorgente) if f.lower().endswith('.exr')]
        
    for nome_file in tqdm(files_exr, desc="Conversion EXR -> PNG", unit="file"):
        percorso_exr = os.path.join(cartella_sorgente, nome_file)
        try:
            # Usa ImageIO per leggere l'EXR (Non fallisce mai)
            img_exr = iio.imread(percorso_exr)
                
            # Estrae il singolo canale (solitamente R) se è multicanale
            if len(img_exr.shape) == 3:
                img_gray = img_exr[:, :, 0] 
            else:
                img_gray = img_exr
                
            # Taglia i valori sotto 0 e sopra 1 per sicurezza
            img_clipped = np.clip(img_gray, 0.0, 1.0)
                
            # Moltiplica per il massimo del 16-bit intero (65535)
            img_16bit = (img_clipped * 65535.0).astype(np.uint16)
                
            # Salvataggio in PNG a 16-bit usando ImageIO
            nome_uscita = os.path.splitext(nome_file)[0] + ".png"
            percorso_uscita = os.path.join(cartella_destinazione, nome_uscita)
                
            iio.imwrite(percorso_uscita, img_16bit)
                
            tqdm.write(f"Converted in PNG 16-bit: {nome_uscita}")
            file_convertiti += 1
                
        except Exception as e:
            tqdm.write(f"Error with {nome_file}: {e}")
                
    print(f"Finished! Total file converted: {file_convertiti}")

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
    # Graph 1: Average Loss (With Trend Line)
    # ==========================================
    axs[0, 0].plot(epochs, losses, marker='o', color='tab:red', alpha=0.3, label='Real Loss')
    if len(losses) > 3:
        trend = np.convolve(losses, np.ones(3)/3, mode='valid')
        axs[0, 0].plot(epochs[len(epochs) - len(trend):], trend, color='darkred', linewidth=2, label='Trend')
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


def _read_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def _moving_average(values, window):
    """Trailing moving average ('valid' mode). Returns None if it cannot be computed."""
    if window <= 1 or len(values) < window:
        return None
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _percent_change(values, ref_epochs):
    """Percentage change with respect to the mean of the first `ref_epochs` values."""
    return 100.0 * (values / values[:ref_epochs].mean() - 1.0)


def _relative_noise(values):
    """High-frequency noise: std of consecutive differences / sqrt(2), as a % of the mean."""
    return 100.0 * np.diff(values).std(ddof=1) / np.sqrt(2) / values.mean()


def _trend(values, k):
    """Mean of the last k values vs mean of the first k values, in %."""
    return 100.0 * (values[-k:].mean() - values[:k].mean()) / values[:k].mean()


def plot_loss_by_timestep_bin(csv_path, avg_csv_path=None, val_csv_path=None,
                              ref_epochs=3, smooth_window=3):
    """
    Plots the training loss grouped by timestep bin and, optionally, compares it with
    the overall average loss and with the validation loss regrouped on the same bins.

    Motivation
    ----------
    During training a timestep t in [0, 999] is drawn at random for every sample, and
    the noise-prediction MSE depends strongly on t. The epoch-average loss therefore
    mixes values of very different scale. Averaging the loss inside timestep bins
    separates these scales, so the relative improvement of each bin can be followed
    on its own.

    Parameters
    ----------
    csv_path : str
        Per-bin training log (``loss_history_bins.csv``). Expected columns: ``epoch``
        plus one column per bin named ``loss_bin<i>_t<start>-<end>`` (e.g.
        ``loss_bin0_t0-333``), where ``<start>-<end>`` is the timestep range of the bin.
    avg_csv_path : str, optional
        Standard training log (``loss_history.csv``) with the ``average_loss`` column.
        If given, the overall average loss is drawn as a gray dashed reference.
    val_csv_path : str, optional
        Validation log (``validation_history.csv``) with columns ``epoch`` and
        ``val_t<timestep>``. The validation timesteps are assigned to the training bin
        that contains them and averaged inside the bin. Epoch 0 (base model) is ignored.
    ref_epochs : int
        Number of initial epochs whose mean is used as the reference (0 %) of the
        percentage panel, for the training and the validation curves alike.
    smooth_window : int
        Window, in epochs, of the trailing moving average drawn over the raw points.

    Output
    ------
    Saves ``<csv_name>_bins.png`` next to the CSV and shows a figure with two panels:
    1. Training loss per bin (log scale, because the bins differ by orders of magnitude).
    2. Percentage change per bin with respect to the mean of the first ``ref_epochs``
       epochs: raw points (transparent), moving average (solid line) and, if provided,
       validation (dashed line) and overall average loss (gray).
    A summary table with the residual noise and the overall trend of every series is
    also printed.

    Notes
    -----
    - The reference is the mean of the first epochs and not a single epoch: the first
      epoch is a noisy point and it is averaged over weights that change a lot during
      the epoch, so using it alone exaggerates the later improvement.
    - The validation uses a few fixed timesteps, whereas a training bin averages over a
      whole range of timesteps. Bins with a single validation timestep, or with
      timesteps that are not spread over the range, are only an approximation:
      compare trends, not absolute values.
    - Training and validation are comparable only if they use the same noise offset
      and, for the validation, the same weights of the same run.
    - The noise in the summary is the std of consecutive differences divided by sqrt(2),
      as a percentage of the series mean. It does not depend on the slow trend.
    """
    rows = _read_csv(csv_path)
    if len(rows) == 0:
        print("The CSV file is empty.")
        return

    epochs = np.array([int(r["epoch"]) for r in rows])
    ref_epochs = max(1, min(ref_epochs, len(epochs)))

    # --- training bins, with their timestep range read from the column name ---
    bins = []
    for key in [k for k in rows[0] if k.startswith("loss_bin")]:
        m = re.search(r"_t(\d+)-(\d+)$", key)
        if m is None:
            continue
        bins.append({
            "label": f"t={m.group(1)}-{m.group(2)}",
            "start": int(m.group(1)),
            "end": int(m.group(2)),
            "values": np.array([float(r[key]) for r in rows]),
        })
    if len(bins) == 0:
        print("No 'loss_bin...' columns found in the CSV.")
        return

    # --- optional overall average loss, aligned on the epochs ---
    avg_values = None
    if avg_csv_path and os.path.exists(avg_csv_path):
        by_epoch = {int(r["epoch"]): float(r["average_loss"]) for r in _read_csv(avg_csv_path)}
        if all(int(e) in by_epoch for e in epochs):
            avg_values = np.array([by_epoch[int(e)] for e in epochs])
        else:
            print("The epochs of the average-loss CSV do not match: reference skipped.")

    # --- optional validation, regrouped on the training bins ---
    val_epochs, val_series = None, {}
    if val_csv_path and os.path.exists(val_csv_path):
        val_rows = [r for r in _read_csv(val_csv_path) if int(r["epoch"]) >= 1]
        if val_rows:
            val_epochs = np.array([int(r["epoch"]) for r in val_rows])
            t_cols = {int(k[5:]): k for k in val_rows[0] if re.fullmatch(r"val_t\d+", k)}
            for b in bins:
                cols = [c for t, c in t_cols.items() if b["start"] <= t <= b["end"]]
                if cols:
                    val_series[b["label"]] = np.array(
                        [np.mean([float(r[c]) for c in cols]) for r in val_rows]
                    )
                else:
                    print(f"No validation timestep falls in the bin {b['label']}.")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axs = plt.subplots(1, 1, figsize=(10, 7))


    ax = axs
    for i, b in enumerate(bins):
        c = colors[i % len(colors)]
        pct = _percent_change(b["values"], ref_epochs)
        smooth = _moving_average(pct, smooth_window)
        ax.plot(epochs, pct, marker='o', markersize=3, linewidth=1, color=c, alpha=0.3,
                label=None if smooth is not None else f"{b['label']} (training)")
        if smooth is not None:
            ax.plot(epochs[smooth_window - 1:], smooth, color=c, linewidth=2.5,
                    label=f"{b['label']} (training)")
        if b["label"] in val_series:
            v = val_series[b["label"]]
            ax.plot(val_epochs, _percent_change(v, min(ref_epochs, len(v))), marker='s',
                    markersize=4, linestyle='--', linewidth=1.5, color=c,
                    label=f"{b['label']} (validazione)")
    if avg_values is not None:
        pct = _percent_change(avg_values, ref_epochs)
        smooth = _moving_average(pct, smooth_window)
        if smooth is not None:
            ax.plot(epochs[smooth_window - 1:], smooth, color='gray', linestyle=':',
                    linewidth=2.5, label="average_loss (training)")
        else:
            ax.plot(epochs, pct, color='gray', linestyle=':', linewidth=2, label="average_loss (training)")
    ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_title(f"Variazione rispetto alla media delle prime {ref_epochs} epoche (%)\n"
                 f"(linea continua: media mobile su {smooth_window} epoche)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Variazione (%)")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=8)

    plt.tight_layout()
    output_png = os.path.splitext(csv_path)[0] + "_bins.png"
    plt.savefig(output_png, dpi=300)
    print(f"\nGraph saved: {output_png}")


    print(f"\n{'Serie':<24}{'Rumore (%)':>12}{f'Ultime/prime {ref_epochs} ep. (%)':>26}")
    series = [(b["label"], b["values"]) for b in bins]
    if avg_values is not None:
        series.insert(0, ("average_loss", avg_values))
    if len(epochs) > ref_epochs:
        for name, v in series:
            print(f"{name:<24}{_relative_noise(v):>12.2f}{_trend(v, ref_epochs):>26.2f}")

    plt.show()



def plot_validation_metrics(csv_path):
    """
    Reads the validation CSV produced by validate_checkpoints.py and generates
    a 2x2 dashboard showing how each LoRA checkpoint performs on unseen heightmaps.
 
    Method
    ------
    For every checkpoint, the validation script measures the noise-prediction MSE on
    held-out latents using FIXED timesteps and FIXED noise. Because both are identical
    for every checkpoint, differences between epochs depend only on the model weights,
    not on random sampling. This is what makes the curve smooth, unlike the training
    loss, where timesteps and noise are drawn at random at every step.
 
    The row with ``epoch == 0`` is the base SDXL model (LoRA disabled) and is used as
    the reference. If it is missing, the first available checkpoint is used instead.
 
    Expected CSV columns
    --------------------
    epoch, val_loss, val_t<timestep> ... (one column per fixed timestep)
 
    ``val_loss`` is the plain mean of the per-timestep losses.
 
    Dashboard panels
    ----------------
    1. Validation loss per epoch (zoomed y-axis) with the reference as a dotted line
       and the best checkpoint highlighted.
    2. Relative change of the validation loss with respect to the reference, in %.
       This shows the real magnitude of the improvement.
    3. Relative change per timestep, in %. Low timesteps mostly reflect fine details,
       high timesteps mostly reflect global structure and mean height.
    4. Absolute loss per timestep (log scale), because the absolute values differ by
       orders of magnitude between low and high timesteps.
 
    Notes
    -----
    - The absolute validation loss is NOT comparable with the training ``average_loss``:
      training samples timesteps uniformly at random in [0, 999], while validation uses
      a small fixed set. Compare shapes and relative changes, not raw values.
    - The relative change in panel 3 is more informative than the absolute values,
      because the mean ``val_loss`` is dominated by the low timesteps, where the losses
      are largest.
    """
 
    if not os.path.exists(csv_path):
        print(f"Error: The file {csv_path} does not exist. Run the validation script first.")
        return
 
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
 
    if len(rows) == 0:
        print("The CSV file is empty.")
        return
 
    epochs = np.array([int(r["epoch"]) for r in rows])
    val_loss = np.array([float(r["val_loss"]) for r in rows])
    t_keys = [k for k in rows[0] if k.startswith("val_t")]
    t_values = {k: np.array([float(r[k]) for r in rows]) for k in t_keys}
 
    has_base = epochs[0] == 0
    ref_label = "SDXL base (epoch 0)" if has_base else f"first checkpoint (epoch {epochs[0]})"
    ref_loss = val_loss[0]
 
    # Best checkpoint among the trained ones (exclude the base model)
    trained_mask = epochs > 0 if has_base else np.ones_like(epochs, dtype=bool)
    if not trained_mask.any():
        print("The CSV contains only the base model: nothing to compare.")
        return
    best_idx = int(np.argmin(np.where(trained_mask, val_loss, np.inf)))
 
    def rel_change(values):
        return 100.0 * (values - values[0]) / values[0]
 
    rel_total = rel_change(val_loss)
 
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Validation on unseen heightmaps (fixed timesteps and noise)", fontsize=18, fontweight='bold')

    ax = axs[0, 0]
    ax.plot(epochs, val_loss, marker='s', color='tab:blue', linewidth=2, label='Validation loss')
    ax.axhline(ref_loss, color='gray', linestyle=':', label=f'{ref_label}: {ref_loss:.5f}')
    ax.scatter([epochs[best_idx]], [val_loss[best_idx]], s=160, facecolors='none',
               edgecolors='tab:red', linewidths=2, zorder=5,
               label=f'Best: epoch {epochs[best_idx]} ({val_loss[best_idx]:.5f})')
    ax.set_title("Validation Loss (zoomed y-axis)", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    ax = axs[0, 1]
    ax.plot(epochs, rel_total, marker='o', color='tab:green', linewidth=2)
    ax.axhline(0.0, color='gray', linestyle=':')
    ax.set_title(f"Validation loss change vs {ref_label}", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Change (%)")
    ax.grid(True, linestyle='--', alpha=0.7)

    ax = axs[1, 0]
    for k in t_keys:
        ax.plot(epochs, rel_change(t_values[k]), marker='o', markersize=4, label=f"t={k[5:]}")
    ax.axhline(0.0, color='gray', linestyle=':')
    ax.set_title("Change per timestep vs reference", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Change (%)")
    ax.grid(True, linestyle='--', alpha=0.7)
    if t_keys:
        ax.legend()
 
    ax = axs[1, 1]
    for k in t_keys:
        ax.plot(epochs, t_values[k], marker='o', markersize=4, label=f"t={k[5:]}")
    ax.set_yscale("log")
    ax.set_title("Validation loss per timestep (log scale)", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, linestyle='--', alpha=0.7)
    if t_keys:
        ax.legend()
 
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
 
    output_png = os.path.splitext(csv_path)[0] + "_dashboard.png"
    plt.savefig(output_png, dpi=300)
    print(f"\nGraph saved: {output_png}")
 
    print(f"\nReference ({ref_label}): {ref_loss:.6f}")
    print(f"Best epoch: {epochs[best_idx]} -> {val_loss[best_idx]:.6f} ({rel_total[best_idx]:+.2f}% vs reference)")
    for k in t_keys:
        print(f"  t={k[5:]:>4}: {t_values[k][0]:.6f} -> {t_values[k][-1]:.6f} ({rel_change(t_values[k])[-1]:+.2f}%)")
 
    plt.show()

def heightmaps_analysis(image_path_source):
    """
    Extracts statistical features (standard deviation and entropy) from a heightmap image.

    This function loads an image, converts it to grayscale if necessary, and computes
    two key terrain descriptors: standard deviation (representing roughness/elevation variety) 
    and Shannon entropy (representing texture complexity and randomness).

    Args:
        image_path_source (str): Path to the heightmap image file to analyze.

    Returns:
        list[float] or None: A list containing [std_dev, entropy] if successful, 
                             or None if an error occurs during processing.
    """

    try:
        img = iio.imread(image_path_source)

        if len(img.shape) == 3 and img.shape[2] >= 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

        std_dev = np.std(img)

        min_val, max_val = np.min(img), np.max(img)

        if min_val == max_val:
            hist, _ = np.histogram(img, bins=256)
        else:
            hist, _ = np.histogram(img, bins=256, range=(min_val, max_val))
        
        total_pixels = hist.sum()
        if total_pixels == 0:
            print(f"Warning: Cannot build histogram for {image_path_source}")
            return None
        
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))

        return [std_dev, entropy]
    except Exception as e:
        print(f"Impossibile to read {image_path_source}: {e}")
        return None

def feature_analysis_heightmaps(image_path_source, image_path_destination, target_images, n_clusters):
    """
    Analyzes heightmap images, clusters them by features, and creates a balanced, diverse subset.

    This method extracts features from all valid images in a source directory, categories 
    them into distinct soil or terrain classes using K-Means clustering, and then performs 
    uniform stratified sampling. It ensures that the final dataset contains a balanced 
    number of representative images from each category to maximize diversity.

    Args:
        image_path_source (str): Path to the directory containing the source heightmap images.
        image_path_destination (str): Path where the optimized, sampled dataset will be saved.
        target_images (int): Total number of images desired in the final dataset.
        n_clusters (int): Number of soil categories/clusters to divide the terrain data into.

    Returns:
        None
    """
    
    features = []
    valid_paths = []
    all_files = os.listdir(image_path_source)
    valid_files = [
        f for f in all_files
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]

    print("Analyzing heightmaps features...")
    for filename in tqdm(valid_files, desc="Processing images"):
        path = os.path.join(image_path_source, filename)
        feat = heightmaps_analysis(path)
        if feat is not None:
            features.append(feat)
            valid_paths.append(path)

    features = np.array(features)
    print(f"Complete analysis on {len(valid_paths)} immages.")

    print(f"Clustering into {n_clusters} soil categories...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    counts = np.bincount(labels)
    for idx, count in enumerate(counts):
        print(f"   - Category {idx}: {count} images found")

    os.makedirs(image_path_destination, exist_ok=True)
    images_per_cluster = target_images // n_clusters

    files_to_copy = []
    for cluster_idx in range(n_clusters):
        indices = np.where(labels == cluster_idx)[0]

        if len(indices) > images_per_cluster:
            step = len(indices) / images_per_cluster
            selected_indices = [indices[int(i * step)] for i in range(images_per_cluster)]
        else:
            selected_indices = indices

        for idx in selected_indices:
            files_to_copy.append(valid_paths[idx])

    print(f"Copying the top {len(files_to_copy)} diverse images to '{image_path_destination}'...")

    for src_path in tqdm(files_to_copy, desc='Copying files'):
        dst_path = os.path.join(image_path_destination, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)

    print(f"Done! Your new diverse and optimized dataset is ready in: {image_path_destination}")