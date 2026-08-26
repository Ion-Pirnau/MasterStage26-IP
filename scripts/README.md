# Terrain Generation Pipeline

Scegli la tua lingua / Choose your language:
- [Italiano](#versione-italiana)
- [English](#english-version)

---

<a name="versione-italiana"></a>
# Versione Italiana
 
Per iniziare a generare i tuoi terreni procedurali, segui questi semplici passaggi:

### 1. 📖 Configurazione
Prima di avviare il processo, leggi attentamente il file **README** presente nella sottocartella delle impostazioni. Lì troverai la spiegazione dettagliata di tutti i parametri presenti nel file `settings.json`.

### 2. 🚀 Esecuzione
Per avviare la pipeline e iniziare la generazione automatizzata in Blender, esegui il file principale tramite terminale o il tuo editor Python:

```bash
python main.py
```

### 3. 🧠 Modello Stable Diffusion XL (SDXL)
Per la parte relativa alla generazione tramite modelli di machine learning com Stable Diffusion XL, è necessario installare le librerie specifiche per la GPU. Assicurarsi che vi siano presenti le seguenti le librerie:

```bash
# NOTA: Prima di procedere, verifica la versione di CUDA installata sul tuo sistema (es. tramite il comando 'nvcc --version' o 'nvidia-smi'). 

# Diciamo a pip dove trovare la versione specifica di PyTorch per CUDA
pip install torch torchvision torchaudio -index-url https://pytorch.org

# Librerie Hugging Face per la generazione
pip install diffusers transformers accelerate safetensors
```

[⬆ Torna su / Back to top](#terrain-generation-pipeline)

---

<a name="english-version"></a>
## English Version

To correctly set up and run the terrain generation pipeline, follow these steps:

### 1. 📖 Configuration
Before launching the script, please read the **README.md** file located inside the settings subfolder. It contains a full breakdown of the `settings.json` file, explaining how to adjust geometry, noise levels, and export options.

### 2. 🚀 Running the Pipeline
To start the automated workflow and begin generating terrains in Blender, simply run the main Python entry point:

```bash
python main.py
```

### 3. 🧠 Stable Diffusion XL (SDXL) Model
For the AI generation part, you must install specific libraries tailored for GPU usage. Ensure that you have the following libraries:

```bash
# NOTE: Before proceeding, check the CUDA version installed on your system (e.g., using the 'nvcc --version' or 'nvidia-smi' command).

# Tell pip where to find the specific PyTorch version for CUDA
pip install torch torchvision torchaudio -index-url https://pytorch.org

# Hugging Face libraries for generation
pip install diffusers transformers accelerate safetensors

```


[⬆ Torna su / Back to top](#terrain-generation-pipeline)