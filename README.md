# Procedural Terrain Generation & AI Analysis for Geological Applications

![Blender](https://img.shields.io/badge/Blender-2589BD?style=for-the-badge&logo=blender&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)


Select your language / Seleziona la lingua:
- [🇮🇹 Italiano](#versione-italiana)
- [🇬🇧 English](#english-version)

---

<a name="english-version"></a>
## English Version

This repository contains the work developed during my Master’s Degree Internship. The project focuses on a multi-stage pipeline for procedural terrain modeling, topographic rendering, and the application of Generative AI for geological data synthesis and analysis.

### 🎯 Project Overview
The core objective is to bridge the gap between procedural 3D modeling and modern Deep Learning. The project evolves from manual and parametric terrain generation in Blender to the fine-tuning of Latent Diffusion Models for creating synthetic geological datasets, which are subsequently validated using state-of-the-art segmentation and detection tools (SAM2/SAM3).

### 💻 Tech Stack
- **3D Modeling:** Blender (Python API)
- **Languages:** Python o C++
- **AI/ML:** PyTorch, Stable Diffusion, LoRA, SAM2/SAM3 (da definire meglio)
- **Geodata:** GDAL, GeoTIFF, ASC formats
- **Rendering:** Grayscale Maps

[Back to top](#procedural-terrain-generation--ai-analysis-for-geological-applications)

---

<a name="versione-italiana"></a>
## Versione Italiana

Questa repository ospita il progetto di **Stage Magistrale** focalizzato sulla generazione procedurale di terreni, modellazione 3D e l'applicazione di modelli di diffusione generativa per l'analisi geologica.

### 📌 Obiettivo del Progetto
L'obiettivo è creare una pipeline completa che parta dalla modellazione procedurale di dati topografici fino all'addestramento di modelli di Deep Learning (Diffusion Model) per la generazione sintetica di terreni, validati poi tramite strumenti di segmentazione e detection (SAM2/SAM3).

### 🛠️ Tech Stack
- **Software Modellazione 3d:** Blender `(used 4.2)` (Python API)
- **Linguaggi di Programmazione:** Python (3.11+ `(used 3.11.9)`) o C++
- **AI/ML:** PyTorch, Stable Diffusion, LoRA, SAM2/SAM3 (da definire meglio)
- **Geodata:** GDAL, GeoTIFF, ASC formats
- **Rendering:** Grayscale Maps


[Torna su](#procedural-terrain-generation--ai-analysis-for-geological-applications)


## 📁 Repository Structure / Struttura della Repository

```text
project/
│
├── requirements.txt
├── .gitignore
│
├── terrain_generator/                  # Blender terrain generator
│   ├── terrain_generator_mountains_plains_v1.blend
│   ├── dataset_setting.json            # Set configuration
│   └── README.md
│
├── scripts/                 # Python scripts
│   ├── generator/
│   ├── utils/
│   ├── main.py             # python script - for rendering pipeline
│   ├── main_convert_tiff_obj.py             # python script - for reading GeoTIFF and create a triangular mesh from regular grid
│   ├── test_main.py        # python script for test
│   └── README.md
│
├── landscape_renders/                  # Renders Image sample directly from BLENDER FILE
│
├── data/                    # Generated data
│   ├── heightmaps/
│   ├── meshes/
│   └── renders/
│
└── README.md