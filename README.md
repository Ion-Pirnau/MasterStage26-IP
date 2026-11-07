# Procedural Terrain Generation & AI Analysis for Geological Applications

![Blender](https://img.shields.io/badge/Blender-4.2-2589BD?style=for-the-badge&logo=blender&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

Select your language / Seleziona la lingua:
- [🇮🇹 Italiano](#versione-italiana)
- [🇬🇧 English](#english-version)

---

<a name="english-version"></a>
# 🇬🇧 English Version

This repository contains the work developed during my Master's Degree thesis internship: a multi-stage pipeline for procedural terrain modeling, topographic rendering, and the use of Generative AI for geological data synthesis and analysis.

## Table of Contents
- [Project Overview](#project-overview)
- [Pipeline](#pipeline)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Author and Acknowledgements](#author-and-acknowledgements)

## Project Overview
The goal is to bridge procedural 3D modeling and modern Deep Learning. The project goes from manual and parametric terrain generation in Blender to the training of generative models (Latent Diffusion Models such as Stable Diffusion XL, and a GAN-based approach) that produce synthetic geological terrain datasets. The synthetic data is then validated with state-of-the-art segmentation and detection tools (SAM2/SAM3).

## Pipeline
1. **Procedural terrain generation** in Blender, driven by a JSON configuration file.
2. **Rendering** of grayscale heightmaps and images.
3. **Format conversion** between GeoTIFF/ASC, heightmaps and OBJ meshes.
4. **Generative model training** (SDXL with LoRA) on the generated data.
5. **Validation** of the synthetic terrains with SAM2/SAM3.

[TO DO: add a diagram or a short sentence on how data flows between the steps]

## Tech Stack
| Area | Tools |
|---|---|
| 3D modeling | Blender 4.2 (Python API) |
| Language | Python 3.11+ (developed with 3.11.9) |
| AI / ML | PyTorch, Stable Diffusion XL, LoRA, GAN, SAM2/SAM3 [TO DO: exact versions] |
| Geodata | GDAL, GeoTIFF, ASC |
| Rendering | Grayscale heightmaps |

## Repository Structure
```text
project/
│
├── requirements.txt
├── .gitignore
│
├── terrain_generator/                # Blender terrain generator
│   ├── terreno_procedurale_tesi_features.blend
│   ├── dataset_setting.json          # Configuration for terrain generation
│   └── README.md
│
├── scripts/                          # Python scripts
│   ├── generator/
│   ├── ml_gan_model/
│   ├── ml_sdxl_model/
│   ├── utils/
│   ├── main.py                       # Rendering pipeline
│   ├── main_convert_utls.py          # Conversions between file types
│   ├── main_convert_tiff_obj.py      # GeoTIFF -> triangular mesh from regular grid
│   ├── test_main.py                  # Tests
│   └── README.md
│
├── landscape_renders/                # Sample renders taken directly from the .blend file
│
├── data/                             # Generated data
│   ├── heightmaps/
│   ├── meshes/
│   ├── tiff_to_convert/
│   ├── model_weights/
│   └── renders/
│
└── README.md                         # This file
```

## Installation
```bash
git clone https://github.com/[TO DO: user]/[TO DO: repo].git
cd [TO DO: repo]

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
```

**Requirements**
- Blender 4.2 installed [TO DO: how Blender is located by the scripts, e.g. path in settings]
- Python 3.11+
- GDAL [TO DO: installation method used]
- NVIDIA GPU with CUDA for the AI part (see `scripts/README.md`)

## Usage
Short version (details in [`scripts/README.md`](scripts/README.md)):

```bash
cd scripts
python main.py
```

Configuration of the terrain generator is described in [`terrain_generator/README.md`](terrain_generator/README.md).

## Results
[DA COMPLETARE: insert 2-3 images from `landscape_renders/`, e.g. `![Sample terrain](landscape_renders/example.png)`, and optionally a comparison between real and synthetic terrains with SAM2/SAM3 outputs]

## Author and Acknowledgements
[DA COMPLETARE: name, university, supervisor, company/host institution, year]

License: [DA COMPLETARE]

[Back to top](#procedural-terrain-generation--ai-analysis-for-geological-applications)

---

<a name="versione-italiana"></a>
# 🇮🇹 Versione Italiana

Questa repository raccoglie il lavoro svolto durante lo **stage di tesi magistrale**: una pipeline in più fasi per la modellazione procedurale di terreni, il rendering topografico e l'uso dell'AI generativa per la sintesi e l'analisi di dati geologici.

## Indice
- [Panoramica](#panoramica)
- [Pipeline](#pipeline-1)
- [Tecnologie](#tecnologie)
- [Struttura della repository](#struttura-della-repository)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
- [Risultati](#risultati)
- [Autore e ringraziamenti](#autore-e-ringraziamenti)

## Panoramica
L'obiettivo è collegare la modellazione 3D procedurale e il Deep Learning moderno. Il progetto parte dalla generazione manuale e parametrica di terreni in Blender e arriva all'addestramento di modelli generativi (Latent Diffusion Model come Stable Diffusion XL, e un approccio basato su GAN) per produrre dataset sintetici di terreni geologici. I dati sintetici vengono poi validati con strumenti di segmentazione e detection allo stato dell'arte (SAM2/SAM3).

## Pipeline
1. **Generazione procedurale del terreno** in Blender, guidata da un file di configurazione JSON.
2. **Rendering** di heightmap in scala di grigi e immagini.
3. **Conversione di formato** tra GeoTIFF/ASC, heightmap e mesh OBJ.
4. **Addestramento dei modelli generativi** (SDXL con LoRA, GAN) sui dati generati.
5. **Validazione** dei terreni sintetici con SAM2/SAM3.

[DA COMPLETARE: aggiungi uno schema o una frase sul flusso dei dati tra i passaggi]

## Tecnologie
| Ambito | Strumenti |
|---|---|
| Modellazione 3D | Blender 4.2 (Python API) |
| Linguaggio | Python 3.11+ (sviluppato con 3.11.9) |
| AI / ML | PyTorch, Stable Diffusion XL, LoRA, GAN, SAM2/SAM3 [DA COMPLETARE: versioni esatte] |
| Geodati | GDAL, GeoTIFF, ASC |
| Rendering | Heightmap in scala di grigi |

## Struttura della repository
La struttura è descritta nella [versione inglese](#repository-structure); i commenti ai file sono in inglese.

## Installazione
```bash
git clone https://github.com/[DA COMPLETARE: utente]/[DA COMPLETARE: repo].git
cd [DA COMPLETARE: repo]

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
```

**Requisiti**
- Blender 4.2 installato [DA COMPLETARE: come gli script trovano Blender, es. percorso nelle impostazioni]
- Python 3.11+
- GDAL [DA COMPLETARE: metodo di installazione usato]
- GPU NVIDIA con CUDA per la parte AI (vedi `scripts/README.md`)

## Utilizzo
Versione breve (dettagli in [`scripts/README.md`](scripts/README.md)):

```bash
cd scripts
python main.py
```

La configurazione del generatore di terreni è descritta in [`terrain_generator/README.md`](terrain_generator/README.md).

## Risultati
[DA COMPLETARE: inserisci 2-3 immagini da `landscape_renders/`, es. `![Terreno di esempio](landscape_renders/esempio.png)`, ed eventualmente un confronto tra terreni reali e sintetici con gli output di SAM2/SAM3]

## Autore e ringraziamenti
[DA COMPLETARE: nome, università, relatore, azienda/ente ospitante, anno]

Licenza: [DA COMPLETARE]

[Torna su](#procedural-terrain-generation--ai-analysis-for-geological-applications)
