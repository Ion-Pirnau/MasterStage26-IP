# Terrain Generation Pipeline Configuration

Scegli la tua lingua / Choose your language:
* [Italiano](#versione-italiana)
* [English](#english-version)

---

<a name="versione-italiana"></a>
# Versione italiana

Questo documento descrive la struttura e i parametri del file `settings.json`. Il file è il cuore della configurazione per la **pipeline Python** di automazione in Blender.

## 🚀 Modalità d'Uso

Questo file è stato progettato per essere flessibile:

*   **Automazione Python:** Utilizzando lo script dedicato, è possibile processare il file JSON per generare dataset di terreni (mesh, render e mappe).
*   **Test in Blender:** In alternativa, è possibile utilizzare il file `terrain_generator_mountains_plains_v1.blend` per effettuare test visivi immediati, regolare i nodi e validare i parametri prima di lanciare la generazione massiva.

---

## 🛠️ Guida ai Parametri JSON

### 1. Dataset Settings
Controlla il comportamento globale della pipeline e le opzioni di esportazione.


| Campo | Descrizione |
| :--- | :--- |
| `generation_mode` | Modalità di generazione (es. `fixed` (singolo, valore standard) o `random` (generazione multipla, valori tra `min` e `max`)). |
| `num_terrains_to_generate` | Numero totale di varianti da creare. |
| `output_folder_...` | Cartelle di destinazione per Render, Mesh e Mappe. |
| `export_render_png` | Se `true`, salva il rendering visivo della scena. |
| `export_mesh_off` | Esporta la geometria in formato .off. |
| `export_heightmap_exr` | Salva la mappa delle altezze in 32-bit. |
| `export_heightmap_asc` | Esporta i dati altimetrici in formato testo ASCII Grid. |
| `export_hillshade` | Genera una mappa d'ombra del rilievo (Hillshade). |

### 2. Node Parameters
Parametri tecnici per la modellazione procedurale del terreno.

#### Configurazione Mesh
*   **Grid_Size**: Dimensione (scala) del terreno nel mondo 3D di Blender.
*   **Resolution**: Risoluzione della mesh. Valori più alti aumentano il dettaglio ma richiedono più memoria.


#### DA AGGIORNARE

[⬆ Torna su / Back to top](#terrain-generation-pipeline-configuration)

---

<a name="english-version"></a>
# English Version

This document describes the structure and parameters of the `settings.json` file. This file is the core configuration for the **Blender automation Python pipeline**.

## 🚀 Usage

This pipeline is designed to be flexible:

*   **Python Automation:** Using the dedicated script, you can process the JSON file to generate massive terrain datasets (meshes, renders, and maps).
*   **Blender Testing:** Alternatively, you can use the `terrain_generator_mountains_plains_v1.blend` file for immediate visual feedback, adjusting nodes and validating parameters before starting batch generation.

---

## 🛠️ JSON Parameters Guide

### 1. Dataset Settings
Controls the global behavior of the pipeline and export options.


| Field | Description |
| :--- | :--- |
| `generation_mode` | Generation mode (e.g., `fixed` (single, standard value) or `random` (multiple generation, values between `min` and `max`)). |
| `num_terrains_to_generate` | Total number of variants to create. |
| `output_folder_...` | Target folders for Renders, Meshes, and Maps. |
| `export_render_png` | If `true`, saves the visual render of the scene. |
| `export_mesh_off` | Exports geometry in .off format. |
| `export_heightmap_exr` | Saves the heightmap in 32-bit format. |
| `export_heightmap_asc` | Exports elevation data in ASCII Grid text format. |
| `export_hillshade` | Generates a relief shadow map (Hillshade). |

### 2. Node Parameters
Technical parameters for procedural terrain modeling.

#### Mesh Configuration
*   **Grid_Size**: Dimension (scale) of the terrain in the Blender 3D world.
*   **Resolution**: Mesh resolution. Higher values increase detail but require more memory.


#### TO UPDATE

[⬆ Torna su / Back to top](#terrain-generation-pipeline-configuration)