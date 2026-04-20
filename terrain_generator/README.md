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
*   **Test in Blender:** In alternativa, è possibile utilizzare il file `test_terrain_generator.blend` per effettuare test visivi immediati, regolare i nodi e validare i parametri prima di lanciare la generazione massiva.

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
| `export_hillshade` | Genera una mappa d'ombra del rilievo (Hillshade) (DA OTTIMIZZARE). |

### 2. Node Parameters
Parametri tecnici per la modellazione procedurale del terreno.

#### Configurazione Mesh
*   **Landscape Size**: Dimensione (scala) del terreno nel mondo 3D di Blender.
*   **Vertices Count**: Risoluzione della mesh. Valori più alti aumentano il dettaglio ma richiedono più memoria.

#### Generazione Procedurale (Base & Rock)
Il terreno è formato dalla sovrapposizione di due tipi di rumore (Noise): **Base** (forma generale) e **Rock** (dettagli rocciosi).
*   **Seed**: Valore per la generazione casuale. Cambiando il seed, la forma cambia radicalmente.
*   **Scale**: Frequenza del rumore. Valori bassi creano forme ampie, valori alti creano dettagli fitti.
*   **Detail**: Complessità delle frastagliature superficiali.
*   **Roughness**: Grado di ruvidità della superficie.

#### Modellazione Avanzata
*   **Detail Weight**: Influenza dello strato "Rock" sulla forma finale.
*   **Height Scale**: Moltiplicatore dell'altezza totale (asse Z).
*   **Clip Valley / Peaks**: Taglia i valori di altezza minimi o massimi (crea pianure o altopiani).
*   **Elevation Power**: Controlla la pendenza. Valori alti rendono le vette più aguzze e le valli più profonde.

#### Materiali
*   **Second Material**: Attiva un materiale secondario basato sulla pendenza (il primario è da migliorare).
*   **Probability**: In modalità random, definisce la probabilità che il secondo materiale venga applicato.

[⬆ Torna su / Back to top](#terrain-generation-pipeline-configuration)

---

<a name="english-version"></a>
# English Version

This document describes the structure and parameters of the `settings.json` file. This file is the core configuration for the **Blender automation Python pipeline**.

## 🚀 Usage

This pipeline is designed to be flexible:

*   **Python Automation:** Using the dedicated script, you can process the JSON file to generate massive terrain datasets (meshes, renders, and maps).
*   **Blender Testing:** Alternatively, you can use the `test_terrain_generator.blend` file for immediate visual feedback, adjusting nodes and validating parameters before starting batch generation.

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
| `export_hillshade` | Generates a relief shadow map (Hillshade) (UNDER OPTIMIZATION). |

### 2. Node Parameters
Technical parameters for procedural terrain modeling.

#### Mesh Configuration
*   **Landscape Size**: Dimension (scale) of the terrain in the Blender 3D world.
*   **Vertices Count**: Mesh resolution. Higher values increase detail but require more memory.

#### Procedural Generation (Base & Rock)
The terrain is formed by layering two types of noise: **Base** (general shape) and **Rock** (rocky details).
*   **Seed**: Random generation value. Changing the seed radically changes the shape.
*   **Scale**: Noise frequency. Lower values create large shapes, higher values create dense details.
*   **Detail**: Complexity of surface jaggedness.
*   **Roughness**: Degree of surface roughness.

#### Advanced Modeling
*   **Detail Weight**: Influence of the "Rock" layer on the final shape.
*   **Height Scale**: Multiplier for the total height (Z-axis).
*   **Clip Valley / Peaks**: Clips the minimum or maximum height values (creates plains or plateaus).
*   **Elevation Power**: Controls the slope. Higher values make peaks sharper and valleys deeper.

#### Materials
*   **Second Material**: Enables a secondary material based on slope (primary material is under improvement).
*   **Probability**: In random mode, defines the probability that the second material will be applied.

[⬆ Torna su / Back to top](#terrain-generation-pipeline-configuration)