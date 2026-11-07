# Terrain Generator

Scegli la tua lingua / Choose your language:
- [🇮🇹 Italiano](#versione-italiana)
- [🇬🇧 English](#english-version)

---

<a name="versione-italiana"></a>
# 🇮🇹 Versione Italiana

Questa cartella contiene il generatore procedurale di terreni in Blender e il file di configurazione JSON che ne controlla il comportamento.

## 📂 Contenuto della cartella

| File | Descrizione |
| :--- | :--- |
| `terreno_procedurale_tesi_features.blend` | Il generatore: contiene la rete di nodi (Geometry Nodes) che crea i terreni proceduralmente. Può essere usato a mano per test visivi, oppure pilotato dalla pipeline Python. |
| `view_heightmap.blend` | Visualizzatore: carica le heightmap già generate (che sono immagini 2D) e le mostra come rilievo nell'ambiente 3D. |
| `dataset_setting.json` | Configurazione della generazione (modalità, numero di terreni, export, parametri dei nodi). |
| `README.md` | Questa guida. |

## 🚀 Modalità d'uso

1. **Test manuale in Blender.** Apri `terreno_procedurale_tesi_features.blend`, regola i nodi e verifica l'aspetto del terreno prima di lanciare la generazione massiva.
2. **Generazione automatica.** Imposta `dataset_setting.json` e avvia la pipeline dalla cartella `scripts/` (vedi `scripts/README.md`). La pipeline genera N terreni e li esporta nelle cartelle indicate.
3. **Visualizzazione dei risultati.** Apri `view_heightmap.blend` per guardare in 3D le heightmap generate.

## 🛠️ Guida ai parametri JSON

Il file ha due blocchi: `dataset_settings` (comportamento globale ed export) e `node_parameters` (parametri dei nodi Blender).

### 1. `dataset_settings`

| Campo | Tipo | Descrizione |
| :--- | :--- | :--- |
| `generation_mode` | stringa | `fixed`: usa il campo `value` di ogni parametro (un solo terreno ripetibile). `random`: per ogni terreno estrae un valore casuale tra `min` e `max`. |
| `camera_ortho_active` | bool | Se `true`, usa una camera ortografica per i render (vista dall'alto senza prospettiva). [DA VERIFICARE] |
| `num_terrains_to_generate` | int | Numero di terreni da creare (es. `150`). |
| `output_folder_renders` | percorso | Cartella dei render PNG. |
| `output_folder_meshes` | percorso | Cartella delle mesh esportate. |
| `output_folder_maps` | percorso | Cartella base delle mappe (heightmap). |
| `output_folder_heightmap_exr` | percorso | Sottocartella delle heightmap EXR. |
| `output_folder_splatmap_png` | percorso | Sottocartella delle splatmap PNG. |
| `export_render_png` | bool | Salva il render visivo della scena. |
| `export_mesh_off` | bool | Esporta la geometria in formato `.off`. |
| `export_heightmap_exr` | bool | Salva la heightmap in EXR a 32 bit. |
| `export_heightmap_exr_with_splat` | bool | Salva la heightmap EXR insieme alla relativa splatmap. |
| `isSplatActive` | bool | Attiva la generazione della splatmap (mappa dei materiali/tipi di terreno). |
| `export_heightmap_asc` | bool | Esporta le quote in formato testo ASCII Grid (`.asc`). |
| `export_hillshade` | bool | Genera la mappa d'ombra del rilievo (hillshade). |

> ⚠️ **Nota Bene:** nella configurazione attuale `export_heightmap_exr_with_splat` è `true` ma `isSplatActive` è `false`. Il File splat non viene creato.

### 2. `node_parameters`

Ogni parametro ha un `type` (`float`, `int`, `boolean`) e questi campi:

- **Numerici** (`float`, `int`): `value` (usato in modalità `fixed`), `min` e `max` (intervallo in modalità `random`). Se `min == max` il parametro è di fatto costante.
- **Booleani**: `value` (usato in `fixed`) e `probability` (probabilità, tra 0 e 1, che la funzione sia attiva in modalità `random`).

#### Mesh

| Parametro | Descrizione |
| :--- | :--- |
| `Dimensione Griglia` | Dimensione (scala) del terreno nel mondo 3D di Blender. |
| `Risoluzione XY` | Risoluzione della griglia lungo X e Y (es. `512` = 512×512 punti). Valori alti aumentano dettaglio e memoria. |
| `Risoluzione` | Livello di suddivisione aggiuntivo, che funziona come un modificatore Subdivision Surface: aggiunge dettaglio sopra la griglia base definita da Risoluzione XY. Con 0 non viene applicata nessuna suddivisione (è il valore attuale nel file). Ogni livello in più aumenta molto il numero di vertici e quindi memoria e tempi di generazione. |

#### Rumore di base

| Parametro | Descrizione |
| :--- | :--- |
| `Scale_Noise_One` | Scala del primo livello di rumore. |
| `Scale_Noise_Two` | Scala del secondo livello di rumore. |
| `Attiva Tipo Materiale` | Attiva la differenziazione per tipo di materiale. |

#### Erosione

| Parametro | Descrizione |
| :--- | :--- |
| `Attiva Erosione` | Attiva l'effetto di erosione. |
| `Noise Scale Erosione` | Scala del rumore che guida l'erosione. |
| `Profondita Erosione` | Profondità dei solchi di erosione. |

#### Crateri

| Parametro | Descrizione |
| :--- | :--- |
| `Attiva Crateri` | Attiva la generazione di crateri. |
| `Seed Crateri` | Seed casuale del posizionamento dei crateri. |
| `Distance Min` | Distanza minima tra crateri. |
| `Density Max` | Densità massima dei crateri. |

#### Vulcani

| Parametro | Descrizione |
| :--- | :--- |
| `Vulcani_Altezza` | Altezza dei coni vulcanici. |
| `Vulcani_Raggio_Interno` | Raggio interno (cratere/caldera). |
| `Vulcani_Raggio_Esterno` | Raggio esterno (base del cono). |

#### Montagne

| Parametro | Descrizione |
| :--- | :--- |
| `Attiva Montagna` | Attiva la generazione di montagne. |
| `Seed Montagna` | Seed casuale delle montagne. |
| `Scale` | Scala del rumore delle montagne. |
| `Detail` | Livello di dettaglio del rumore. |
| `Campo di Forza` | Intensità del campo di forza che modella il rilievo. |
| `Altezza Max Montagna` | Altezza massima delle montagne. |
| `Ripidita Montagna` | Ripidità dei versanti. |
| `Larghezza Montagna` | Larghezza della montagna. |

#### Conoide / Fan

| Parametro | Descrizione |
| :--- | :--- |
| `Attiva Fan` | Attiva la formazione a conoide (fan). |
| `Length` | Lunghezza del conoide. |
| `Angle` | Apertura angolare, in gradi. |
| `Height` | Altezza del conoide. |
| `Edge Noise Scale` | Scala del rumore sul bordo. |
| `Edge Roughness` | Rugosità del bordo. |

[⬆ Torna su / Back to top](#terrain-generator)

---

<a name="english-version"></a>
# 🇬🇧 English Version

This folder contains the procedural terrain generator for Blender and the JSON configuration file that controls it.

## 📂 Folder contents

| File | Description |
| :--- | :--- |
| `terreno_procedurale_tesi_features.blend` | The generator: contains the node network (Geometry Nodes) that builds terrains procedurally. It can be used by hand for visual tests, or driven by the Python pipeline. |
| `view_heightmap.blend` | Viewer: loads already generated heightmaps (which are 2D images) and displays them as relief in the 3D environment. |
| `dataset_setting.json` | Generation configuration (mode, number of terrains, exports, node parameters). |
| `README.md` | This guide. |

## 🚀 Usage

1. **Manual testing in Blender.** Open `terreno_procedurale_tesi_features.blend`, adjust the nodes and check the terrain before launching batch generation.
2. **Automated generation.** Set up `dataset_setting.json` and start the pipeline from the `scripts/` folder (see `scripts/README.md`). The pipeline generates N terrains and exports them to the configured folders.
3. **Viewing the results.** Open `view_heightmap.blend` to see the generated heightmaps in 3D. [TO COMPLETE: how to select/load an image in the file, e.g. which node or texture to change]

## 🛠️ JSON Parameters Guide

The file has two blocks: `dataset_settings` (global behavior and exports) and `node_parameters` (Blender node parameters).

### 1. `dataset_settings`

| Field | Type | Description |
| :--- | :--- | :--- |
| `generation_mode` | string | `fixed`: uses each parameter's `value` (a single repeatable terrain). `random`: draws a random value between `min` and `max` for every terrain. |
| `camera_ortho_active` | bool | If `true`, renders with an orthographic camera (top-down view, no perspective). [TO VERIFY] |
| `num_terrains_to_generate` | int | Number of terrains to create (e.g. `150`). |
| `output_folder_renders` | path | Folder for PNG renders. |
| `output_folder_meshes` | path | Folder for exported meshes. |
| `output_folder_maps` | path | Base folder for maps (heightmaps). |
| `output_folder_heightmap_exr` | path | Subfolder for EXR heightmaps. |
| `output_folder_splatmap_png` | path | Subfolder for PNG splat maps. |
| `export_render_png` | bool | Saves the visual render of the scene. |
| `export_mesh_off` | bool | Exports geometry in `.off` format. |
| `export_heightmap_exr` | bool | Saves the heightmap as 32-bit EXR. |
| `export_heightmap_exr_with_splat` | bool | Saves the EXR heightmap together with its splat map. |
| `isSplatActive` | bool | Enables splat map generation (map of material/terrain types). |
| `export_heightmap_asc` | bool | Exports elevation data as ASCII Grid text (`.asc`). |
| `export_hillshade` | bool | Generates the relief shadow map (hillshade). |

> ⚠️ **Note:** in the current configuration `export_heightmap_exr_with_splat` is `true` but `isSplatActive` is `false`. Right now the splat file is not generated

### 2. `node_parameters`

Each parameter has a `type` (`float`, `int`, `boolean`) and these fields:

- **Numeric** (`float`, `int`): `value` (used in `fixed` mode), `min` and `max` (range in `random` mode). If `min == max` the parameter is effectively constant.
- **Boolean**: `value` (used in `fixed`) and `probability` (a number between 0 and 1: the chance the feature is active in `random` mode).

#### Mesh

| Parameter | Description |
| :--- | :--- |
| `Dimensione Griglia` | Size (scale) of the terrain in the Blender 3D world. |
| `Risoluzione XY` | Grid resolution along X and Y (e.g. `512` = 512×512 points). Higher values increase detail and memory use. |
| `Risoluzione` | Additional subdivision level, working like a Subdivision Surface modifier: it adds detail on top of the base grid defined by Risoluzione XY. At 0 no subdivision is applied (the current value in the file). Each extra level greatly increases the vertex count, and therefore memory use and generation time. |

#### Base noise

| Parameter | Description |
| :--- | :--- |
| `Scale_Noise_One` | Scale of the first noise layer. |
| `Scale_Noise_Two` | Scale of the second noise layer. |
| `Attiva Tipo Materiale` | Enables differentiation by material type. |

#### Erosion

| Parameter | Description |
| :--- | :--- |
| `Attiva Erosione` | Enables the erosion effect. |
| `Noise Scale Erosione` | Scale of the noise driving the erosion. |
| `Profondita Erosione` | Depth of the erosion grooves. |

#### Craters

| Parameter | Description |
| :--- | :--- |
| `Attiva Crateri` | Enables crater generation. |
| `Seed Crateri` | Random seed for crater placement. |
| `Distance Min` | Minimum distance between craters. |
| `Density Max` | Maximum crater density. |

#### Volcanoes

| Parameter | Description |
| :--- | :--- |
| `Vulcani_Altezza` | Height of the volcanic cones. |
| `Vulcani_Raggio_Interno` | Inner radius (crater/caldera). |
| `Vulcani_Raggio_Esterno` | Outer radius (cone base). |

#### Mountains

| Parameter | Description |
| :--- | :--- |
| `Attiva Montagna` | Enables mountain generation. |
| `Seed Montagna` | Random seed for the mountains. |
| `Scale` | Scale of the mountain noise. |
| `Detail` | Noise detail level. |
| `Campo di Forza` | Strength of the force field shaping the relief. |
| `Altezza Max Montagna` | Maximum mountain height. |
| `Ripidita Montagna` | Slope steepness. |
| `Larghezza Montagna` | Mountain width. |

#### Fan

| Parameter | Description |
| :--- | :--- |
| `Attiva Fan` | Enables the fan (alluvial-fan-like) feature. |
| `Length` | Fan length. |
| `Angle` | Angular spread, in degrees. |
| `Height` | Fan height. |
| `Edge Noise Scale` | Scale of the edge noise. |
| `Edge Roughness` | Edge roughness. |

[⬆ Torna su / Back to top](#terrain-generator)
