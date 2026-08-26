from utils.utils_method import get_full_path

#Change with your blender.exe path o just write blender if its already in the PATH system
BLENDER_EXECUTABLE = "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe"

# PATH CONSTANTS
BLEND_FILE = get_full_path("terrain_generator", "terreno_procedurale_tesi_features.blend")
BLENDER_SCRIPT = get_full_path("scripts", "generator", "blender_generator.py")
CONFIG_FILE = get_full_path("terrain_generator", "dataset_setting.json")
HEIGHTMAPS_EXR_LOCATION = get_full_path("data", "heightmaps", "heightmap_exr")
HEIGHTMAPS_PNG_LOCATION = get_full_path("data", "heightmaps", "heightmap_png")

# CONSTANTS VAR FOR IDENTIFYING ELEMENTS IN BLENDER WORLD
TERRAIN_NAME = "Cube"
MODIFIERS_NAME_GM = "GeometryNodes"

# CONSTANTS FOR TYPE OF RENDERING GENERATION
GENERATION_MODE_RANDOM = "random"
GENERATION_MODE_FIXED = "fixed"

# CONSTANT MAP, MAPPING THE SOCKET NAME OF GROUP INPUT TO THE CUSTOM NAME FOR UX
NODE_MAP = {
    "Dimensione Griglia": "Socket_2",
    "Risoluzione XY": "Socket_3",
    "Risoluzione": "Socket_16",
    "Scale_Noise_One": "Socket_23",
    "Scale_Noise_Two": "Socket_24",
    "Attiva Tipo Materiale": "Socket_34",
    "Attiva Erosione": "Socket_30",
    "Noise Scale Erosione": "Socket_31",
    "Profondita Erosione": "Socket_32",
    "Attiva Crateri": "Socket_14",
    "Seed Crateri": "Socket_22",
    "Distance Min": "Socket_5",
    "Density Max": "Socket_6",
    "Vulcani_Altezza": "Socket_8",
    "Vulcani_Raggio_Interno": "Socket_7",
    "Vulcani_Raggio_Esterno": "Socket_9",
    "Attiva Montagna": "Socket_15",
    "Seed Montagna": "Socket_11",
    "Scale": "Socket_12",
    "Detail": "Socket_13",
    "Campo di Forza": "Socket_17",
    "Altezza Max Montagna": "Socket_18",
    "Ripidita Montagna": "Socket_19",
    "Larghezza Montagna": "Socket_20",
    "Attiva Fan": "Socket_21",
    "Length": "Socket_25",
    "Angle": "Socket_26",
    "Height": "Socket_27",
    "Edge Noise Scale": "Socket_28",
    "Edge Roughness": "Socket_29",
}

#CAM_CLIP_END = 50000.0
#SUN_Z_LOCATION = 10000

# CONSTANTS FOR CAMERAS IN THE SCENE
CAMERA_ORTHO_ON_TOP = "cameraOnTop"
CAMERA_ON_SIDE = "cameraOnSide"

# OTHER CONSTANTS
LANDSCAPE_NODE_MODIFIER = "Socket_2"
TIFF_FOLDER = "tiff_to_convert"
MESH_FOLDER = "meshes"
OUTPUT_TIFF_OFF_CONVERT_NAME = "convert_tiff_off_"
TIFF_EXTENSION = "tif"
OFF_EXTENSION = "off"
OBJ_EXTENSION = "obj"


# FOR RENDERING CONSTANTS
NAME_GENERATED_TERRAIN = "newterrain_"
EXTENSION_RENDERING = "png"
INBLEND_IMAGE_NAME = "Heightmap_Export"
FILE_FORMAT_IMG = "OPEN_EXR"
NAME_HEIGHTMAPS = "heightmap0_"
NAME_HILLSHADE = "hillshade_"
EXTENSION_HEIGHTMAP_EXR = "exr"
EXTENSION_HEIGHTMAP_ASC = "asc"
EXTENSION_HILLSHADE = "PNG"
COLOR_MODE_HILLSHADE = "BW"
RESOLUTION_XY = 2048


# FOR STABLE DIFFUSION MODEL XL
PATH_FOLDER_SDXL = "data/render/sdxl_img"