from utils.utils_method import get_full_path

#Change with your blender.exe path o just write blender if its already in the PATH system
BLENDER_EXECUTABLE = "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe"

BLEND_FILE = get_full_path("terrain_generator", "test_terrain_generator.blend")
BLENDER_SCRIPT = get_full_path("scripts", "generator", "blender_generator.py")

CONFIG_FILE = get_full_path("terrain_generator", "dataset_setting.json")
TERRAIN_NAME = "terrain-gen"
MODIFIERS_NAME_GM = "GeometryNodes"

NAME_GENERATED_TERRAIN = "terrain_"
EXTENSION_RENDERING = "png"

GENERATION_MODE_RANDOM = "random"
GENERATION_MODE_FIXED = "fixed"

NODE_MAP = {
    "Landscape Size": "Socket_2",   # Esempio: cambia Socket_1 con quello vero!
    "Vertices Count": "Socket_4",
    "Base Seed": "Socket_5",
    "Base Scale": "Socket_6",
    "Base Detail": "Socket_7",
    "Rock Seed": "Socket_8",
    "Rock Scale": "Socket_9",
    "Rock Detail": "Socket_10",
    "Base Roughness": "Socket_11",
    "Rock Roughness": "Socket_12",
    "Detail Weight": "Socket_13",
    "Height Scale": "Socket_14",
    "Clip Valley": "Socket_15",
    "Clip Peaks": "Socket_16",
    "Elevation Power": "Socket_17",
    "Second Material": "Socket_18"
}

INBLEND_IMAGE_NAME = "Heightmap_Export"
FILE_FORMAT_IMG = "OPEN_EXR"
NAME_HEIGHTMAPS = "heightmap_"
NAME_HILLSHADE = "hillshade_"
EXTENSION_HEIGHTMAP_EXR = "exr"
EXTENSION_HEIGHTMAP_ASC = "asc"
EXTENSION_HILLSHADE = "PNG"
COLOR_MODE_HILLSHADE = "BW"