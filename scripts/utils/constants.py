from utils.utils_method import get_full_path

#Change with your blender.exe path o just write blender if its already in the PATH system
BLENDER_EXECUTABLE = "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe"

# PATH CONSTANTS
BLEND_FILE = get_full_path("terrain_generator", "terrain_generator_mountains_plains_v1.blend")
BLENDER_SCRIPT = get_full_path("scripts", "generator", "blender_generator.py")
CONFIG_FILE = get_full_path("terrain_generator", "dataset_setting.json")

# CONSTANTS VAR FOR IDENTIFYING ELEMENTS IN BLENDER WORLD
TERRAIN_NAME = "Cube"
MODIFIERS_NAME_GM = "GeometryNodes"

# CONSTANTS FOR TYPE OF RENDERING GENERATION
GENERATION_MODE_RANDOM = "random"
GENERATION_MODE_FIXED = "fixed"

# CONSTANT MAP, MAPPING THE SOCKET NAME OF GROUP INPUT TO THE CUSTOM NAME FOR UX
NODE_MAP = {
    "HeightmapIsolinee": "Socket_29",
    "Factor mix Texture": "Socket_30",
    "Grid_Size": "Socket_2",
    "Resolution": "Socket_3",
    "Noise W Factor": "Socket_4",
    "Noise Scale": "Socket_5",
    "Noise Detail": "Socket_6",
    "Noise Roughness": "Socket_7",
    "Noise Lacunarity": "Socket_8",
    "Noise Distortion": "Socket_9",
    "Noise Magnitude": "Socket_10",
    "Magnitude Z Offset": "Socket_11",
    "Crater On": "Socket_17",
    "Noise W Crater": "Socket_12",
    "Scale Crater": "Socket_13",
    "Radius Influence": "Socket_14",
    "From Max": "Socket_15",
    "To Max 1": "Socket_16",
    "Erosion On": "Socket_24",
    "Erosion Scale": "Socket_18",
    "From Min Threshold": "Socket_21",
    "From Max Threshold": "Socket_19",
    "To Min": "Socket_20",
    "To Max 2": "Socket_22",
    "Erosion Depth": "Socket_23",
    "Sediment On": "Socket_25",
    "Sediment Height": "Socket_26"
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
NAME_HEIGHTMAPS = "heightmap_"
NAME_HILLSHADE = "hillshade_"
EXTENSION_HEIGHTMAP_EXR = "exr"
EXTENSION_HEIGHTMAP_ASC = "asc"
EXTENSION_HILLSHADE = "PNG"
COLOR_MODE_HILLSHADE = "BW"