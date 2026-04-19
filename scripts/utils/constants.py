from utils_method import get_full_path

#Change with your blender.exe path o just write blender if its already in the PATH system
BLENDER_EXECUTABLE = "C:\\Program Files\\Blender Foundation\Blender 4.2\\blender-launcher.exe"

BLEND_FILE = get_full_path("terrain_generator", "test_terrain_generator.blend")
BLENDER_SCRIPT = get_full_path("scripts", "generator", "blender_generator")

CONFIG_FILE = get_full_path("config.json")
TERRAIN_NAME = "terrain-gen"
MODIFIERS_NAME_GM = "GeometryNodes"

NAME_GENERATED_TERRAIN = "terrain_"
EXTENSION_RENDERING = "png"