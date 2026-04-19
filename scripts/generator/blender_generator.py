import os
import numpy as np
import bpy, json, random, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from utils.utils_method import get_full_path, obj_to_off
from utils.constants import *
def load_config():
    print(CONFIG_FILE)
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, IOError) as e:
        print(f"Error during opening: {e}")
        return None

def set_modifier_value(modifier, identifier, value):
    blender_key = NODE_MAP.get(identifier)
    print(blender_key)
    if blender_key in modifier.keys():
        modifier[blender_key] = value
    else:
        print(f"Attention: Input '{identifier}' not found in Geometry Nodes!")


def export_heightmap(obj, filepath, res_x, res_y, landscape_size=10):
    if not obj:
        print("Object not found!")
        return
    
    if "Heightmap_Export" in bpy.data.images:
        bpy.data.images.remove(bpy.data.images["Heightmap_Export"])

    image = bpy.data.images.new("Heightmap_Export", width=res_x, height=res_y, float_buffer=True)
    
    #mesh = obj.to_mesh()
    #bpy.context.view_layer.update()

    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    size_x = obj.dimensions.x
    size_y = obj.dimensions.y

    step_x = size_x / res_x
    step_y = size_y / res_y
    # Partiamo dall'angolo in basso a sinistra (relativo all'origine dell'oggetto)
    start_x = obj.location.x - (size_x / 2)
    start_y = obj.location.y - (size_y / 2)

    heights = []
    direction = (0, 0, -1)
    ray_start_z = obj.location.z + obj.dimensions.z + 10 # Sopra l'oggetto

    for y in range(res_y):
        for x in range(res_x):
            # Centriamo il raggio nel pixel
            target_x = start_x + (x * step_x) + (step_x / 2)
            target_y = start_y + (y * step_y) + (step_y / 2)
            
            origin = (target_x, target_y, ray_start_z)
            hit, loc, norm, idx, h_obj, mat = bpy.context.scene.ray_cast(depsgraph, origin, direction)
            
            # Se colpisce, usa l'altezza Z globale meno la base dell'oggetto
            heights.append(loc.z if hit else 0.0)

    # 4. Normalizzazione Pro
    h_array = np.array(heights)
    h_min, h_max = h_array.min(), h_array.max()
    denom = (h_max - h_min) if h_max != h_min else 1.0
    norm_h = (h_array - h_min) / denom

    # 5. Creazione pixel (RGBA)
    pixels = np.ones((res_x * res_y, 4), dtype=np.float32)
    pixels[:, 0:3] = norm_h.reshape(-1, 1) # Imposta R, G, B contemporaneamente


    image.pixels = pixels.flatten()
    image.filepath_raw = filepath
    image.file_format = FILE_FORMAT_IMG
    image.save()

    print(f"Heightmap saved in: {filepath}")


def generate_terrain():
    config = load_config()
    settings = config["dataset_settings"]
    params = config["node_parameters"]

    terrain_obj = bpy.data.objects.get(TERRAIN_NAME)
    mod = terrain_obj.modifiers.get(MODIFIERS_NAME_GM)

    print("\n--- DEBUG IDENTIFIERS GEOMETRY NODES ---")
    for key, value in mod.items():
        print(f"Intern Key is: '{key}' (Actual value: {value})")
    print("-------------------------------------------\n")

    mode = settings.get("generate_mode", "fixed")
    num_to_generate = settings["num_terrains_to_generate"] if mode == "random" else 1

    print(f"Mode: {mode}. Generation of {num_to_generate} terrains...")

    for i in range(num_to_generate):
        print(f"\n--- Generation {i+1}/{num_to_generate} ---")
        for param_name, param_data in params.items():
            if mode == "fixed":
                val = param_data["value"]
            else:
                if param_data["type"] == "boolean":
                    val = random.random() < param_data["probability"]
                elif param_data["type"] == "int":
                    val = random.randint(int(param_data["min"]), int(param_data["max"]))
                else:
                    val = random.uniform(param_data["min"], param_data["max"])
            
            set_modifier_value(mod, param_name, val)
            print(f"Set {param_name} = {val}")

        bpy.context.view_layer.update()

        if settings["export_render_png"]:
            print(get_full_path(settings["output_folder_renders"]))
            render_path = get_full_path(settings["output_folder_renders"], f"{NAME_GENERATED_TERRAIN}{i:04d}.{EXTENSION_RENDERING}")
            bpy.context.scene.render.filepath = render_path
            bpy.ops.render.render(write_still=True)
            print(f"Render saved in {render_path}")
        
        if settings["export_mesh_off"]:
            obj_path = get_full_path(settings["output_folder_meshes"], f"terrain_{i:04d}.obj")
            off_path = get_full_path(settings["output_folder_meshes"], f"terrain_{i:04d}.off")

            bpy.ops.object.select_all(action="DESELECT")
            terrain_obj.select_set(True)

            bpy.ops.wm.obj_export(filepath=obj_path, export_selected_objects=True)

            obj_to_off(obj_path, off_path)

            print(f"Mesh OFF salvata in {off_path}")

        if settings["export_heightmap"]:
            export_heightmap(terrain_obj, 
                             get_full_path(settings["output_folder_maps"], f"{NAME_HEIGHTMAPS}{i:04d}.{EXTENSION_HEIGHTMAP}"),
                             res_x=256,
                             res_y=256)



if __name__ == "__main__" :
    generate_terrain()



