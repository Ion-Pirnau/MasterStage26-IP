import os
import bpy, json, random, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from utils.utils_method import get_full_path, obj_to_off
from utils.constants import CONFIG_FILE, TERRAIN_NAME, MODIFIERS_NAME_GM, NAME_GENERATED_TERRAIN, EXTENSION_RENDERING, NODE_MAP

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

if __name__ == "__main__" :
    generate_terrain()



