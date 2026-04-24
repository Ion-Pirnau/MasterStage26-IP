import os
import numpy as np
import bpy, json, random, sys
import mathutils, math
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 

from utils.utils_method import get_full_path, obj_to_off
from utils.constants import *

def load_config():
    """
    Loads the configuration file in JSON format.

    This function reads the global CONFIG_FILE and parses its content
    into a Python dictionary. In case of failure, it handles file-related
    exceptions and returns None.

    Parameters
    ----------
    None

    Returns
    -------
    dict or None
        Parsed configuration dictionary if successful, otherwise None.

    Raises
    ------
    None
        Exceptions are caught internally and reported via print.

    Notes
    -----
    - The configuration file is expected to be in valid JSON format.
    - Errors during file access or parsing are handled gracefully.
    """

    print(CONFIG_FILE)
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, IOError) as e:
        print(f"Error during opening: {e}")
        return None

def set_modifier_value(modifier, identifier, value):
    """
    Sets a value on a Geometry Nodes modifier input.

    This function maps a parameter name (identifier)
    to the corresponding Blender internal key using NODE_MAP, and
    assigns the given value to the modifier.

    Parameters
    ----------
    modifier : bpy.types.Modifier
        The Geometry Nodes modifier to be updated.

    identifier : str
        Logical name of the parameter (defined in configuration).

    value : Any
        Value to assign to the modifier input.

    Returns
    -------
    None

    Raises
    ------
    None
        Missing keys are handled with a warning message.

    Notes
    -----
    - The mapping between identifier and Blender input is defined in NODE_MAP.
    - If the input is not found, a warning is printed.
    """


    blender_key = NODE_MAP.get(identifier)
    print(blender_key)
    if blender_key in modifier.keys():
        modifier[blender_key] = value
    else:
        print(f"Attention: Input '{identifier}' not found in Geometry Nodes!")


def export_heightmap_exr(obj, filepath, res_x, res_y) -> bool:
    """
    Generates and exports a heightmap image from a 3D terrain mesh.

    This function performs ray casting over a grid defined by the given
    resolution to sample height values from the evaluated mesh. The resulting
    height values are normalized and saved as a grayscale image.

    Parameters
    ----------
    obj : bpy.types.Object
        The terrain object from which to extract height data.

    filepath : str
        Output path where the heightmap image will be saved.

    res_x : int
        Horizontal resolution of the heightmap.

    res_y : int
        Vertical resolution of the heightmap.

    Returns
    -------
    bool
        True if the heightmap is successfully generated and saved,
        False otherwise.

    Raises
    ------
    None
        Errors are handled internally with print messages.

    Notes
    -----
    - Uses ray casting from above the terrain to sample heights.
    - The evaluated mesh (with modifiers applied) is used.
    - Output is a normalized grayscale image (R=G=B).
    - Requires NumPy and Blender's mathutils module.
    """

    if not obj:
        print("Object not found!")
        return False
    
    if INBLEND_IMAGE_NAME in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[INBLEND_IMAGE_NAME])

    image = bpy.data.images.new(INBLEND_IMAGE_NAME, width=res_x, height=res_y, 
                                float_buffer=True, is_data=True)

    # Obtain the evalutated mesh (the real generated landscape)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)

    # Calculate the bounding-box
    bbox = [eval_obj.matrix_world @ mathutils.Vector(b) for b in eval_obj.bound_box]
    
    min_x = min(v.x for v in bbox)
    max_x = max(v.x for v in bbox)
    min_y = min(v.y for v in bbox)
    max_y = max(v.y for v in bbox)
    max_z = max(v.z for v in bbox)
    min_z = min(v.z for v in bbox)
    
    size_x = max_x - min_x
    size_y = max_y - min_y

    step_x = size_x / res_x
    step_y = size_y / res_y

    heights = np.zeros(res_x * res_y, dtype=np.float32)
    direction = mathutils.Vector((0, 0, -1))
    # Start the ray from +10m above the max_z
    ray_start_z = max_z + 10.0 

    # RAYCASTING
    idx = 0
    for y in range(res_y):
        for x in range(res_x):
            target_x = min_x + (x * step_x) + (step_x / 2)
            target_y = min_y + (y * step_y) + (step_y / 2)
            
            origin = mathutils.Vector((target_x, target_y, ray_start_z))
            
            # Use the eval_obj for the raycast
            hit, loc, norm, face_idx, hit_obj, mat = bpy.context.scene.ray_cast(depsgraph, origin, direction)
            
            heights[idx] = loc.z if hit else min_z
            idx += 1

    # NUMPY normalization
    h_min, h_max = heights.min(), heights.max()
    denom = (h_max - h_min) if h_max != h_min else 1.0
    norm_h = (heights - h_min) / denom

    # Creation Pixel
    pixels = np.ones((res_x * res_y, 4), dtype=np.float32)
    pixels[:, 0] = norm_h # R
    pixels[:, 1] = norm_h # G
    pixels[:, 2] = norm_h # B

    image.pixels.foreach_set(pixels.ravel())

    image.filepath_raw = filepath
    image.file_format = FILE_FORMAT_IMG
    image.save()

    print(f"Heightmap saved in: {filepath}")
    return True


def export_heightmap_asc(obj, filepath, res_x, res_y) -> bool:
    """
    Exports a terrain heightmap in ESRI ASCII Grid (.asc) format.

    This function samples elevation values from a 3D terrain mesh using
    ray casting over a regular grid. The sampled heights are written to
    an ASCII raster file, including a GIS-compatible header describing
    spatial metadata.

    Parameters
    ----------
    obj : bpy.types.Object
        The terrain object from which elevation data is extracted.

    filepath : str
        Output path for the generated .asc file.

    res_x : int
        Number of columns (horizontal resolution) of the grid.

    res_y : int
        Number of rows (vertical resolution) of the grid.

    Returns
    -------
    bool
        True if the export is successful, False otherwise.

    Raises
    ------
    None
        File I/O errors are handled internally.

    Notes
    -----
    - The function uses ray casting from above the terrain to sample heights.
    - The evaluated mesh (with modifiers applied) is used.
    - The output follows the ESRI ASCII Grid specification:
        * ncols, nrows
        * xllcorner, yllcorner
        * cellsize
        * NODATA_value
    - Grid data is written from top row to bottom row, as required by the format.
    - Missing values are filled with -9999.
    """

    if not obj:
        print("Error: Object not found!")
        return False

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)

    bbox = [eval_obj.matrix_world @ mathutils.Vector(b) for b in eval_obj.bound_box]
    
    min_x = min(v.x for v in bbox)
    max_x = max(v.x for v in bbox)
    min_y = min(v.y for v in bbox)
    max_y = max(v.y for v in bbox)
    max_z = max(v.z for v in bbox)
    
    size_x = max_x - min_x
    size_y = max_y - min_y

    step_x = size_x / res_x
    step_y = size_y / res_y

    # PREPARE THE GIS HEADER
    # xllcorner and yllcorner are the bottom-left corner
    header = (
        f"ncols         {res_x}\n"
        f"nrows         {res_y}\n"
        f"xllcorner     {min_x:.6f}\n"
        f"yllcorner     {min_y:.6f}\n"
        f"cellsize      {step_x:.6f}\n"
        f"NODATA_value  -9999\n"
    )

    grid_data = []
    direction = mathutils.Vector((0, 0, -1))
    ray_start_z = max_z + 10.0 

    # ATTENTION: The ASC format writes the data from the top ROW to the bottom.
    for y in range(res_y - 1, -1, -1):
        row_heights = []
        for x in range(res_x):
            target_x = min_x + (x * step_x) + (step_x / 2)
            target_y = min_y + (y * step_y) + (step_y / 2)
            
            origin = mathutils.Vector((target_x, target_y, ray_start_z))
            
            hit, loc, norm, face_idx, hit_obj, mat = bpy.context.scene.ray_cast(depsgraph, origin, direction)
            
            if hit:
                row_heights.append(f"{loc.z:.6f}")
            else:
                row_heights.append("-9999")
                
        grid_data.append(" ".join(row_heights))

    # Write on File
    try:
        with open(filepath, 'w') as f:
            f.write(header)
            for row_string in grid_data:
                f.write(row_string + "\n")
        print("Esportazione ASC completata con successo!")
        return True
        
    except IOError as e:
        print(f"Errore durante la scrittura del file ASC: {e}")
        return False


def export_hillshade(obj, filepath, res_x, res_y):
    """
    Generates and renders a hillshade visualization of a terrain.

    This function configures the Blender scene to simulate a GIS-style
    hillshade rendering by adjusting lighting, materials, camera, and
    ambient occlusion settings. The terrain is rendered using an orthographic
    projection to produce a top-down shaded relief image.

    Parameters
    ----------
    obj : bpy.types.Object
        The terrain object to be rendered.

    filepath : str
        Output path for the rendered hillshade image.

    res_x : int
        Horizontal resolution of the output image.

    res_y : int
        Vertical resolution of the output image.

    Returns
    -------
    bool
        True if the rendering completes successfully, False otherwise.

    Raises
    ------
    None
        Errors are handled internally and reported via print statements.

    Notes
    -----
    - Uses an orthographic camera positioned above the terrain.
    - Applies a neutral material override to enhance shading visibility.
    - Configures a directional light (SUN) to simulate illumination angle.
    - Adjusts Ambient Occlusion parameters to improve terrain detail perception.
    - The rendering is performed using Blender's rendering engine.
    - The camera is automatically scaled to fully cover the terrain bounding box.
    """

    if not obj:
        print("Errore: Oggetto non trovato!")
        return False

    print("Start configuration Hillshade scene...")

    bpy.context.scene.render.resolution_x = res_x
    bpy.context.scene.render.resolution_y = res_y
    bpy.context.scene.render.image_settings.file_format = EXTENSION_HILLSHADE
    bpy.context.scene.render.image_settings.color_mode = COLOR_MODE_HILLSHADE
    bpy.context.scene.render.filepath = filepath

    mat_name = "GIS_Hillshade"
    if mat_name not in bpy.data.materials:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1.0)
        bsdf.inputs['Roughness'].default_value = 1.0
        bsdf.inputs['Specular IOR Level'].default_value = 0.0
    
    bpy.context.view_layer.material_override = bpy.data.materials[mat_name]

    obj.cycles.shadow_terminator_shading_offset = 0.1

    sun_obj = next((light for light in bpy.context.scene.objects if light.type == 'LIGHT' and light.data.type == 'SUN'), None)
    if sun_obj:
        sun_obj.rotation_euler = (math.radians(45), 0, math.radians(135))
        sun_obj.data.energy = 3.0

        sun_obj.location = (0, 0, SUN_Z_LOCATION)
    else:
        print("ATTENTION: No light in the scene.")
    
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes

    ao_node = nodes.get("Ambient Occlusion")

    if ao_node:
        ao_node.inputs['Distance'].default_value = 5.0
        
        ao_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        
        print("Updated Parameters")
    else:
        print("Ambient Occlusion node not found")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    
    bbox = [eval_obj.matrix_world @ mathutils.Vector(b) for b in eval_obj.bound_box]
    min_x = min(v.x for v in bbox)
    max_x = max(v.x for v in bbox)
    min_y = min(v.y for v in bbox)
    max_y = max(v.y for v in bbox)

    print(max_x)
    print(min_x)

    print(max_y)
    print(min_y)
    
    size_x = (max_x - min_x) * 2
    size_y = (max_y - min_y) * 2

    mod = obj.modifiers.get("GeometryNodes")
    landscape_size = mod["Socket_2"] if mod and "Socket_2" in mod else 10.0

    cam = bpy.context.scene.camera
    if cam:
        cam.data.type = 'ORTHO'
        cam.data.clip_end = CAM_CLIP_END
        #cam.data.ortho_scale = max(size_x, size_y)
        cam.data.ortho_scale = landscape_size
       
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        cam.location = (center_x, center_y, 100)
        cam.rotation_euler = (0, 0, 0)
        #cam.location = (0, 0, 1000) 
        #cam.rotation_euler = (0, 0, 0)
    else:
        print("ATTENTION: No active camera in the scene.")


    bpy.ops.render.render(write_still=True)
    print(f"Render Hillshade saved in {filepath}")
    bpy.context.view_layer.material_override = None
    
    return True

def generate_terrain():
    """
    Generates terrain samples using Geometry Nodes and exports dataset outputs.

    This function loads configuration parameters, applies them to a terrain
    object via its Geometry Nodes modifier, and generates multiple terrain
    instances depending on the selected mode (fixed or random).

    For each generated terrain, it can optionally:
    - Render an image
    - Export the mesh (OBJ and OFF)
    - Generate and save a heightmap

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    None
        Errors are handled internally or propagated from Blender operations.

    Notes
    -----
    - Supports two modes:
        * "fixed": uses predefined parameter values
        * "random": samples parameters based on ranges/probabilities
    - Requires a valid Blender scene with:
        * A terrain object (TERRAIN_NAME)
        * A Geometry Nodes modifier (MODIFIERS_NAME_GM)
    - Outputs are controlled via the dataset_settings section of the config.
    - Geometry Nodes parameters are dynamically applied using set_modifier_value().
    """

    config = load_config()
    settings = config["dataset_settings"]
    params = config["node_parameters"]

    res_xy = params["Vertices Count"]["value"]

    terrain_obj = bpy.data.objects.get(TERRAIN_NAME)
    mod = terrain_obj.modifiers.get(MODIFIERS_NAME_GM)

    print("\n--- DEBUG IDENTIFIERS GEOMETRY NODES ---")
    for key, value in mod.items():
        print(f"Intern Key is: '{key}' (Actual value: {value})")
    print("-------------------------------------------\n")

    mode = settings.get("generate_mode", "fixed")
    num_to_generate = settings["num_terrains_to_generate"] if mode == GENERATION_MODE_RANDOM else 1

    print(f"Mode: {mode}. Generation of {num_to_generate} terrains...")

    for i in range(num_to_generate):
        print(f"\n--- Generation {i+1}/{num_to_generate} ---")
        for param_name, param_data in params.items():
            if mode == GENERATION_MODE_FIXED:
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

        if settings["export_heightmap_exr"]:
            export_heightmap_exr(terrain_obj, 
                             get_full_path(settings["output_folder_maps"], f"{NAME_HEIGHTMAPS}{i:04d}.{EXTENSION_HEIGHTMAP_EXR}"),
                             res_x=res_xy,
                             res_y=res_xy)

        if settings["export_heightmap_asc"]:
            export_heightmap_asc(terrain_obj, 
                             get_full_path(settings["output_folder_maps"], f"{NAME_HEIGHTMAPS}{i:04d}.{EXTENSION_HEIGHTMAP_ASC}"),
                             res_x=res_xy,
                             res_y=res_xy)

        if settings["export_hillshade"]:
            print(get_full_path(settings["output_folder_renders"]))
            render_path = get_full_path(settings["output_folder_renders"], f"{NAME_HILLSHADE}{i:04d}.{EXTENSION_HILLSHADE}")
            export_hillshade(terrain_obj,
                             render_path,
                             res_x=res_xy,
                             res_y=res_xy)






if __name__ == "__main__" :
    generate_terrain()



