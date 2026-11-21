"""
Blender-side script of the PNG -> 3D terrain pipeline.

For every PNG heightmap found in the input folder it assigns the image to the
Image Texture node of the Geometry Nodes modifier, lets Blender rebuild the 3D
terrain, and exports it as an OBJ file.

Arguments (after "--"): <input_dir> <obj_output_dir>
"""

import os
import sys
import glob
import numpy as np

import bpy

TERRAIN_NAME = "Plane"      
MODIFIER_NAME = "GeometryNodes"       
IMAGE_NODE_NAME = "Image Texture.005"     
INPUT_EXTENSIONS = (".png",)

IMAGE_NODE_IDNAMES = ("GeometryNodeImageTexture", "ShaderNodeTexImage")



def parse_args():
    """
    Read the command-line arguments passed to the script after ``--``.

    Returns
    -------
    tuple[str, str]
        Absolute path of the input folder with the PNG heightmaps and absolute
        path of the output folder for the OBJ files.

    Raises
    ------
    SystemExit
        If the ``--`` separator or one of the two arguments is missing.
    """
    if "--" not in sys.argv:
        raise SystemExit("Usage: blender -b file.blend -P script.py -- <input_dir> <obj_output_dir>")
    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) < 2:
        raise SystemExit("Both <input_dir> and <obj_output_dir> are required.")
    return os.path.abspath(args[0]), os.path.abspath(args[1])


def find_terrain():
    """
    Find the terrain object and its Geometry Nodes modifier.

    Returns
    -------
    tuple[bpy.types.Object, bpy.types.NodesModifier]
        The terrain object and the modifier that generates it.

    Raises
    ------
    RuntimeError
        If no mesh object with a matching Geometry Nodes modifier exists.

    Notes
    -----
    - ``TERRAIN_NAME`` and ``MODIFIER_NAME`` restrict the search; when None,
      the first mesh object with a Geometry Nodes modifier is used.
    """
    candidates = [bpy.data.objects[TERRAIN_NAME]] if TERRAIN_NAME else list(bpy.data.objects)
    for obj in candidates:
        if obj.type != 'MESH':
            continue
        for mod in obj.modifiers:
            if mod.type == 'NODES' and (MODIFIER_NAME is None or mod.name == MODIFIER_NAME):
                return obj, mod
    raise RuntimeError("No terrain object with a Geometry Nodes modifier found.")


def find_image_node(mod):
    """
    Find the Image Texture node that reads the heightmap.

    Parameters
    ----------
    mod : bpy.types.NodesModifier
        The Geometry Nodes modifier whose node group is searched.

    Returns
    -------
    bpy.types.Node
        The Image Texture node.

    Raises
    ------
    RuntimeError
        If the node group contains no Image Texture node.

    Notes
    -----
    - If ``IMAGE_NODE_NAME`` is set, the node with that name is returned;
      otherwise the first node whose type is in ``IMAGE_NODE_IDNAMES`` is used.
    - Only the top level of the node group is searched. A node placed inside
      a nested group must be selected with ``IMAGE_NODE_NAME`` after adapting
      this function.
    """
    nodes = mod.node_group.nodes
    if IMAGE_NODE_NAME:
        return nodes[IMAGE_NODE_NAME]
    for node in nodes:
        if node.bl_idname in IMAGE_NODE_IDNAMES:
            return node
    raise RuntimeError("No Image Texture node found in the node group.")


def get_node_image(image_node):
    """
    Return the image currently assigned to an Image Texture node.

    Parameters
    ----------
    image_node : bpy.types.Node
        The Image Texture node.

    Returns
    -------
    bpy.types.Image or None
        The assigned image, or None if the node has no image.

    Notes
    -----
    - In Geometry Nodes the image is the default value of the input socket
      named "Image". In shader/texture nodes it is the ``image`` property.
    """
    if hasattr(image_node, "image"):
        return image_node.image
    return image_node.inputs["Image"].default_value


def set_node_image(image_node, img):
    """
    Assign an image to an Image Texture node.

    Parameters
    ----------
    image_node : bpy.types.Node
        The Image Texture node.

    img : bpy.types.Image
        The image to assign.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the "Image" input socket is linked to another node, because its
        default value is ignored in that case.

    Notes
    -----
    - In Geometry Nodes the image is set on the input socket named "Image"
      (the node has no ``image`` attribute). In shader/texture nodes it is
      set on the ``image`` property.
    """
    if hasattr(image_node, "image"):
        image_node.image = img
        return

    socket = image_node.inputs["Image"]
    if socket.is_linked:
        raise RuntimeError("The 'Image' socket is linked to another node; "
                           "its value cannot be set from the script.")
    socket.default_value = img


def open_image_in_node(image_node, path):
    """
    Open a PNG heightmap in the Image Texture node.

    This does the same thing as the "Open" button of the node. The image that
    is already assigned to the node is reused and pointed to the new file, so
    the node keeps its settings (interpolation, extension, links).

    Parameters
    ----------
    image_node : bpy.types.Node
        The Image Texture node that feeds the terrain height.

    path : str
        Absolute path of the PNG file to open.

    Returns
    -------
    bpy.types.Image
        The image now assigned to the node.

    Notes
    -----
    - If the node has no image, or its image is packed inside the .blend file
      (a packed image would ignore the new file path), a new image is loaded
      from disk and assigned to the node.
    - The color space is set to ``Non-Color`` so that pixel values are used as
      raw heights, without the sRGB transform.
    """
    img = get_node_image(image_node)

    if img is None or img.packed_file is not None:
        img = bpy.data.images.load(path, check_existing=False)
        set_node_image(image_node, img)
    else:
        img.source = 'FILE'
        img.filepath = path
        img.reload()

    img.colorspace_settings.name = 'Non-Color'
    img.update()
    return img


def refresh_geometry(obj, mod):
    """
    Force Blender to re-evaluate the Geometry Nodes of the terrain.

    Parameters
    ----------
    obj : bpy.types.Object
        The terrain object.

    mod : bpy.types.NodesModifier
        The Geometry Nodes modifier of the terrain.

    Returns
    -------
    None

    Notes
    -----
    - In background mode nothing triggers a redraw, so the node tree and the
      object are tagged as changed and the dependency graph is updated
      explicitly.
    """
    mod.node_group.update_tag()
    obj.update_tag()
    bpy.context.view_layer.update()
    bpy.context.evaluated_depsgraph_get().update()


def terrain_height_range(obj):
    """
    Measure the minimum and maximum Z of the evaluated terrain.

    Parameters
    ----------
    obj : bpy.types.Object
        The terrain object.

    Returns
    -------
    tuple[float, float]
        Minimum and maximum Z of the vertices after the modifiers are applied.

    Notes
    -----
    - Used as a sanity check: if the range does not change when a new image is
      opened, the Geometry Nodes were not updated.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", coords)
    finally:
        eval_obj.to_mesh_clear()
    z = coords[2::3]
    return float(z.min()), float(z.max())


def export_obj(obj, filepath):
    """
    Export the evaluated terrain mesh as an OBJ file.

    Parameters
    ----------
    obj : bpy.types.Object
        The terrain object to export.

    filepath : str
        Destination path of the ``.obj`` file.

    Returns
    -------
    None

    Notes
    -----
    - Only the given object is exported, with its modifiers applied (so the
      mesh generated by the Geometry Nodes is what ends up in the file).
    - The object is made visible and selectable first, because a hidden object
      cannot be selected and would be skipped by the exporter.
    - Axes are exported as ``forward = Y`` and ``up = Z``, i.e. Blender's own
      coordinates, so the terrain height stays on the Z axis. The Blender
      default (Y-up) would move the heights to the Y axis and break the later
      OFF -> TIFF conversion.
    """
    obj.hide_set(False)
    obj.hide_viewport = False
    obj.hide_select = False

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(
        filepath=filepath,
        export_selected_objects=True,
        forward_axis='Y',
        up_axis='Z',
    )


def main():
    """
    Convert every PNG heightmap in the input folder into an OBJ terrain.

    For each image the function opens it in the Image Texture node, refreshes
    the Geometry Nodes result, exports the OBJ file and prints
    ``BLENDER_IMAGE_DONE`` so that the launcher can update its progress bar.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If the arguments are invalid or no PNG file is found.

    RuntimeError
        If the terrain object or the Image Texture node cannot be found, or
        if the node's "Image" socket is linked.

    Notes
    -----
    - Each OBJ file has the same base name as its PNG file.
    - The height range of every terrain is printed. A warning is shown when it
      is identical to the previous one, which usually means that the Geometry
      Nodes did not update (or that two images are identical).
    """
    input_dir, obj_dir = parse_args()
    os.makedirs(obj_dir, exist_ok=True)

    files = sorted(
        f for ext in INPUT_EXTENSIONS
        for f in glob.glob(os.path.join(input_dir, f"*{ext}"))
    )
    if not files:
        raise SystemExit(f"No input images found in {input_dir}")

    obj, mod = find_terrain()
    image_node = find_image_node(mod)
    print(f"Terrain: {obj.name} | Modifier: {mod.name} | Image node: {image_node.name}")

    previous_range = None
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]

        img = open_image_in_node(image_node, path)
        print(f"Opened {stem}: {img.size[0]}x{img.size[1]} px")

        refresh_geometry(obj, mod)

        z_range = terrain_height_range(obj)
        print(f"Height range Z: {z_range[0]:.4f} .. {z_range[1]:.4f}")
        if previous_range is not None and z_range == previous_range:
            print("WARNING: height range identical to the previous terrain "
                  "(Geometry Nodes may not have updated).")
        previous_range = z_range

        export_obj(obj, os.path.join(obj_dir, f"{stem}.obj"))
        print(f"OBJ saved: {stem}.obj")
        print("BLENDER_IMAGE_DONE", flush=True)


if __name__ == "__main__":
    main()