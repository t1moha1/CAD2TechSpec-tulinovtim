"""
Script to run within blender.

Provide arguments after `--`.
For example: `blender -b -P blender_script.py -- --help`
"""

import argparse
import json
import math
import os
import random
import sys

import bpy
from mathutils import Vector
from mathutils.noise import random_unit_vector
import pickle
import shutil

MAX_DEPTH = 5.0
FORMAT_VERSION = 6

UNIFORM_LIGHT_DIRECTION = None
BASIC_AMBIENT_COLOR = None
BASIC_DIFFUSE_COLOR = None
DEFAULT_NUM_IMAGES = 20
DEFAULT_LIGHT_MODE = "uniform"
DEFAULT_CAMERA_POSE = "random"
DEFAULT_CAMERA_DIST_MIN = 2.0
DEFAULT_CAMERA_DIST_MAX = 2.0
DEFAULT_FAST_MODE = True
DEFAULT_EXTRACT_MATERIAL = False
DEFAULT_DELETE_MATERIAL = False


def get_raw_script_args(argv):
    if "--" not in argv:
        raise SystemExit("Missing '--'. Use: blender -b -P render_script_type2.py -- <script args>")
    return argv[argv.index("--") + 1 :]


def parse_args(raw_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_path_pkl', type=str, default='./example_material/example_object_path.pkl')
    parser.add_argument('--parent_dir', type=str, default='./example_material')
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--backend", type=str, default="BLENDER_EEVEE")
    parser.add_argument("--light_mode", type=str, default="uniform")
    parser.add_argument("--camera_pose", type=str, default="random")
    parser.add_argument("--camera_dist_min", type=float, default=2.0)
    parser.add_argument("--camera_dist_max", type=float, default=2.0)
    parser.add_argument("--fast_mode", action="store_true", default=True)
    parser.add_argument("--extract_material", action="store_true", default=False)
    parser.add_argument("--delete_material", action="store_true")

    default_uniform_light_direction = [0.09387503, -0.63953443, -0.7630093]
    parser.add_argument(
        "--uniform_light_direction",
        type=float,
        nargs='+',
        default=default_uniform_light_direction,
        help="Set the uniform light direction"
    )
    parser.add_argument("--basic_ambient", type=float, default=0.3)
    parser.add_argument("--basic_diffuse", type=float, default=0.7)
    return parser.parse_args(raw_args)


def build_render_options(args):
    options = {
        "num_images": args.num_images if isinstance(args.num_images, int) and args.num_images > 0 else DEFAULT_NUM_IMAGES,
        "light_mode": args.light_mode if args.light_mode in ["random", "uniform", "camera", "basic"] else DEFAULT_LIGHT_MODE,
        "camera_pose": args.camera_pose if args.camera_pose in ["random", "z-circular", "z-circular-elevated"] else DEFAULT_CAMERA_POSE,
        "camera_dist_min": args.camera_dist_min if args.camera_dist_min > 0 else DEFAULT_CAMERA_DIST_MIN,
        "camera_dist_max": args.camera_dist_max if args.camera_dist_max > 0 else DEFAULT_CAMERA_DIST_MAX,
        "fast_mode": args.fast_mode if isinstance(args.fast_mode, bool) else DEFAULT_FAST_MODE,
        "extract_material": args.extract_material if isinstance(args.extract_material, bool) else DEFAULT_EXTRACT_MATERIAL,
        "delete_material": args.delete_material if isinstance(args.delete_material, bool) else DEFAULT_DELETE_MATERIAL,
    }

    if options["camera_dist_min"] > options["camera_dist_max"]:
        options["camera_dist_min"], options["camera_dist_max"] = options["camera_dist_max"], options["camera_dist_min"]

    if options["light_mode"] == "basic" and options["extract_material"]:
        options["extract_material"] = False

    if options["delete_material"] and options["extract_material"]:
        options["extract_material"] = False

    return options


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def clear_lights():
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Light):
            obj.select_set(True)
    bpy.ops.object.delete()


def import_model(path):
    clear_scene()
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".stl":
        bpy.ops.import_mesh.stl(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".dae":
        bpy.ops.wm.collada_import(filepath=path)
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=path)
    else:
        raise RuntimeError(f"unexpected extension: {ext}")


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)

    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    bpy.context.view_layer.update()

    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    bpy.ops.object.select_all(action="DESELECT")


def create_camera():
    camera_data = bpy.data.cameras.new(name="Camera")
    camera_object = bpy.data.objects.new("Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object


def set_camera(direction, camera_dist=2.0):
    camera_pos = -camera_dist * direction
    bpy.context.scene.camera.location = camera_pos

    rot_quat = direction.to_track_quat("-Z", "Y")
    bpy.context.scene.camera.rotation_euler = rot_quat.to_euler()

    bpy.context.view_layer.update()


def randomize_camera(camera_dist=2.0):
    direction = random_unit_vector()
    set_camera(direction, camera_dist=camera_dist)


def pan_camera(time, axis="Z", camera_dist=2.0, elevation=0.1):
    angle = time * math.pi * 2
    direction = [-math.cos(angle), -math.sin(angle), elevation]
    assert axis in ["X", "Y", "Z"]
    if axis == "X":
        direction = [direction[2], *direction[:2]]
    elif axis == "Y":
        direction = [direction[0], elevation, direction[1]]
    direction = Vector(direction).normalized()
    set_camera(direction, camera_dist=camera_dist)


def place_camera(time, camera_pose_mode="random", camera_dist_min=2.0, camera_dist_max=2.0):
    camera_dist = random.uniform(camera_dist_min, camera_dist_max)
    if camera_pose_mode == "random":
        randomize_camera(camera_dist=camera_dist)
    elif camera_pose_mode == "z-circular":
        pan_camera(time, axis="Z", camera_dist=camera_dist)
    elif camera_pose_mode == "z-circular-elevated":
        pan_camera(time, axis="Z", camera_dist=camera_dist, elevation=-0.2617993878)
    else:
        raise ValueError(f"Unknown camera pose mode: {camera_pose_mode}")


def create_light(location, energy=1.0, angle=0.5 * math.pi / 180):
    light_data = bpy.data.lights.new(name="Light", type="SUN")
    light_data.energy = energy
    light_data.angle = angle
    light_object = bpy.data.objects.new(name="Light", object_data=light_data)

    direction = -location
    rot_quat = direction.to_track_quat("-Z", "Y")
    light_object.rotation_euler = rot_quat.to_euler()
    bpy.context.view_layer.update()

    bpy.context.collection.objects.link(light_object)
    light_object.location = location


def create_random_lights(count=4, distance=2.0, energy=1.5):
    clear_lights()
    for _ in range(count):
        create_light(random_unit_vector() * distance, energy=energy)


def create_camera_light():
    clear_lights()
    create_light(bpy.context.scene.camera.location, energy=5.0)


def create_uniform_light(backend):
    clear_lights()
    pos = Vector(UNIFORM_LIGHT_DIRECTION)
    angle = 0.0092 if backend == "CYCLES" else math.pi
    create_light(pos, energy=5.0, angle=angle)
    create_light(-pos, energy=5.0, angle=angle)


def create_vertex_color_shaders():
    for obj in bpy.context.scene.objects.values():
        if not isinstance(obj.data, (bpy.types.Mesh)):
            continue

        if len(obj.data.materials):
            continue

        color_keys = (obj.data.vertex_colors or {}).keys()
        if not len(color_keys):
            continue

        mat = bpy.data.materials.new(name="VertexColored")
        mat.use_nodes = True

        bsdf_node = None
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                bsdf_node = node
        assert bsdf_node is not None, "material has no Principled BSDF node to modify"

        socket_map = {}
        for input in bsdf_node.inputs:
            socket_map[input.name] = input

        if "Specular" in socket_map:
            socket_map["Specular"].default_value = 0.0
        elif "Specular IOR Level" in socket_map:
            socket_map["Specular IOR Level"].default_value = 0.0
        if "Roughness" in socket_map:
            socket_map["Roughness"].default_value = 1.0

        v_color = mat.node_tree.nodes.new("ShaderNodeVertexColor")
        v_color.layer_name = color_keys[0]

        mat.node_tree.links.new(v_color.outputs[0], socket_map["Base Color"])

        obj.data.materials.append(mat)


def create_default_materials():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            if not len(obj.data.materials):
                mat = bpy.data.materials.new(name="DefaultMaterial")
                mat.use_nodes = True

                bsdf = None
                for node in mat.node_tree.nodes:
                    if node.type == "BSDF_PRINCIPLED":
                        bsdf = node
                        break
                if bsdf is not None:
                    inputs = {socket.name: socket for socket in bsdf.inputs}
                    if "Base Color" in inputs:
                        inputs["Base Color"].default_value = (0.62, 0.64, 0.68, 1.0)
                    if "Metallic" in inputs:
                        inputs["Metallic"].default_value = 1.0
                    if "Roughness" in inputs:
                        inputs["Roughness"].default_value = 0.2

                obj.data.materials.append(mat)


def apply_metallic_look_to_all_materials():
    for mat in find_materials():
        mat.use_nodes = True
        bsdf = None
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                bsdf = node
                break
        if bsdf is None:
            continue

        inputs = {socket.name: socket for socket in bsdf.inputs}
        if "Base Color" in inputs:
            inputs["Base Color"].default_value = (0.62, 0.64, 0.68, 1.0)
        if "Metallic" in inputs:
            inputs["Metallic"].default_value = 1.0
        if "Roughness" in inputs:
            inputs["Roughness"].default_value = 0.18


def find_materials():
    all_materials = set()
    for obj in bpy.context.scene.objects.values():
        if not isinstance(obj.data, bpy.types.Mesh):
            continue
        for mat in obj.data.materials:
            all_materials.add(mat)
    return all_materials


def delete_all_materials():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Mesh):
            obj.data.materials.clear()


def setup_material_extraction_shaders(capturing_material_alpha: bool):
    """
    Change every material to emit texture colors (or alpha) rather than having
    an actual reflective color. Returns a function to undo the changes to the
    materials.
    """
    undo_fns = []
    for mat in find_materials():
        undo_fn = setup_material_extraction_shader_for_material(mat, capturing_material_alpha)
        if undo_fn is not None:
            undo_fns.append(undo_fn)
    return lambda: [undo_fn() for undo_fn in undo_fns]


def setup_material_extraction_shader_for_material(mat, capturing_material_alpha: bool):
    mat.use_nodes = True

    bsdf_node = None
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            bsdf_node = node
    assert bsdf_node is not None, "material has no Principled BSDF node to modify"

    socket_map = {}
    for input in bsdf_node.inputs:
        socket_map[input.name] = input

    def pick_socket(*names):
        for name in names:
            if name in socket_map:
                return socket_map[name]
        raise AssertionError(f"None of {names} in {list(socket_map.keys())}")

    base_color_socket = pick_socket("Base Color")
    alpha_socket = pick_socket("Alpha")
    emission_socket = pick_socket("Emission", "Emission Color")
    emission_strength_socket = pick_socket("Emission Strength")
    specular_socket = pick_socket("Specular", "Specular IOR Level")

    old_base_color = get_socket_value(mat.node_tree, base_color_socket)
    old_alpha = get_socket_value(mat.node_tree, alpha_socket)
    old_emission = get_socket_value(mat.node_tree, emission_socket)
    old_emission_strength = get_socket_value(mat.node_tree, emission_strength_socket)
    old_specular = get_socket_value(mat.node_tree, specular_socket)

    clear_socket_input(mat.node_tree, base_color_socket)
    base_color_socket.default_value = [0, 0, 0, 1]
    clear_socket_input(mat.node_tree, alpha_socket)
    alpha_socket.default_value = 1
    clear_socket_input(mat.node_tree, specular_socket)
    specular_socket.default_value = 0.0

    old_blend_method = mat.blend_method
    mat.blend_method = "OPAQUE"

    if capturing_material_alpha:
        set_socket_value(mat.node_tree, emission_socket, old_alpha)
    else:
        set_socket_value(mat.node_tree, emission_socket, old_base_color)
    clear_socket_input(mat.node_tree, emission_strength_socket)
    emission_strength_socket.default_value = 1.0

    def undo_fn():
        mat.blend_method = old_blend_method
        set_socket_value(mat.node_tree, base_color_socket, old_base_color)
        set_socket_value(mat.node_tree, alpha_socket, old_alpha)
        set_socket_value(mat.node_tree, emission_socket, old_emission)
        set_socket_value(mat.node_tree, emission_strength_socket, old_emission_strength)
        set_socket_value(mat.node_tree, specular_socket, old_specular)

    return undo_fn


def get_socket_value(tree, socket):
    default = socket.default_value
    if not isinstance(default, float):
        default = list(default)
    for link in tree.links:
        if link.to_socket == socket:
            return (link.from_socket, default)
    return (None, default)


def clear_socket_input(tree, socket):
    for link in list(tree.links):
        if link.to_socket == socket:
            tree.links.remove(link)


def set_socket_value(tree, socket, socket_and_default):
    clear_socket_input(tree, socket)
    old_source_socket, default = socket_and_default
    if isinstance(default, float) and not isinstance(socket.default_value, float):
        socket.default_value = [default] * 3 + [1.0]
    else:
        socket.default_value = default
    if old_source_socket is not None:
        tree.links.new(old_source_socket, socket)


def get_compositor_tree(scene):
    if hasattr(scene, "compositing_node_group"):
        tree = scene.compositing_node_group
        if tree is None:
            tree = bpy.data.node_groups.new("CAD2TechSpecCompositor", "CompositorNodeTree")
            scene.compositing_node_group = tree
        return tree

    scene.use_nodes = True
    return scene.node_tree


def setup_nodes(output_path, capturing_material_alpha: bool = False, basic_lighting: bool = False):
    tree = get_compositor_tree(bpy.context.scene)
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)

    def node_op(op: str, *args, clamp=False):
        node = tree.nodes.new(type="CompositorNodeMath")
        node.operation = op
        if clamp:
            node.use_clamp = True
        for i, arg in enumerate(args):
            if isinstance(arg, (int, float)):
                node.inputs[i].default_value = arg
            else:
                links.new(arg, node.inputs[i])
        return node.outputs[0]

    def node_clamp(x, maximum=1.0):
        return node_op("MINIMUM", x, maximum)

    def node_mul(x, y, **kwargs):
        return node_op("MULTIPLY", x, y, **kwargs)

    def node_add(x, y, **kwargs):
        return node_op("ADD", x, y, **kwargs)

    def node_abs(x, **kwargs):
        return node_op("ABSOLUTE", x, **kwargs)

    def try_new_node(*node_types):
        last_error = None
        for node_type in node_types:
            try:
                return tree.nodes.new(type=node_type)
            except RuntimeError as exc:
                last_error = exc
        raise last_error

    def set_output_node_path(node, path_value):
        if hasattr(node, "base_path"):
            node.base_path = path_value
            return
        if hasattr(node, "directory"):
            node.directory = path_value
            return
        raise AttributeError("Compositor output node path attribute not found")

    input_node = tree.nodes.new(type="CompositorNodeRLayers")
    input_node.scene = bpy.context.scene

    input_sockets = {}
    for output in input_node.outputs:
        input_sockets[output.name] = output

    if capturing_material_alpha:
        color_socket = input_sockets["Image"]
    else:
        raw_color_socket = input_sockets["Image"]
        if basic_lighting:
            try:
                normal_xyz = tree.nodes.new(type="CompositorNodeSeparateXYZ")
                tree.links.new(input_sockets["Normal"], normal_xyz.inputs[0])
                normal_x, normal_y, normal_z = [normal_xyz.outputs[i] for i in range(3)]
                dot = node_add(
                    node_mul(UNIFORM_LIGHT_DIRECTION[0], normal_x),
                    node_add(
                        node_mul(UNIFORM_LIGHT_DIRECTION[1], normal_y),
                        node_mul(UNIFORM_LIGHT_DIRECTION[2], normal_z),
                    ),
                )
                diffuse = node_abs(dot)
                brightness = node_add(BASIC_AMBIENT_COLOR, node_mul(BASIC_DIFFUSE_COLOR, diffuse))
                rgba_node = try_new_node("CompositorNodeSepRGBA", "CompositorNodeSeparateColor")
                if hasattr(rgba_node, "mode"):
                    rgba_node.mode = "RGB"
                tree.links.new(raw_color_socket, rgba_node.inputs[0])
                combine_node = try_new_node("CompositorNodeCombRGBA", "CompositorNodeCombineColor")
                if hasattr(combine_node, "mode"):
                    combine_node.mode = "RGB"
                for i in range(3):
                    tree.links.new(node_mul(rgba_node.outputs[i], brightness), combine_node.inputs[i])
                if len(rgba_node.outputs) > 3 and len(combine_node.inputs) > 3:
                    tree.links.new(rgba_node.outputs[3], combine_node.inputs[3])
                raw_color_socket = combine_node.outputs[0]
            except RuntimeError:
                basic_lighting = False

        color_node = tree.nodes.new(type="CompositorNodeConvertColorSpace")
        color_node.from_color_space = "Linear Rec.2020"
        color_node.to_color_space = "sRGB"
        tree.links.new(raw_color_socket, color_node.inputs[0])
        color_socket = color_node.outputs[0]

    split_node = try_new_node("CompositorNodeSepRGBA", "CompositorNodeSeparateColor")
    if hasattr(split_node, "mode"):
        split_node.mode = "RGB"
    tree.links.new(color_socket, split_node.inputs[0])

    split_outputs = [split_node.outputs[i] for i in range(len(split_node.outputs))]
    alpha_output = split_outputs[3] if len(split_outputs) > 3 else None

    channel_sources = {}
    if capturing_material_alpha:
        channel_sources["MatAlpha"] = split_outputs[0]
    else:
        channel_sources["r"] = split_outputs[0]
        channel_sources["g"] = split_outputs[1] if len(split_outputs) > 1 else split_outputs[0]
        channel_sources["b"] = split_outputs[2] if len(split_outputs) > 2 else split_outputs[0]
        channel_sources["a"] = alpha_output if alpha_output is not None else split_outputs[0]

    for channel, source_socket in channel_sources.items():
        output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        set_output_node_path(output_node, f"{output_path}_{channel}")
        links.new(source_socket, output_node.inputs[0])

    if capturing_material_alpha:
        return

    # Blender 5 compositor may not expose legacy math nodes in all modes.
    # Keep depth export compatible by writing the depth pass directly.
    depth_out = input_sockets["Depth"]
    output_node = tree.nodes.new(type="CompositorNodeOutputFile")
    set_output_node_path(output_node, f"{output_path}_depth")
    links.new(depth_out, output_node.inputs[0])


def render_scene(output_path, fast_mode: bool, extract_material: bool, basic_lighting: bool):
    use_workbench = False
    bpy.context.scene.render.engine == "CYCLES"
    bpy.context.scene.cycles.samples = 16
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    bpy.context.view_layer.update()
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    if basic_lighting:
        bpy.context.scene.view_layers["ViewLayer"].use_pass_normal = True
    bpy.context.scene.view_settings.view_transform = "Raw"
    bpy.context.scene.render.film_transparent = False
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"
    bpy.context.scene.render.image_settings.color_depth = "8"
    if bpy.context.scene.world and bpy.context.scene.world.node_tree:
        bg = bpy.context.scene.world.node_tree.nodes.get("Background")
        if bg is not None:
            bg.inputs[0].default_value = (0.72, 0.82, 0.95, 1.0)
    bpy.context.scene.render.filepath = output_path
    if extract_material:
        for do_alpha in [False, True]:
            undo_fn = setup_material_extraction_shaders(capturing_material_alpha=do_alpha)
            setup_nodes(output_path, capturing_material_alpha=do_alpha)
            bpy.ops.render.render(write_still=True)
            undo_fn()
    else:
        setup_nodes(output_path, basic_lighting=basic_lighting)
        bpy.ops.render.render(write_still=True)

    expected_channels = ["r", "g", "b", "a", "depth", *(["MatAlpha"] if extract_material else [])]
    missing_channels = []
    for channel_name in expected_channels:
        sub_dir = f"{output_path}_{channel_name}"
        name, ext = os.path.splitext(output_path)
        channel_path = f"{name}_{channel_name}{ext}"

        if os.path.isdir(sub_dir):
            files = os.listdir(sub_dir)
            if files:
                image_path = os.path.join(sub_dir, files[0])
                if channel_name == "depth" or not use_workbench:
                    os.rename(image_path, channel_path)
                else:
                    os.remove(image_path)
                os.removedirs(sub_dir)
                continue

        if not os.path.exists(channel_path):
            missing_channels.append(channel_name)

    if missing_channels and os.path.exists(output_path):
        for channel_name in missing_channels:
            name, ext = os.path.splitext(output_path)
            channel_path = f"{name}_{channel_name}{ext}"
            shutil.copy2(output_path, channel_path)

    if use_workbench:
        bpy.context.scene.use_nodes = False
        bpy.context.scene.render.engine == "CYCLES"
        bpy.context.scene.cycles.samples = 16
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        bpy.context.scene.render.image_settings.color_mode = "RGBA"
        bpy.context.scene.render.image_settings.color_depth = "16"
        os.remove(output_path)
        bpy.ops.render.render(write_still=True)


def scene_fov():
    x_fov = bpy.context.scene.camera.data.angle_x
    y_fov = bpy.context.scene.camera.data.angle_y
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y
    if bpy.context.scene.camera.data.angle == x_fov:
        y_fov = 2 * math.atan(math.tan(x_fov / 2) * height / width)
    else:
        x_fov = 2 * math.atan(math.tan(y_fov / 2) * width / height)
    return x_fov, y_fov


def write_camera_metadata(path):
    x_fov, y_fov = scene_fov()
    bbox_min, bbox_max = scene_bbox()
    matrix = bpy.context.scene.camera.matrix_world
    with open(path, "w") as f:
        json.dump(
            dict(
                format_version=FORMAT_VERSION,
                max_depth=MAX_DEPTH,
                bbox=[list(bbox_min), list(bbox_max)],
                origin=list(matrix.col[3])[:3],
                x_fov=x_fov,
                y_fov=y_fov,
                x=list(matrix.col[0])[:3],
                y=list(-matrix.col[1])[:3],
                z=list(-matrix.col[2])[:3],
            ),
            f,
        )


def save_rendering_dataset(
    input_path: str,
    output_path: str,
    num_images: int,
    backend: str,
    light_mode: str,
    camera_pose: str,
    camera_dist_min: float,
    camera_dist_max: float,
    fast_mode: bool,
    extract_material: bool,
    delete_material: bool,
):
    basic_lighting = light_mode == "basic"

    import_model(input_path)
    bpy.context.scene.render.engine = backend
    normalize_scene()
    if light_mode == "random":
        create_random_lights()
    elif light_mode == "uniform":
        create_uniform_light(backend)
    create_camera()
    create_vertex_color_shaders()
    create_default_materials()
    apply_metallic_look_to_all_materials()

    if delete_material:
        delete_all_materials()
    if extract_material or basic_lighting:
        create_default_materials()
    if basic_lighting:
        setup_material_extraction_shaders(capturing_material_alpha=False)

    camera_data = []
    camera = bpy.context.scene.camera
    camera_angle_x = camera.data.angle_x

    for i in range(num_images):
        t = i / max(num_images - 1, 1)
        place_camera(
            t,
            camera_pose_mode=camera_pose,
            camera_dist_min=camera_dist_min,
            camera_dist_max=camera_dist_max,
        )
        if light_mode == "camera":
            create_camera_light()

        transform_matrix = bpy.context.scene.camera.matrix_world
        rotation = transform_matrix.to_euler()[2]

        transform_matrix_list = []
        for row in transform_matrix:
            transform_matrix_list.append(list(row))
        camera_data.append({
            "file_path": f"{i:05}",
            "rotation": rotation,
            "transform_matrix": transform_matrix_list
        })

        render_scene(
            os.path.join(output_path, f"{i:05}.png"),
            fast_mode=fast_mode,
            extract_material=extract_material,
            basic_lighting=basic_lighting,
        )
        write_camera_metadata(os.path.join(output_path, f"{i:05}.json"))
    output_data = {"camera_angle_x": camera_angle_x, "frames": camera_data}
    camera_data_json = json.dumps(output_data, indent=4)
    with open(os.path.join(output_path, 'transforms_train.json'), 'w') as file:
        file.write(camera_data_json)
    with open(os.path.join(output_path, "info.json"), "w") as f:
        info = dict(
            backend=backend,
            light_mode=light_mode,
            fast_mode=fast_mode,
            extract_material=extract_material,
            format_version=FORMAT_VERSION,
            channels=["R", "G", "B", "A", "D", *(["MatAlpha"] if extract_material else [])],
            scale=0.5,
        )
        json.dump(info, f)


def render_script_type2():
    raw_args = get_raw_script_args(sys.argv)
    args = parse_args(raw_args)
    render_options = build_render_options(args)

    global UNIFORM_LIGHT_DIRECTION, BASIC_AMBIENT_COLOR, BASIC_DIFFUSE_COLOR
    UNIFORM_LIGHT_DIRECTION = args.uniform_light_direction
    BASIC_AMBIENT_COLOR = args.basic_ambient
    BASIC_DIFFUSE_COLOR = args.basic_diffuse

    uid_paths = pickle.load(open(args.object_path_pkl, 'rb'))
    for uid in uid_paths:
        if not os.path.exists(uid):
            print('object not exist, check the file path')
            continue

        cur_output_path = os.path.join(args.parent_dir, 'rendered_imgs/%s'%(uid.split('/')[-1].split('.')[0]))
        save_rendering_dataset(
            input_path=uid,
            output_path=cur_output_path,
            num_images=render_options["num_images"],
            backend="CYCLES",
            light_mode=render_options["light_mode"],
            camera_pose=render_options["camera_pose"],
            camera_dist_min=render_options["camera_dist_min"],
            camera_dist_max=render_options["camera_dist_max"],
            fast_mode=render_options["fast_mode"],
            extract_material=render_options["extract_material"],
            delete_material=render_options["delete_material"],
        )


render_script_type2()
