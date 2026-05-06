
import bpy
import bmesh
import os
import numpy as np
import torch
import math
import mathutils
import gc

def get_mesh_vertices(mesh_obj):
    vertices = []
    world_matrix = mesh_obj.matrix_world
    for vertex in mesh_obj.data.vertices:
        local_coord = vertex.co
        world_coord = world_matrix @ local_coord
        vertices.append([world_coord.x, world_coord.y, world_coord.z])
    return np.array(vertices)

def get_all_vertices(mesh_objects):
    all_vertices = []
    for mesh_obj in mesh_objects:
        vertices = get_mesh_vertices(mesh_obj)
        all_vertices.append(torch.from_numpy(vertices))
    return all_vertices

def get_mesh_faces(mesh_obj):
    faces = []
    for face in mesh_obj.data.polygons:
        faces.append(list(face.vertices))
    return np.array(faces)

def get_all_faces(mesh_objects):
    all_faces = []
    vertex_count = 0  
    for mesh_obj in mesh_objects:
        faces = get_mesh_faces(mesh_obj)
        if len(all_faces) > 0:
            faces = faces + vertex_count
        all_faces.append(torch.from_numpy(faces))
        vertex_count += len(mesh_obj.data.vertices)
    return all_faces

def clear_scene():
    bpy.ops.wm.read_homefile(use_empty=True)
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # objects_to_delete = [obj for obj in bpy.data.objects if obj.type != 'CAMERA']
    # for obj in objects_to_delete:
    #     bpy.data.objects.remove(obj, do_unlink=True)

def import_model(filepath, frame_idx=None):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at: {filepath}")
    
    clear_scene()
    
    # --- Start modification: Determine import method by file extension ---
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext in ['.glb', '.gltf']:
        # Blender has built-in glTF importer
        bpy.ops.import_scene.gltf(filepath=filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    # --- End modification ---

    # 2. Set scene to animation start frame
    if frame_idx is None:
        bpy.context.scene.frame_set(bpy.context.scene.frame_start)
        print(f"Set scene to frame {bpy.context.scene.frame_start} to capture the first frame.")
    else:
        bpy.context.scene.frame_set(frame_idx)
        print(f"Set scene to frame {frame_idx}.")
        
    # 3. Get all original mesh objects
    original_mesh_objects = []
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            if obj.name.lower() not in ['cube', 'icosphere']:
                original_mesh_objects.append(obj)

    # This logic applies to both glb and fbx files
    if not original_mesh_objects:
        raise ValueError(f"No mesh objects found in the imported file: {filepath}")
        
    depsgraph = bpy.context.evaluated_depsgraph_get()
    static_mesh_objects = []
    
    # 4. Create a static copy for each mesh object at current frame
    for obj in original_mesh_objects:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh_data = bpy.data.meshes.new_from_object(eval_obj)
        static_obj = bpy.data.objects.new(obj.name + "_static_frame0", mesh_data)
        static_obj.matrix_world = obj.matrix_world
        bpy.context.collection.objects.link(static_obj)
        static_mesh_objects.append(static_obj)
        
    # 5. Remove original animated objects from scene
    for obj in original_mesh_objects:
        bpy.data.objects.remove(obj, do_unlink=True)
        
    print("Applying transforms to the new static meshes...")
    for obj in static_mesh_objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
    # print(f"Triangulating mesh: {obj.name}") # Optional: reduce log noise
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        obj.select_set(False)
        
    print(f"Successfully created {len(static_mesh_objects)} static mesh objects from {filepath}.")
    return static_mesh_objects

def full_blender_cleanup():
    """
    A comprehensive Blender cleanup function for loop operations to prevent memory leaks.
    """
    # 1. Clear all objects in current scene
    # This is faster than read_homefile and doesn't reset global configurations like render settings
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
        
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 2. [Key Step] Clear all orphan data blocks
    # This is core to preventing memory leaks. It deletes all data blocks not used by any objects
    # (Meshes, Materials, Textures, Images, etc.)
    # This operation is equivalent to selecting "Orphan Data" in Blender's Outliner and clicking "Purge"
    try:
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        print("Purged orphan data blocks.")
    except RuntimeError:
        print("No orphan data to purge or error during purge.")

    # 3. (Optional but recommended) Force Python garbage collection
    # Ensures Python interpreter also releases memory it occupies
    gc.collect()

def create_mesh(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    for v in verts:
        bm.verts.new(v)
    bm.verts.ensure_lookup_table()
    for f in faces:
        try:
            bm.faces.new([bm.verts[i] for i in f])
        except:
            continue
    bm.to_mesh(mesh)
    mesh.update()
    return obj

def create_gradient_material():
    mat = bpy.data.materials.new(name="GradientMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    # Create nodes
    node_tex_coord = nodes.new(type='ShaderNodeTexCoord')
    node_mapping = nodes.new(type='ShaderNodeMapping')
    node_gradient = nodes.new(type='ShaderNodeTexGradient')
    node_color_ramp = nodes.new(type='ShaderNodeValToRGB')
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    # Set ColorRamp nodes
    color_ramp = node_color_ramp.color_ramp
    color_ramp.elements[0].position = 0.0
    color_ramp.elements[0].color = (1, 0, 0, 1)  # Red
    color_ramp.elements[1].position = 1.0
    color_ramp.elements[1].color = (0, 0, 1, 1)  # Blue
    # Link nodes
    links.new(node_tex_coord.outputs['Generated'], node_mapping.inputs['Vector'])
    links.new(node_mapping.outputs['Vector'], node_gradient.inputs['Vector'])
    links.new(node_gradient.outputs['Color'], node_color_ramp.inputs['Fac'])
    links.new(node_color_ramp.outputs['Color'], node_bsdf.inputs['Base Color'])
    links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])
    return mat

def setup_camera_and_light_to_focus_object(camera, vertex_trajectories, frame_percentage=1.0, size=None, center=None, azi=0, ele=0):
    if size is None:
        center = np.mean(vertex_trajectories, axis=(0, 1))
        min_coords = np.min(vertex_trajectories, axis=(0, 1))
        max_coords = np.max(vertex_trajectories, axis=(0, 1))
        size = np.max(max_coords - min_coords)
    # Set camera
    fov = camera.data.angle
    distance = size / (2 * math.tan(fov / 2) * frame_percentage)
    base_offset = mathutils.Vector((0, -distance, 0))
    azi_rad = math.radians(azi)
    ele_rad = math.radians(ele)
    rotation = mathutils.Euler((math.radians(ele), 0, math.radians(azi)), 'ZYX')
    final_offset = rotation.to_matrix() @ base_offset
    camera.location = mathutils.Vector(center) + final_offset
    direction = mathutils.Vector(center) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    # Set light
    key_light, fill_light, rim_light, ambient_light = setup_lighting()
    key_light.location = camera.location + mathutils.Vector((0, 0, distance/2))
    key_direction = mathutils.Vector(center) - key_light.location
    key_light.rotation_euler = key_direction.to_track_quat('-Z', 'Y').to_euler()
    fill_offset = mathutils.Vector((math.cos(azi_rad + math.pi/2), -math.sin(azi_rad + math.pi/2), 0))
    fill_light.location = camera.location + fill_offset * distance
    fill_direction = mathutils.Vector(center) - fill_light.location
    fill_light.rotation_euler = fill_direction.to_track_quat('-Z', 'Y').to_euler()
    rim_x = -distance * math.cos(ele_rad) * math.sin(azi_rad)
    rim_y = distance * math.cos(ele_rad) * math.cos(azi_rad)
    rim_light.location = mathutils.Vector(center) + mathutils.Vector((rim_x, rim_y, distance))
    rim_direction = mathutils.Vector(center) - rim_light.location
    rim_light.rotation_euler = rim_direction.to_track_quat('-Z', 'Y').to_euler()
    ambient_light.location = mathutils.Vector(center) + mathutils.Vector((0, 0, distance*1.5))
    ambient_direction = mathutils.Vector(center) - ambient_light.location
    ambient_light.rotation_euler = ambient_direction.to_track_quat('-Z', 'Y').to_euler()
    # Set global light
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.1, 0.1, 0.1, 1)
    bg.inputs[1].default_value = 0.2

def setup_lighting():
    # Delete scene light
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj) 
    # Create lights
    key_light = bpy.data.lights.new(name="Key_Light", type='SUN')
    key_light.energy = 3.0 
    key_light.angle = 0.5  
    key_light_obj = bpy.data.objects.new(name="Key_Light", object_data=key_light)
    bpy.context.collection.objects.link(key_light_obj)
    fill_light = bpy.data.lights.new(name="Fill_Light", type='AREA')
    fill_light.energy = 1.5  
    fill_light.size = 5.0  
    fill_light_obj = bpy.data.objects.new(name="Fill_Light", object_data=fill_light)
    bpy.context.collection.objects.link(fill_light_obj)
    rim_light = bpy.data.lights.new(name="Rim_Light", type='SPOT')
    rim_light.energy = 2.0
    rim_light.spot_size = math.radians(45)  
    rim_light.spot_blend = 0.5  
    rim_light_obj = bpy.data.objects.new(name="Rim_Light", object_data=rim_light)
    bpy.context.collection.objects.link(rim_light_obj)
    ambient_light = bpy.data.lights.new(name="Ambient_Light", type='AREA')
    ambient_light.energy = 0.5  
    ambient_light.size = 10.0
    ambient_light_obj = bpy.data.objects.new(name="Ambient_Light", object_data=ambient_light)
    bpy.context.collection.objects.link(ambient_light_obj)
    return key_light_obj, fill_light_obj, rim_light_obj, ambient_light_obj

def setup_camera_to_fit_object(camera, vertex_trajectories, azi=0, ele=0, margin=0.1):
    min_coords = np.min(vertex_trajectories, axis=(0, 1))
    max_coords = np.max(vertex_trajectories, axis=(0, 1))
    center = (min_coords + max_coords) / 2.0
    size = np.max(max_coords - min_coords)
    fov = camera.data.angle
    effective_size = size * (1 + margin)
    distance = effective_size / (2 * math.tan(fov / 2))
    base_offset = mathutils.Vector((0, -distance, 0)) 
    rotation_matrix = mathutils.Euler((math.radians(ele), 0, math.radians(azi)), 'ZYX').to_matrix()
    final_offset = rotation_matrix @ base_offset
    camera.location = mathutils.Vector(center) + final_offset
    direction = mathutils.Vector(center) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    azi_rad = math.radians(azi)
    ele_rad = math.radians(ele)
    key_light, fill_light, rim_light, ambient_light = setup_lighting()
    key_light.location = camera.location + mathutils.Vector((0, 0, distance/2))
    key_direction = mathutils.Vector(center) - key_light.location
    key_light.rotation_euler = key_direction.to_track_quat('-Z', 'Y').to_euler()
    fill_offset = mathutils.Vector((math.cos(azi_rad + math.pi/2), -math.sin(azi_rad + math.pi/2), 0))
    fill_light.location = camera.location + fill_offset * distance
    fill_direction = mathutils.Vector(center) - fill_light.location
    fill_light.rotation_euler = fill_direction.to_track_quat('-Z', 'Y').to_euler()
    rim_x = -distance * math.cos(ele_rad) * math.sin(azi_rad)
    rim_y = distance * math.cos(ele_rad) * math.cos(azi_rad)
    rim_light.location = mathutils.Vector(center) + mathutils.Vector((rim_x, rim_y, distance))
    rim_direction = mathutils.Vector(center) - rim_light.location
    rim_light.rotation_euler = rim_direction.to_track_quat('-Z', 'Y').to_euler()
    ambient_light.location = mathutils.Vector(center) + mathutils.Vector((0, 0, distance*1.5))
    ambient_direction = mathutils.Vector(center) - ambient_light.location
    ambient_light.rotation_euler = ambient_direction.to_track_quat('-Z', 'Y').to_euler()
    # Set global light
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.1, 0.1, 0.1, 1)
    bg.inputs[1].default_value = 0.2

def render_dynamic_mesh_direct_to_video(vertices, face_data, video_save_dir, save_name, resolution=512, azi=0, ele=0, export=False):
    # Clear scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    if "LookAtTarget" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["LookAtTarget"], do_unlink=True)
    # Load mesh data
    if hasattr(vertices, 'numpy'): vertices = vertices.numpy()
    if hasattr(face_data, 'numpy'): face_data = face_data.numpy()  
    vertex_trajectories = vertices
    # Create mesh & material
    initial_verts = vertex_trajectories[0]
    mesh_obj = create_mesh("DynamicMesh", initial_verts, face_data)
    gradient_mat = create_gradient_material()
    mesh_obj.data.materials.append(gradient_mat)
    # Set scene
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = len(vertex_trajectories) - 1
    # Set camera
    cam = bpy.data.objects['Camera']
    setup_camera_and_light_to_focus_object(cam, vertex_trajectories, azi=azi, ele=ele)
    # setup_camera_to_fit_object(cam, vertex_trajectories, azi=azi, ele=ele)
    # Create shape key animation 
    mesh_obj.shape_key_add(name='Basis')
    if mesh_obj.data.shape_keys.animation_data is None:
        mesh_obj.data.shape_keys.animation_data_create()
    for frame_idx in range(len(vertex_trajectories)):
        shape_key = mesh_obj.shape_key_add(name=f'Frame_{frame_idx}')
        shape_key.data.foreach_set('co', vertex_trajectories[frame_idx].flatten())
        shape_key.value = 0
        shape_key.keyframe_insert(data_path='value', frame=frame_idx - 1 if frame_idx > 0 else 0)
        shape_key.value = 1
        shape_key.keyframe_insert(data_path='value', frame=frame_idx)
        shape_key.value = 0
        shape_key.keyframe_insert(data_path='value', frame=frame_idx + 1)
        fcurve = mesh_obj.data.shape_keys.animation_data.action.fcurves.find(f'key_blocks["{shape_key.name}"].value')
        if fcurve:
            for kf_point in fcurve.keyframe_points:
                kf_point.interpolation = 'CONSTANT'
    # Set scene and camera
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = len(vertex_trajectories) - 1
    # Export fbx
    if export:
        # export_dir = os.path.join(video_save_dir, "exported_fbxs")
        # if not os.path.exists(export_dir):
        #     os.makedirs(export_dir)
        # export_path = os.path.join(export_dir, f"{save_name}.fbx")
        # bpy.ops.export_scene.fbx(
        #     filepath=export_path,
        #     use_selection=True,
        #     bake_anim=True,
        #     bake_anim_use_all_bones=False,
        #     bake_anim_use_nla_strips=False,
        #     bake_anim_use_all_actions=False,
        #     bake_anim_force_startend_keying=True,
        #     add_leaf_bones=False,
        #     path_mode='COPY',
        #     embed_textures=True,
        #     use_mesh_modifiers=True,
        #     use_mesh_edges=True,
        #     use_tspace=True,
        #     use_custom_props=True,
        #     use_active_collection=False,
        # )
        # Ensure object is selected
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = mesh_obj
        mesh_obj.select_set(True)
        export_dir = os.path.join(video_save_dir, "exported_abcs")
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        export_path = os.path.join(export_dir, f"{save_name}.abc")
        # Execute Alembic export
        bpy.ops.wm.alembic_export(
            filepath=export_path,
            start=scene.frame_start,
            end=scene.frame_end,
            selected=True,  # Export only selected objects
            # Other settings can remain default, usually sufficient
        )
        print(f"Alembic file exported to: {export_path}")
    # Render settings
    scene.render.engine = 'CYCLES'
    scene.render.fps = 15
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.cycles.samples = 32
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    scene.cycles.device = "GPU"
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    # scene.render.image_settings.color_mode = 'RGB'
    scene.sequencer_colorspace_settings.name = 'sRGB'
    video_path = os.path.join(video_save_dir, f"{save_name}.mp4")
    scene.render.filepath = video_path
    # Start rendering
    print(f"Start rendering (10 FPS), the rendered results are saved to : {video_path}")
    bpy.ops.render.render(animation=True)
    print("Rendering finished!!!")