
import bpy
import os
import pickle
import mathutils
from mathutils import Vector
import math
import numpy as np

def setup_environment_lighting(env_color=(1.0, 1.0, 1.0), env_strength=2.0):
    world = bpy.data.worlds.get('World')
    if world is None:
        world = bpy.data.worlds.new(name='World')
    bpy.context.scene.world = world
    
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get('Background')
    if not bg_node:
        world.node_tree.nodes.clear()
        output_node = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
        bg_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
        world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

    bg_node.inputs['Color'].default_value = (*env_color, 1.0)
    bg_node.inputs['Strength'].default_value = env_strength
    
    return {
        'color': tuple(env_color),
        'strength': env_strength
    }

def setup_scene_with_dynamic_camera(mesh_obj, trajs, azi=0.0, ele=0.0, env_color=(1.0, 1.0, 1.0), env_strength=2.0):
    
    print("Calculating dynamic bounding box for camera setup...")
    all_positions = []
    for traj in trajs:
        if hasattr(traj, 'numpy'):
            traj_np = traj.numpy()
        else:
            traj_np = np.array(traj)
        # traj_np shape: [num_frames, num_vertices, 3]
        all_positions.append(traj_np.reshape(-1, 3))
    
    all_positions = np.vstack(all_positions)  # [total_points, 3]
    
    bbox_min = np.min(all_positions, axis=0)
    bbox_max = np.max(all_positions, axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = bbox_max - bbox_min
    max_dimension = np.max(bbox_size)
    
    print(f"Bounding box center: {bbox_center}")
    print(f"Bounding box size: {bbox_size}")
    print(f"Max dimension: {max_dimension}")
    
    camera_distance = max_dimension * 2.5  
    
    base_location = Vector((0, -camera_distance, 0))
    rotation = mathutils.Euler((math.radians(ele), 0, math.radians(azi)), 'ZYX')
    camera_offset = rotation.to_matrix() @ base_location
    camera_location = Vector(bbox_center) + camera_offset
    
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    
    direction = Vector(bbox_center) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = max_dimension * 1.3 
    
    cameras = [camera]
    
    print(f"Setting up environment lighting: color={env_color}, strength={env_strength}")
    setup_environment_lighting(env_color, env_strength)
    
    bpy.context.scene.render.film_transparent = True
    
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.cycles.max_bounces = 2
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transmission_bounces = 1
    bpy.context.scene.cycles.volume_bounces = 0
    bpy.context.scene.cycles.transparent_max_bounces = 1
    
    for obj in cameras:
        obj.lock_location = (True, True, True)
        obj.lock_rotation = (True, True, True)
        obj.lock_scale = (True, True, True)
    
    print(f"Camera setup complete: ortho_scale={camera.data.ortho_scale:.2f}")
    
    return cameras

def setup_scene_from_params(params_path: str):
    
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Param file not found: {params_path}")
    
    with open(params_path, 'rb') as f:
        scene_params = pickle.load(f)

    cam_params = scene_params.get('camera')
    if not cam_params:
        raise ValueError("No 'camera' parameter found.")

    target_location = Vector(cam_params['target'])
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=target_location)
    target_empty = bpy.context.object
    target_empty.name = "RecreatedModelCenterTarget"

    cam_location = Vector(cam_params['location'])
    bpy.ops.object.camera_add(location=cam_location)
    camera_obj = bpy.context.object
    camera_obj.name = "RecreatedCamera"
    
    camera = camera_obj.data
    camera.angle = cam_params['fov']
    camera.clip_start = cam_params['clip_start']
    camera.clip_end = cam_params['clip_end']
    
    constraint = camera_obj.constraints.new(type='TRACK_TO')
    constraint.target = target_empty
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    print(f"    - Camera {cam_location} created, tracking {target_location}.")
    print(f"    - Camera FOV: {math.degrees(cam_params['fov']):.2f}°")

    env_params = scene_params.get('environment')
    if env_params:
        env_color = env_params.get('color', (1.0, 1.0, 1.0))
        env_strength = env_params.get('strength', 5.0)
        setup_environment_lighting(env_color, env_strength)
        print(f"    - Environment: color={env_color}, strength={env_strength}")
    else:
        print("    - Warning: No environment lighting found in params, using defaults")
        setup_environment_lighting((1.0, 1.0, 1.0), 5.0)

    bpy.context.scene.render.film_transparent = True
    
    render_settings = {
        'resolution_x': cam_params.get('resolution_x', 480),
        'resolution_y': cam_params.get('resolution_y', 480)
    }
    
    print(f"    - Render at: {render_settings['resolution_x']}x{render_settings['resolution_y']}")
    
    return {
        'camera': camera_obj,
        'render_settings': render_settings
    }

def move_vertices_with_trajectory(mesh_obj, frame, trajectories):
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='OBJECT')
    if mesh_obj.data.shape_keys is None:
        basis = mesh_obj.shape_key_add(name='Basis')
        basis.interpolation = 'KEY_LINEAR'  
    shape_key_name = f"Frame_{frame}"
    shape_key = mesh_obj.data.shape_keys.key_blocks.get(shape_key_name)
    if not shape_key:
        shape_key = mesh_obj.shape_key_add(name=shape_key_name)
        shape_key.interpolation = 'KEY_LINEAR'
    positions = trajectories[frame].numpy()
    for idx, pos in enumerate(positions):
        shape_key.data[idx].co = Vector(pos)


def drive_mesh_and_render_with_pkl(mesh_objects, trajs, output_base_path, azi=0.0, ele=0.0, 
                                   export_format=None, camera_params_path=None,
                                   env_color=(1.0, 1.0, 1.0), env_strength=2.0,
                                   just_export=False):
   
    video_dir = os.path.dirname(output_base_path)
    filename = os.path.basename(output_base_path)
    os.makedirs(video_dir, exist_ok=True)

    print("Setting up animations for all meshes...")
    num_frames = 0
    
    for i in range(len(mesh_objects)):
        mesh_obj = mesh_objects[i]
        traj = trajs[i]
        num_frames = traj.shape[0]
        for frame in range(num_frames):
            move_vertices_with_trajectory(mesh_obj, frame, traj)
        
        for frame in range(num_frames):
            for shape_key in mesh_obj.data.shape_keys.key_blocks[1:]:
                shape_key.value = 0
                shape_key.keyframe_insert("value", frame=frame)
                if mesh_obj.data.shape_keys.animation_data and mesh_obj.data.shape_keys.animation_data.action:
                    fcurves = mesh_obj.data.shape_keys.animation_data.action.fcurves
                    for fc in fcurves:
                        for kf in fc.keyframe_points:
                            kf.interpolation = 'CONSTANT'
            
            current_shape_key = mesh_obj.data.shape_keys.key_blocks[f"Frame_{frame}"]
            current_shape_key.value = 1
            current_shape_key.keyframe_insert("value", frame=frame)

    print("Setting up scene...")
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = num_frames - 1
    
    use_params_file = camera_params_path and os.path.exists(camera_params_path)

    if use_params_file:
        print(f"Using '{camera_params_path}' to set up scene.")
        
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'CAMERA' or "ModelCenterTarget" in obj.name or "Recreated" in obj.name:
                obj.select_set(True)
        bpy.ops.object.delete()
        
        setup_result = setup_scene_from_params(camera_params_path)
        active_camera = setup_result['camera']
        render_settings = setup_result['render_settings']
        
        video_path = os.path.join(video_dir, f'{filename}.mp4')
        
        scene.render.resolution_x = render_settings.get('resolution_x', 480)
        scene.render.resolution_y = render_settings.get('resolution_y', 480)
        
    else:
        print(f"No camera params file found, setting up camera and lighting automatically.")
        
        setup_result = setup_camera_and_lighting(
            start_frame=0, 
            end_frame=num_frames - 1, 
            env_color=env_color, 
            env_strength=env_strength
        )
        
        if setup_result is None:
            raise RuntimeError("Camera setup failed.")
        
        active_camera = bpy.context.scene.camera
        video_path = os.path.join(video_dir, f'{filename}_auto_cam.mp4')
        
        scene.render.resolution_x = 480
        scene.render.resolution_y = 480
    
    scene.camera = active_camera
    scene.render.film_transparent = True
    
    print(f"    - Scene camera: {active_camera.name}")
    print(f"    - Animation frame range: {scene.frame_start} - {scene.frame_end}")

    if export_format is not None:

        print(f"Exporting animated mesh to {export_format.upper()} format...")
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objects:
            obj.select_set(True)
        
        export_path = os.path.join(video_dir, f"{filename}.{export_format}")
        if export_format == "abc":
            bpy.ops.wm.alembic_export(
                filepath=export_path, 
                start=scene.frame_start, 
                end=scene.frame_end, 
                selected=True, 
                uvs=True, 
                face_sets=True
            )
        elif export_format == "fbx":
            bpy.ops.export_scene.fbx(
                filepath=export_path, 
                use_selection=True, 
                bake_anim=True, 
                bake_anim_use_all_bones=False, 
                bake_anim_use_nla_strips=False, 
                bake_anim_use_all_actions=False, 
                bake_anim_force_startend_keying=True, 
                add_leaf_bones=False, 
                path_mode='COPY', 
                embed_textures=True, 
                use_mesh_modifiers=True, 
                use_mesh_edges=True, 
                use_tspace=True, 
                use_custom_props=True, 
                use_active_collection=False
            )
        elif export_format == "glb":
            print("Exporting GLB format...")
            
            original_selection = [obj for obj in bpy.context.selected_objects]
            original_active = bpy.context.view_layer.objects.active
            
            bpy.ops.object.select_all(action='DESELECT')
            for obj in mesh_objects:
                obj.select_set(True)
            
            bpy.context.view_layer.objects.active = mesh_objects[0]
            
            bpy.ops.object.duplicate()
            bpy.ops.object.join()
            
            merged_obj = bpy.context.active_object
            merged_obj.name = f"{filename}_merged"
            
            if merged_obj.data.shape_keys and merged_obj.data.shape_keys.animation_data:
                print(f"Merged object has {len(merged_obj.data.shape_keys.key_blocks)} Shape Keys")
            
            bpy.ops.object.select_all(action='DESELECT')
            merged_obj.select_set(True)
            bpy.context.view_layer.objects.active = merged_obj
            
            bpy.ops.export_scene.gltf(
                filepath=export_path,
                export_format='GLB',
                use_selection=True,
                export_apply=True,
                export_animations=True,
                export_morph=True,
                export_morph_normal=True,
                export_morph_tangent=False,
                export_skins=False,
                export_materials='EXPORT',
                export_image_format='AUTO',
                export_frame_range=True,
                export_frame_step=1,
                export_force_sampling=True, 
                export_current_frame=False
            )
            
            bpy.ops.object.select_all(action='DESELECT')
            merged_obj.select_set(True)
            bpy.ops.object.delete()
            
            bpy.ops.object.select_all(action='DESELECT')
            for obj in original_selection:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = original_active
            
            print(f"Successfully exported animated mesh to: {export_path}")
        
        bpy.ops.object.select_all(action='DESELECT')

    if just_export:
        print("just_export=True, skipping rendering process.")
        return

    print("Setting up direct video rendering...")
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 64
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transmission_bounces = 1
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 1
    
    scene.render.filepath = video_path
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scene.render.resolution_percentage = 100
    scene.render.fps = 15

    print(f"Rendering animation directly to video: {video_path}")
    bpy.ops.render.render(animation=True)
    
    print(f"Successfully rendered video to: {video_path}")

    # except Exception as e:
    #     print(f"An error occurred: {str(e)}")
    #     import traceback
    #     traceback.print_exc()
    #     raise

def drive_mesh_and_render_with_pkl_frames(mesh_objects, trajs, output_base_path, azi=0.0, ele=0.0, 
                                   export_format="none", camera_params_path=None,
                                   env_color=(1.0, 1.0, 1.0), env_strength=2.0,
                                   just_export=False):
   
    video_dir = os.path.dirname(output_base_path)
    filename = os.path.basename(output_base_path)
    os.makedirs(video_dir, exist_ok=True)

    print("Setting up animations for all meshes...")
    num_frames = 0
    
    for i in range(len(mesh_objects)):
        mesh_obj = mesh_objects[i]
        traj = trajs[i]
        num_frames = traj.shape[0]
        for frame in range(num_frames):
            move_vertices_with_trajectory(mesh_obj, frame, traj)
        
        for frame in range(num_frames):
            for shape_key in mesh_obj.data.shape_keys.key_blocks[1:]:
                shape_key.value = 0
                shape_key.keyframe_insert("value", frame=frame)
                if mesh_obj.data.shape_keys.animation_data and mesh_obj.data.shape_keys.animation_data.action:
                    fcurves = mesh_obj.data.shape_keys.animation_data.action.fcurves
                    for fc in fcurves:
                        for kf in fc.keyframe_points:
                            kf.interpolation = 'CONSTANT'
            
            current_shape_key = mesh_obj.data.shape_keys.key_blocks[f"Frame_{frame}"]
            current_shape_key.value = 1
            current_shape_key.keyframe_insert("value", frame=frame)

    print("Setting up scene...")
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = num_frames - 1
    
    use_params_file = camera_params_path and os.path.exists(camera_params_path)
    
    if use_params_file:
        print(f"Using '{camera_params_path}' to set up scene (ignoring azi/ele parameters).")
        
        # Clean up old objects
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type == 'CAMERA' or "ModelCenterTarget" in obj.name or "Recreated" in obj.name:
                obj.select_set(True)
        bpy.ops.object.delete()
        
        setup_result = setup_scene_from_params(camera_params_path)
        active_camera = setup_result['camera']
        render_settings = setup_result['render_settings']
        
        scene.render.resolution_x = render_settings.get('resolution_x', 480)
        scene.render.resolution_y = render_settings.get('resolution_y', 480)
        
    else:
        
        print(f"No parameter file provided, using dynamic camera setup (Azi={azi}, Ele={ele}).")
        
        # Clean up old cameras and lights (to prevent overlap)
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if obj.type in {'CAMERA', 'LIGHT'} or "ModelCenterTarget" in obj.name or "AutoCamera" in obj.name:
                obj.select_set(True)
        if bpy.context.selected_objects:
            bpy.ops.object.delete()

        cameras = setup_scene_with_dynamic_camera(
            mesh_obj=mesh_objects[0], 
            trajs=trajs, 
            azi=azi, 
            ele=ele, 
            env_color=env_color, 
            env_strength=env_strength
        )
        
        if not cameras:
            raise RuntimeError("Camera setup failed.")
        
        active_camera = cameras[0]
        
        scene.render.resolution_x = 512
        scene.render.resolution_y = 512
    
    scene.camera = active_camera
    scene.render.film_transparent = True
    
    print(f"    - Scene camera set: {active_camera.name}")
    print(f"    - Animation frame range: {scene.frame_start} - {scene.frame_end}")

    if export_format != "none":
        print(f"Exporting animated mesh to {export_format.upper()} format...")
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objects:
            obj.select_set(True)
        
        export_path = os.path.join(video_dir, f"{filename}.{export_format}")
        if export_format == "abc":
            bpy.ops.wm.alembic_export(filepath=export_path, start=scene.frame_start, end=scene.frame_end, selected=True, uvs=True, face_sets=True)
        elif export_format == "fbx":
            bpy.ops.export_scene.fbx(filepath=export_path, use_selection=True, bake_anim=True, bake_anim_use_all_bones=False, bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False, bake_anim_force_startend_keying=True, add_leaf_bones=False, path_mode='COPY', embed_textures=True, use_mesh_modifiers=True, use_mesh_edges=True, use_tspace=True, use_custom_props=True, use_active_collection=False)
        
        print(f"Successfully exported animated mesh to: {export_path}")
        bpy.ops.object.select_all(action='DESELECT')

    if just_export:
        print("just_export=True, skipping rendering process.")
        return

    frames_output_dir = os.path.join(video_dir, filename+"_azi{}".format(azi))
    os.makedirs(frames_output_dir, exist_ok=True)
    
    print(f"Setting up RGBA image sequence rendering to: {frames_output_dir}")
    
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 64
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transmission_bounces = 1
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 1
    
    scene.render.filepath = os.path.join(frames_output_dir, "frame_")
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.compression = 15
    scene.render.resolution_percentage = 100
    scene.render.fps = 15

    print(f"Rendering animation frames to: {frames_output_dir}")
    bpy.ops.render.render(animation=True)
    
    print(f"Successfully rendered frames to: {frames_output_dir}")

def setup_environment_lighting(env_color=(1.0, 1.0, 1.0), env_strength=2.0):

    world = bpy.data.worlds.get('World')
    if world is None:
        world = bpy.data.worlds.new(name='World')
    bpy.context.scene.world = world
    
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get('Background')
    if not bg_node:
        world.node_tree.nodes.clear()
        output_node = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
        bg_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
        world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

    bg_node.inputs['Color'].default_value = (*env_color, 1.0)
    bg_node.inputs['Strength'].default_value = env_strength
    
    return {
        'color': tuple(env_color),
        'strength': env_strength
    }

def setup_camera_and_lighting(start_frame, end_frame, env_color=(1.0, 1.0, 1.0), env_strength=2.0):

    print(f"Auto camera setup (Frame {start_frame} - {end_frame})...")
    
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type in {'CAMERA', 'LIGHT'} or "ModelCenterTarget" in obj.name or "Recreated" in obj.name:
            obj.select_set(True)
    if bpy.context.selected_objects:
        bpy.ops.object.delete()

    env_params = setup_environment_lighting(env_color, env_strength)

    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH' and obj.visible_get()]
    if not mesh_objects:
        print("Warning: No visible mesh objects found in the scene.")
        return None

    grand_min = Vector((math.inf, math.inf, math.inf))
    grand_max = Vector((-math.inf, -math.inf, -math.inf))
    
    original_frame = bpy.context.scene.frame_current
    start_frame, end_frame = int(start_frame), int(end_frame)
    sample_interval = max(1, (end_frame - start_frame + 1) // 20)
    
    print("  - Computing dynamic bounding box...")
    for frame in range(start_frame, end_frame + 1, sample_interval):
        bpy.context.scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        
        for obj in mesh_objects:
            eval_obj = obj.evaluated_get(depsgraph)
            if eval_obj.type != 'MESH' or not eval_obj.data.vertices:
                continue
            for corner in [eval_obj.matrix_world @ Vector(c) for c in eval_obj.bound_box]:
                grand_min.x = min(grand_min.x, corner.x)
                grand_min.y = min(grand_min.y, corner.y)
                grand_min.z = min(grand_min.z, corner.z)
                grand_max.x = max(grand_max.x, corner.x)
                grand_max.y = max(grand_max.y, corner.y)
                grand_max.z = max(grand_max.z, corner.z)
    
    bpy.context.scene.frame_set(original_frame)

    if not all(math.isfinite(c) for c in grand_min) or not all(math.isfinite(c) for c in grand_max):
        print("Warning: Invalid bounding box, using default values")
        center = Vector((0,0,0))
        size = 1.0
    else:
        center = (grand_min + grand_max) / 2.0
        size = max((grand_max - grand_min).to_tuple())
    
    if size == 0: size = 1.0
    
    print(f"  - Scene center: {center}")
    print(f"  - Scene size: {size}")

    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera_obj = bpy.context.object
    camera_obj.name = "AutoCamera"
    bpy.context.scene.camera = camera_obj

    bpy.ops.object.empty_add(type='PLAIN_AXES', location=center)
    target_empty = bpy.context.object
    target_empty.name = "ModelCenterTarget"
    
    constraint = camera_obj.constraints.new(type='TRACK_TO')
    constraint.target = target_empty
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    camera = camera_obj.data
    fov = camera.angle
    distance = (size / 2.0) / math.tan(fov / 2.0) if math.tan(fov / 2.0) > 0 else size
    
    camera_obj.location = (center.x, center.y - (distance * 1.5), center.z + (size * 0.2))

    camera.clip_start = 0.01
    camera.clip_end = max(size * 10, 10000.0)

    return {
        "camera": {
            'location': tuple(camera_obj.location),
            'target': tuple(center),
            'fov': camera.angle,
            'clip_start': camera.clip_start,
            'clip_end': camera.clip_end,
            'resolution_x': bpy.context.scene.render.resolution_x,
            'resolution_y': bpy.context.scene.render.resolution_y
        },
        "environment": env_params
    }