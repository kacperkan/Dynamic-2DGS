#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import (
    GaussianExtractor,
    to_cam_open3d,
    post_process_mesh,
)
from utils.render_utils import generate_path, create_videos
from utils.system_utils import load_config_from_file, merge_config
from utils.pose_utils import pose_spherical
import open3d as o3d
import numpy as np

# from utils.camera_utils import get_camera_trajectory_pose
from mesh_renderer import render_mesh, mesh_shape_renderer
import cv2
import copy

import json
import imageio
import os
from PIL import Image
from read_gt_mesh import load_obj


def clean_mesh(
    mesh,
    edge_threshold: float = 0.1,
    min_triangles_connected: int = -1,
    fill_holes: bool = True,
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Performs the following steps to clean the mesh:

    1. edge_threshold_filter
    2. remove_duplicated_vertices, remove_duplicated_triangles, remove_degenerate_triangles
    3. remove small connected components
    4. remove_unreferenced_vertices
    5. fill_holes

    :param vertices: (3, N) torch.Tensor of type torch.float32
    :param faces: (3, M) torch.Tensor of type torch.long
    :param colors: (3, N) torch.Tensor of type torch.float32 in range (0...1) giving RGB colors per vertex
    :param edge_threshold: maximum length per edge (otherwise removes that face). If <=0, will not do this filtering
    :param min_triangles_connected: minimum number of triangles in a connected component (otherwise removes those faces). If <=0, will not do this filtering
    :param fill_holes: If true, will perform trimesh fill_holes step, otherwise not.

    :return: (vertices, faces, colors) tuple as torch.Tensors of similar shape and type
    """
    """
    if edge_threshold > 0:
        # remove long edges
        faces = edge_threshold_filter(vertices, faces, edge_threshold)
    """

    # cleanup via open3d
    # mesh = torch_to_o3d_mesh(vertices, faces) #, colors)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    if min_triangles_connected > 0:
        # remove small components via open3d
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles()
        )
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        triangles_to_remove = (
            cluster_n_triangles[triangle_clusters] < min_triangles_connected
        )
        mesh.remove_triangles_by_mask(triangles_to_remove)

    # cleanup via open3d
    mesh.remove_unreferenced_vertices()

    if fill_holes:
        # misc cleanups via trimesh
        mesh = o3d_to_trimesh(mesh)
        mesh.process()
        mesh.fill_holes()
    return mesh


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    # hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument(
        "--voxel_size",
        default=0.004,
        type=float,
        help="Mesh: voxel size for TSDF",
    )
    parser.add_argument(
        "--depth_trunc",
        default=3.0,
        type=float,
        help="Mesh: Max depth range for TSDF",
    )
    parser.add_argument(
        "--num_cluster",
        default=1000,
        type=int,
        help="Mesh: number of connected clusters to export",
    )
    parser.add_argument(
        "--unbounded",
        action="store_true",
        help="Mesh: using unbounded mode for meshing",
    )
    parser.add_argument(
        "--mesh_res",
        default=1024,
        type=int,
        help="Mesh: resolution for unbounded mesh extraction",
    )
    parser.add_argument(
        "--white_background2",
        default=False,
        type=bool,
        help="Mesh: resolution for unbounded mesh extraction",
    )
    parser.add_argument("--config", type=str, default=None)
    args = get_combined_args(parser)
    # args = parser.parse_args(sys.argv[1:])
    args.depth_trunc = 6
    depth_filtering = True
    # model._white_background = args.depth_trunc
    print(model._white_background)
    # model._white_background =  args.white_background2
    # print(model._white_background)
    # args.voxel_size = 0.002
    # args.num_cluster = 860

    # Load config file
    """
    if args.config:
        config_data = load_config_from_file(args.config)
        combined_args = merge_config(config_data, args)
        args = Namespace(**combined_args)
    """
    print("Rendering " + args.model_path)

    dataset, iteration, pipe = (
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
    )
    # gaussians = GaussianModel(dataset.sh_degree,hp.extract(args))

    deform = DeformModel(
        K=dataset.K,
        deform_type=dataset.deform_type,
        is_blender=dataset.is_blender,
        skinning=dataset.skinning,
        hyper_dim=dataset.hyper_dim,
        node_num=dataset.node_num,
        pred_opacity=dataset.pred_opacity,
        pred_color=dataset.pred_color,
        use_hash=dataset.use_hash,
        hash_time=dataset.hash_time,
        d_rot_as_res=dataset.d_rot_as_res,
        local_frame=dataset.local_frame,
        progressive_brand_time=dataset.progressive_brand_time,
        max_d_scale=dataset.max_d_scale,
    )
    deform.load_weights(dataset.model_path, iteration=iteration)

    gs_fea_dim = (
        deform.deform.node_num
        if dataset.skinning and deform.name == "node"
        else dataset.hyper_dim
    )
    gaussians = GaussianModel(
        dataset.sh_degree,
        fea_dim=gs_fea_dim,
        with_motion_mask=dataset.gs_with_motion_mask,
    )

    # gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_dir = os.path.join(
        args.model_path, "train", "ours_{}".format(scene.loaded_iter)
    )
    test_dir = os.path.join(
        args.model_path, "test", "ours_{}".format(scene.loaded_iter)
    )
    gaussExtractor = GaussianExtractor(
        gaussians, render, pipe, bg_color=bg_color
    )

    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(
            scene.getTrainCameras(),
            pipeline,
            background,
            deform,
            state="train",
            depth_filtering=depth_filtering,
        )
        gaussExtractor.export_image(train_dir)

    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(
            scene.getTestCameras(),
            pipeline,
            background,
            deform,
            state="test",
            depth_filtering=depth_filtering,
        )
        gaussExtractor.export_image(test_dir)

    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(
            args.model_path, "traj", "ours_{}".format(scene.loaded_iter)
        )
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 240
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        gaussExtractor.reconstruction(
            cam_traj,
            pipeline,
            background,
            deform,
            state="video",
            depth_filtering=depth_filtering,
        )
        gaussExtractor.export_image(traj_dir)
        create_videos(
            base_dir=traj_dir,
            input_dir=traj_dir,
            out_name="render_traj",
            num_frames=n_fames,
        )

    file_path = os.path.join(args.source_path, "transforms_test.json")

    # 假设 JSON 文件名为 'data.json'
    with open(file_path, "r") as file:
        data = json.load(file)

    images_save_path = os.path.join(args.model_path, "mesh_image")
    meshshape_save_path = os.path.join(args.model_path, "mesh_shape")
    meshshape_gt_save_path = os.path.join(args.model_path, "mesh_shape_gt")

    if not os.path.exists(images_save_path):
        os.mkdir(images_save_path)

    if not os.path.exists(meshshape_save_path):
        os.mkdir(meshshape_save_path)

    if not os.path.exists(meshshape_gt_save_path):
        os.mkdir(meshshape_gt_save_path)

    frame = 100
    render_poses = torch.stack(
        [
            pose_spherical(angle, -15.0, 4.0)
            for angle in np.linspace(-180, 180, frame + 1)[:-1]
        ],
        0,
    )
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    if not args.skip_mesh:
        renderings = []
        for i in range(len(render_poses)):
            # if i %10 ==0:
            mesh_time = i / len(render_poses)
            print("export mesh ...")
            os.makedirs(train_dir, exist_ok=True)
            # set the active_sh to 0 to export only diffuse texture
            gaussExtractor.gaussians.active_sh_degree = 0
            gaussExtractor.reconstruction(
                scene.getTrainCameras_mesh(mesh_time=mesh_time),
                pipeline,
                background,
                deform,
                state="mesh",
                depth_filtering=depth_filtering,
            )
            # extract the mesh and save
            if args.unbounded:
                # name = f'fuse_unbounded_{mesh_time}.ply'
                name = f"frame_{i}.ply"
                mesh = gaussExtractor.extract_mesh_unbounded(
                    resolution=args.mesh_res
                )
            else:
                name = f"frame_{i}.ply"
                mesh = gaussExtractor.extract_mesh_bounded(
                    voxel_size=args.voxel_size,
                    sdf_trunc=5 * args.voxel_size,
                    depth_trunc=args.depth_trunc,
                )

            # o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
            # print("mesh saved at {}".format(os.path.join(train_dir, name)))
            # post-process the mesh and save, saving the largest N clusters
            mesh_post = post_process_mesh(
                mesh, cluster_to_keep=args.num_cluster
            )
            # mesh_post = mesh_post.fill_holes()

            # o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh_post)
            print(
                "mesh post processed saved at {}".format(
                    os.path.join(train_dir, name)
                )
            )

            verts = (
                torch.from_numpy(np.asarray(mesh_post.vertices))
                .unsqueeze(0)
                .to(torch.float32)
                .cuda()
            )
            faces = (
                torch.from_numpy(np.asarray(mesh_post.triangles))
                .to(torch.int32)
                .cuda()
            )
            vertex_colors = (
                torch.from_numpy(np.asarray(mesh_post.vertex_colors))
                .unsqueeze(0)
                .to(torch.float32)
                .cuda()
            )

            cam = scene.getTestCameras()

            viewpoint_cam = copy.deepcopy(cam[0])
            pose = render_poses[i]

            matrix = np.linalg.inv(np.array(pose))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            viewpoint_cam.R = R
            viewpoint_cam.T = T
            viewpoint_cam = viewpoint_cam.cpu()

            rets = render_mesh(
                viewpoint_cam,
                verts,
                faces,
                vertex_colors,
                whitebackground=True,
            )
            mesh_image = rets["render"]
            mesh_img = mesh_image.cpu().detach().numpy()
            mesh_img = mesh_img * 255
            imagename = str(i).zfill(5)
            print("save images")
            cv2.imwrite(images_save_path + f"/{imagename}.png", mesh_img)

            mesh_image_shape = mesh_shape_renderer(verts, faces, viewpoint_cam)
            mesh_image_shape_np = mesh_image_shape.detach().cpu().numpy() * 255
            imagename = str(i).zfill(5)
            print("save images")
            cv2.imwrite(
                meshshape_save_path + f"/{imagename}.png", mesh_image_shape_np
            )
            renderings.append(to8b(mesh_image_shape_np))
            print(mesh_image_shape_np.shape)

    images = []
    for k in range(len(os.listdir(meshshape_save_path))):
        images.append(
            imageio.imread(
                meshshape_save_path + "/" + str(k).zfill(5) + ".png"
            )
        )

    video_name = "output_video.mp4"
    fps = 25

    with imageio.get_writer(video_name, fps=fps) as video_writer:
        for image in images:
            video_writer.append_data(image)
