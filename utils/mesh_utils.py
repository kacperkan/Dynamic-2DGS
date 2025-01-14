#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import copy
import math
import os
from functools import partial

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
from tqdm import tqdm

from gaussian_renderer import render
from utils.render_utils import save_img_f32, save_img_u8


def post_process_mesh(mesh: trimesh.Trimesh, num_triangles_to_keep: int = 100):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """

    print(
        "Removing all the objects with the number of triangles smaller than {}".format(
            num_triangles_to_keep
        )
    )
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh_0.cluster_connected_triangles()
        )

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = (
        cluster_n_triangles[triangle_clusters] < num_triangles_to_keep
    )
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx=viewpoint_cam.image_width / 2,
            cy=viewpoint_cam.image_height / 2,
            fx=viewpoint_cam.image_width
            / (2 * math.tan(viewpoint_cam.FoVx / 2.0)),
            fy=viewpoint_cam.image_height
            / (2 * math.tan(viewpoint_cam.FoVy / 2.0)),
        )

        extrinsic = np.asarray(
            (viewpoint_cam.world_view_transform.T).cpu().numpy()
        )
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(
        ...     gaussians,
        ...     render,
        ...     pipe,
        ... )
        >>> gaussExtrator.reconstruction(
        ...     view_points
        ... )
        >>> mesh = gaussExtractor.export_mesh_bounded(
        ...     ...
        ... )
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.alphamaps = []
        self.rgbmaps = []
        self.normals = []
        self.depth_normals = []
        self.points = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(
        self,
        viewpoint_stack,
        pipeline,
        background,
        deform,
        state,
        depth_filtering,
    ):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(
            enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"
        ):
            # render_pkg = self.render(viewpoint_cam, self.gaussians)
            # if load2gpt_on_the_fly:
            #    viewpoint_cam.load2device()
            fid = viewpoint_cam.fid
            xyz = self.gaussians.get_xyz

            """
            if deform.name == 'mlp':
                time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
            elif deform.name == 'node':
                time_input = deform.deform.expand_time(fid)
            """
            if state == "mesh":
                if deform.name == "mlp":
                    time_input2 = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                elif deform.name == "node":
                    time_input2 = deform.deform.expand_time(fid)
                time_input = (
                    torch.tensor(viewpoint_cam.time)
                    .to(xyz.device)
                    .repeat(time_input2.shape[0], 1)
                )
            else:
                if deform.name == "mlp":
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                elif deform.name == "node":
                    time_input = deform.deform.expand_time(fid)

            d_values = deform.step(
                xyz.detach(),
                time_input,
                feature=self.gaussians.feature,
                motion_mask=self.gaussians.motion_mask,
            )
            d_xyz, d_rotation, d_scaling, d_opacity, d_color = (
                d_values["d_xyz"],
                d_values["d_rotation"],
                d_values["d_scaling"],
                d_values["d_opacity"],
                d_values["d_color"],
            )
            results = render(
                viewpoint_cam,
                self.gaussians,
                pipeline,
                background,
                d_xyz,
                d_rotation,
                d_scaling,
                d_opacity=d_opacity,
                d_color=d_color,
                d_rot_as_res=deform.d_rot_as_res,
                depth_filtering=depth_filtering,
            )
            alpha = results["alpha"]
            rendering = torch.clamp(
                torch.cat([results["render"], alpha]), 0.0, 1.0
            )

            rgb = rendering[:3]
            # mask = (1-(torch.all(rgb == 0, dim=0)).to(torch.int))
            # alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize(
                results["rend_normal"], dim=0
            )
            # normal = normal*mask
            depth = results["depth"]
            # depth = depth*mask
            # depth = render_pkg['surf_depth']
            depth_normal = results["surf_normal"]
            # depth_normal = depth_normal*mask
            # point = render_pkg['surf_point']
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            self.alphamaps.append(alpha.cpu())
            self.normals.append(normal.cpu())
            self.depth_normals.append(depth_normal.cpu())
            # self.points.append(point.cpu())

        self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        self.depthmaps = torch.stack(self.depthmaps, dim=0)
        self.alphamaps = torch.stack(self.alphamaps, dim=0)
        self.depth_normals = torch.stack(self.depth_normals, dim=0)
        # self.points = torch.stack(self.points, dim=0)

    @torch.no_grad()
    def extract_mesh_bounded(
        self,
        voxel_size=0.004,
        sdf_trunc=0.02,
        depth_trunc=3,
        mask_backgrond=True,
    ):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.

        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f"voxel_size: {voxel_size}")
        print(f"sdf_trunc: {sdf_trunc}")
        print(f"depth_truc: {depth_trunc}")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        for i, cam_o3d in tqdm(
            enumerate(to_cam_open3d(self.viewpoint_stack)),
            desc="TSDF integration progress",
        ):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]

            # if we have mask provided, use it
            if mask_backgrond and (
                self.viewpoint_stack[i].gt_alpha_mask is not None
            ):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(
                    np.asarray(
                        rgb.permute(1, 2, 0).cpu().numpy() * 255,
                        order="C",
                        dtype=np.uint8,
                    )
                ),
                o3d.geometry.Image(
                    np.asarray(depth.permute(1, 2, 0).cpu().numpy(), order="C")
                ),
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
                depth_scale=1.0,
            )

            volume.integrate(
                rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic
            )

        mesh = volume.extract_triangle_mesh()
        return mesh

    # @torch.no_grad()
    # def extract_mesh_unbounded(self, resolution=1024):
    #     """
    #     Experimental features, extracting meshes from unbounded scenes, not fully test across datasets.
    #     #TODO: support color mesh exporting

    #     sdf_trunc: truncation value
    #     return o3d.mesh
    #     """

    #     def contract(x):
    #         mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
    #         return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))

    #     def uncontract(y):
    #         mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
    #         return torch.where(mag < 1, y, (1 / (2 - mag) * (y / mag)))

    #     def compute_sdf_perframe(
    #         i, points, depthmap, rgbmap, normalmap, viewpoint_cam
    #     ):
    #         """
    #         compute per frame sdf
    #         """
    #         new_points = (
    #             torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    #             @ viewpoint_cam.full_proj_transform
    #         )
    #         z = new_points[..., -1:]
    #         pix_coords = new_points[..., :2] / new_points[..., -1:]
    #         mask_proj = (
    #             (pix_coords > -1.0) & (pix_coords < 1.0) & (z > 0)
    #         ).all(dim=-1)
    #         sampled_depth = torch.nn.functional.grid_sample(
    #             depthmap.cuda()[None],
    #             pix_coords[None, None],
    #             mode="bilinear",
    #             padding_mode="border",
    #             align_corners=True,
    #         ).reshape(-1, 1)
    #         sampled_rgb = (
    #             torch.nn.functional.grid_sample(
    #                 rgbmap.cuda()[None],
    #                 pix_coords[None, None],
    #                 mode="bilinear",
    #                 padding_mode="border",
    #                 align_corners=True,
    #             )
    #             .reshape(3, -1)
    #             .T
    #         )
    #         sampled_normal = (
    #             torch.nn.functional.grid_sample(
    #                 normalmap.cuda()[None],
    #                 pix_coords[None, None],
    #                 mode="bilinear",
    #                 padding_mode="border",
    #                 align_corners=True,
    #             )
    #             .reshape(3, -1)
    #             .T
    #         )
    #         sdf = sampled_depth - z
    #         return sdf, sampled_rgb, sampled_normal, mask_proj

    #     def compute_unbounded_tsdf(
    #         samples, inv_contraction, voxel_size, return_rgb=False
    #     ):
    #         """
    #         Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
    #         """
    #         if inv_contraction is not None:
    #             samples = inv_contraction(samples)
    #             mask = torch.linalg.norm(samples, dim=-1) > 1
    #             # adaptive sdf_truncation
    #             sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
    #             sdf_trunc[mask] *= 1 / (
    #                 2 - torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9)
    #             )
    #         else:
    #             sdf_trunc = 5 * voxel_size

    #         tsdfs = torch.ones_like(samples[:, 0]) * 1
    #         rgbs = torch.zeros((samples.shape[0], 3)).cuda()

    #         weights = torch.ones_like(samples[:, 0])
    #         for i, viewpoint_cam in tqdm(
    #             enumerate(self.viewpoint_stack),
    #             desc="TSDF integration progress",
    #         ):
    #             sdf, rgb, normal, mask_proj = compute_sdf_perframe(
    #                 i,
    #                 samples,
    #                 depthmap=self.depthmaps[i],
    #                 rgbmap=self.rgbmaps[i],
    #                 normalmap=self.depth_normals[i],
    #                 viewpoint_cam=self.viewpoint_stack[i],
    #             )

    #             # volume integration
    #             sdf = sdf.flatten()
    #             mask_proj = mask_proj & (sdf > -sdf_trunc)
    #             sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[
    #                 mask_proj
    #             ]
    #             w = weights[mask_proj]
    #             wp = w + 1
    #             tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
    #             rgbs[mask_proj] = (
    #                 rgbs[mask_proj] * w[:, None] + rgb[mask_proj]
    #             ) / wp[:, None]
    #             # update weight
    #             weights[mask_proj] = wp

    #         if return_rgb:
    #             return tsdfs, rgbs

    #         return tsdfs

    #     from utils.render_utils import focus_point_fn

    #     torch.cuda.empty_cache()
    #     c2ws = np.array(
    #         [
    #             np.linalg.inv(
    #                 np.asarray((cam.world_view_transform.T).cpu().numpy())
    #             )
    #             for cam in self.viewpoint_stack
    #         ]
    #     )
    #     poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
    #     center = focus_point_fn(poses)
    #     radius = np.linalg.norm(c2ws[:, :3, 3] - center, axis=-1).min()
    #     center = torch.from_numpy(center).float().cuda()

    #     def normalize(x):
    #         return (x - center) / radius

    #     def unnormalize(x):
    #         return (x * radius) + center

    #     def inv_contraction(x):
    #         return unnormalize(uncontract(x))

    #     N = resolution
    #     voxel_size = radius * 2 / N
    #     print(f"Computing sdf gird resolution {N} x {N} x {N}")
    #     print(f"Define the voxel_size as {voxel_size}")

    #     def sdf_function(x):
    #         return compute_unbounded_tsdf(x, inv_contraction, voxel_size)

    #     R = (
    #         contract(normalize(self.gaussians.get_xyz))
    #         .norm(dim=-1)
    #         .cpu()
    #         .numpy()
    #     )
    #     R = np.quantile(R, q=0.95)
    #     R = min(R + 0.01, 1.9)

    #     from utils.mcube_utils import marching_cubes_with_contraction

    #     mesh = marching_cubes_with_contraction(
    #         sdf=sdf_function,
    #         bounding_box_min=(-R, -R, -R),
    #         bounding_box_max=(R, R, R),
    #         level=0,
    #         resolution=N,
    #         inv_contraction=inv_contraction,
    #     )

    #     # coloring the mesh
    #     torch.cuda.empty_cache()
    #     mesh = mesh.as_open3d
    #     print("texturing mesh ... ")
    #     _, rgbs = compute_unbounded_tsdf(
    #         torch.tensor(np.asarray(mesh.vertices)).float().cuda(),
    #         inv_contraction=None,
    #         voxel_size=voxel_size,
    #         return_rgb=True,
    #     )
    #     mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
    #     return mesh

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(
            enumerate(self.viewpoint_stack), desc="export images"
        ):
            gt = viewpoint_cam.original_image[0:3, :, :]
            save_img_u8(
                gt.permute(1, 2, 0).cpu().numpy(),
                os.path.join(gts_path, "{0:05d}".format(idx) + ".png"),
            )
            save_img_u8(
                self.rgbmaps[idx].permute(1, 2, 0).cpu().numpy(),
                os.path.join(render_path, "{0:05d}".format(idx) + ".png"),
            )
            save_img_f32(
                self.depthmaps[idx][0].cpu().numpy(),
                os.path.join(vis_path, "depth_{0:05d}".format(idx) + ".tiff"),
            )
            max_depth = np.max(self.depthmaps[idx][0].cpu().numpy())
            cv2.imwrite(
                os.path.join(vis_path, "depth_{0:05d}".format(idx) + ".png"),
                self.depthmaps[idx][0].cpu().numpy() / max_depth * 255,
            )
            # save_img_u8(self.depthmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".png"))
            save_img_u8(
                self.normals[idx].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5,
                os.path.join(vis_path, "normal_{0:05d}".format(idx) + ".png"),
            )
            save_img_u8(
                self.depth_normals[idx].permute(1, 2, 0).cpu().numpy() * 0.5
                + 0.5,
                os.path.join(
                    vis_path, "depth_normal_{0:05d}".format(idx) + ".png"
                ),
            )
