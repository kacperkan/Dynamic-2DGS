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

import math
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import numpy as np
import torch
import torch.nn.functional
import tqdm
from typing_extensions import Literal

from arguments import ModelParams, OptimizationParams, PipelineParams
from cam_utils import OrbitCamera
from gaussian_renderer import render
from scene import GaussianModel, Scene
from train import training_report
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def landmark_interpolate(landmarks, steps, step, interpolation="log"):
    stage = (step >= np.array(steps)).sum()
    if stage == len(steps):
        return max(0, landmarks[-1])
    elif stage == 0:
        return 0
    else:
        ldm1, ldm2 = landmarks[stage - 1], landmarks[stage]
        if ldm2 <= 0:
            return 0
        step1, step2 = steps[stage - 1], steps[stage]
        ratio = (step - step1) / (step2 - step1)
        if interpolation == "log":
            return np.exp(np.log(ldm1) * (1 - ratio) + np.log(ldm2) * ratio)
        elif interpolation == "linear":
            return ldm1 * (1 - ratio) + ldm2 * ratio
        else:
            print(f"Unknown interpolation type: {interpolation}")
            raise NotImplementedError


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, fid):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.fid = fid
        self.c2w = c2w

        w2c = np.linalg.inv(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = (
            torch.tensor(w2c).transpose(0, 1).cuda().float()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear,
                zfar=self.zfar,
                fovX=self.FoVx,
                fovY=self.FoVy,
            )
            .transpose(0, 1)
            .cuda()
            .float()
        )
        self.full_proj_transform = (
            self.world_view_transform @ self.projection_matrix
        )
        self.camera_center = -torch.tensor(c2w[:3, 3]).cuda()

    def reset_extrinsic(self, R, T):
        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class GUI:
    def __init__(
        self,
        args,
        dataset: ModelParams,
        opt: OptimizationParams,
        pipe: PipelineParams,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
    ) -> None:
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations

        self.checkpoint_iterations = checkpoint_iterations
        self.checkpoint = checkpoint

        self.tb_writer = prepare_output_and_logger(dataset)

        self.gaussians = GaussianModel(dataset.sh_degree, args=dataset)

        self.scene = Scene(dataset, self.gaussians, load_iteration=-1)
        self.gaussians.training_setup(opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(
            bg_color, dtype=torch.float32, device="cuda"
        )

        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = (
            1 if self.scene.loaded_iter is None else self.scene.loaded_iter
        )

        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_ms_ssim = 0.0
        self.best_lpips = np.inf
        self.best_alex_lpips = np.inf
        self.best_iteration = 0
        self.progress_bar = tqdm.tqdm(
            range(opt.iterations), desc="Training progress"
        )
        # For UI
        self.visualization_mode = "RGB"

        # self.gui = args.gui # enable gui
        self.W = args.W
        self.H = args.H
        self.cam = OrbitCamera(args.W, args.H, r=args.radius, fovy=args.fovy)
        self.mode = "render"
        self.seed = "random"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.training = False

    def _train(self, stage: Literal["coarse", "fine"]):
        steps = (
            self.opt.coarse_iterations
            if stage == "coarse"
            else self.opt.iterations
        )
        for _ in tqdm.trange(steps):
            self.train_step(stage)

    # no gui mode
    def train(self):
        self._train("coarse")
        self._train("fine")

    def train_step(self, stage: Literal["coarse", "fine"]):
        self.iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iteration % self.opt.oneupSHdegree_step == 0:
            self.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self.viewpoint_stack:
            viewpoint_stack = self.scene.getTrainCameras().copy()
            self.viewpoint_stack = viewpoint_stack

        viewpoint_cam = self.viewpoint_stack.pop(
            randint(0, len(self.viewpoint_stack) - 1)
        )
        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        # Render
        random_bg_color = (
            (not self.dataset.white_background and self.opt.random_bg_color)
            and self.opt.gt_alpha_mask_as_scene_mask
            and viewpoint_cam.gt_alpha_mask is not None
        )
        render_pkg_re = render(
            viewpoint_cam,
            self.gaussians,
            self.pipe,
            self.background,
            random_bg_color=random_bg_color,
        )
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg_re["render"],
            render_pkg_re["viewspace_points"],
            render_pkg_re["visibility_filter"],
            render_pkg_re["radii"],
        )

        lambda_normal = 0.02 if self.iteration > 8000 else 0.0
        lambda_dist = 1000 if self.iteration > 8000 else 0.0
        rend_dist = render_pkg_re["rend_dist"]
        rend_normal = render_pkg_re["rend_normal"]
        surf_normal = render_pkg_re["surf_normal"]

        if stage == "fine":
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()
        else:
            normal_error = 0.0
            normal_loss = 0.0
            dist_loss = 0.0

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        if (
            self.dataset.white_background
            and viewpoint_cam.gt_alpha_mask is not None
            and self.opt.gt_alpha_mask_as_scene_mask
        ):
            gt_alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
            mask_loss = torch.nn.functional.binary_cross_entropy(
                render_pkg_re["alpha"], gt_alpha_mask
            )
        else:
            mask_loss = 0.0

        Ll1 = l1_loss(image, gt_image)
        loss_img = (
            1.0 - self.opt.lambda_dssim
        ) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = loss_img + normal_loss + dist_loss + 0.001 * mask_loss
        if stage == "fine" and self.dataset.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = self.gaussians.compute_regulation(
                self.dataset.time_smoothness_weight,
                self.dataset.l1_time_planes,
                self.dataset.plane_tv_weight,
            )
            loss += tv_loss

        loss.backward()
        viewspace_point_tensor_grad = viewspace_point_tensor

        self.iter_end.record()

        if self.dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device("cpu")

        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = (
                0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            )
            if self.iteration % 10 == 0:
                self.progress_bar.set_postfix(
                    {"Loss": f"{self.ema_loss_for_log:.{7}f}"}
                )
                self.progress_bar.update(10)
            if self.iteration == self.opt.iterations:
                self.progress_bar.close()

            # Keep track of max radii in image-space for pruning
            if self.gaussians.max_radii2D.shape[0] == 0:
                self.gaussians.max_radii2D = torch.zeros_like(radii)
            self.gaussians.max_radii2D[visibility_filter] = torch.max(
                self.gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter],
            )

            # Log and save
            cur_psnr, cur_ssim, cur_lpips, cur_ms_ssim, cur_alex_lpips = (
                training_report(
                    self.tb_writer,
                    self.iteration,
                    Ll1,
                    loss,
                    l1_loss,
                    self.iter_start.elapsed_time(self.iter_end),
                    self.testing_iterations,
                    self.scene,
                    render,
                    (self.pipe, self.background),
                    self.dataset.load2gpu_on_the_fly,
                    progress_bar=self.progress_bar,
                    loss_dict={
                        "normal": normal_loss,
                        "dist": dist_loss,
                        "mask": mask_loss,
                        "tv": tv_loss,
                    },
                )
            )
            if self.iteration in self.testing_iterations:
                if cur_psnr.item() > self.best_psnr:
                    self.best_psnr = cur_psnr.item()
                    self.best_iteration = self.iteration
                    self.best_ssim = cur_ssim.item()
                    self.best_ms_ssim = cur_ms_ssim.item()
                    self.best_lpips = cur_lpips.item()
                    self.best_alex_lpips = cur_alex_lpips.item()

            # Densification
            if self.iteration < self.opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor_grad, visibility_filter
                )

                if stage == "coarse":
                    opacity_threshold = self.opt.opacity_threshold_coarse
                    densify_threshold = self.opt.densify_grad_threshold_coarse
                else:
                    opacity_threshold = (
                        self.opt.opacity_threshold_fine_init
                        - self.iteration
                        * (
                            self.opt.opacity_threshold_fine_init
                            - self.opt.opacity_threshold_fine_after
                        )
                        / (self.opt.densify_until_iter)
                    )
                    densify_threshold = (
                        self.opt.densify_grad_threshold_fine_init
                        - self.iteration
                        * (
                            self.opt.densify_grad_threshold_fine_init
                            - self.opt.densify_grad_threshold_after
                        )
                        / (self.opt.densify_until_iter)
                    )
                if (
                    self.iteration > self.opt.densify_from_iter
                    and self.iteration % self.opt.densification_interval == 0
                    and self.gaussians.get_xyz.shape[0] < 360000
                ):
                    size_threshold = (
                        20
                        if self.iteration > self.opt.opacity_reset_interval
                        else None
                    )

                    self.gaussians.densify(
                        densify_threshold,
                        opacity_threshold,
                        self.scene.cameras_extent,
                        size_threshold,
                        5,
                        5,
                        self.scene.model_path,
                        self.iteration,
                        stage,
                    )
                if (
                    self.iteration > self.opt.pruning_from_iter
                    and self.iteration % self.opt.pruning_interval == 0
                    and self.gaussians.get_xyz.shape[0] > 200000
                ):
                    size_threshold = (
                        20
                        if self.iteration > self.opt.opacity_reset_interval
                        else None
                    )

                    self.gaussians.prune(
                        densify_threshold,
                        opacity_threshold,
                        self.scene.cameras_extent,
                        size_threshold,
                    )

                    # torch.cuda.empty_cache()
                if self.iteration % self.opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    self.gaussians.reset_opacity()

            # Optimizer step
            if self.iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.update_learning_rate(self.iteration)
                self.gaussians.optimizer.zero_grad(set_to_none=True)

            if self.iteration in self.checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(self.iteration))
                torch.save(
                    (self.gaussians.capture(), self.iteration),
                    self.scene.model_path
                    + "/chkpnt"
                    + f"_{stage}_"
                    + str(self.iteration)
                    + ".pth",
                )

        self.progress_bar.set_description(
            "Best PSNR={} in Iteration {}, SSIM={}, LPIPS={}, MS-SSIM={}, ALex-LPIPS={}".format(
                "%.5f" % self.best_psnr,
                self.best_iteration,
                "%.5f" % self.best_ssim,
                "%.5f" % self.best_lpips,
                "%.5f" % self.best_ms_ssim,
                "%.5f" % self.best_alex_lpips,
            )
        )
        self.iteration += 1
        del viewpoint_cam


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--W", type=int, default=800, help="GUI width")
    parser.add_argument("--H", type=int, default=800, help="GUI height")
    parser.add_argument(
        "--elevation",
        type=float,
        default=0,
        help="default GUI camera elevation",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=5,
        help="default GUI camera radius from center",
    )
    parser.add_argument(
        "--fovy", type=float, default=50, help="default GUI camera fovy"
    )

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=list(range(1000, 100_0001, 1000)),
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[7_000, 10_000, 20_000, 30_000, 40000],
    )

    parser.add_argument(
        "--checkpoint_iterations", nargs="+", type=int, default=[]
    )
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform-type", type=str, default="mlp")
    parser.add_argument(
        "--white_background2",
        default=False,
        type=bool,
        help="Mesh: resolution for unbounded mesh extraction",
    )
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if not args.model_path.endswith(args.deform_type):
        args.model_path = os.path.join(
            os.path.dirname(os.path.normpath(args.model_path)),
            os.path.basename(os.path.normpath(args.model_path))
            + f"_{args.deform_type}",
        )

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    gui = GUI(
        args=args,
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        checkpoint_iterations=args.checkpoint_iterations,
        checkpoint=args.start_checkpoint,
    )

    gui.train()

    # All done
    print("\nTraining complete.")
