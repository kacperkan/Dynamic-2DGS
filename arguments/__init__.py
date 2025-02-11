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

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            if key.startswith("_"):
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            # if shorthand:
            #     if t == bool:
            #         group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
            #     else:
            #         group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            # else:
            if t is bool:
                group.add_argument(
                    "--" + key, default=value, action="store_true"
                )
            else:
                group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.K = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.load2gpu_on_the_fly = False
        self.is_blender = False
        self.deform_type = "node"
        self.skinning = False
        self.hyper_dim = 8
        self.node_num = 1024
        self.pred_opacity = False
        self.pred_color = False
        self.use_hash = False
        self.hash_time = False
        self.d_rot_as_rotmat = False  # Debug!!!
        self.d_rot_as_res = True  # Debug!!!
        self.local_frame = False
        self.progressive_brand_time = False
        self.gs_with_motion_mask = False
        self.init_isotropic_gs_with_all_colmap_pcl = False
        self.as_gs_force_with_motion_mask = False  # Only for scenes with both static and dynamic parts and without alpha mask
        self.max_d_scale = -1.0
        self.is_scene_static = False

        self.net_width = 64  # width of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.timebase_pe = 4  # useless
        self.defor_depth = 1  # depth of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.posebase_pe = 10  # useless
        self.scale_rotation_pe = 2  # useless
        self.opacity_pe = 2  # useless
        self.timenet_width = 64  # useless
        self.timenet_output = 32  # useless
        self.bounds = 1.6
        self.plane_tv_weight = 0.0001  # TV loss of spatial grid
        self.time_smoothness_weight = 0.01  # TV loss of temporal grid
        self.l1_time_planes = 0.0001  # TV loss of temporal grid
        self.kplanes_config = {
            "grid_dimensions": 2,
            "input_coordinate_dim": 4,
            "output_coordinate_dim": 16,
            "resolution": [
                64,
                64,
                64,
                300,
            ],  # [64,64,64]: resolution of spatial grid. 25: resolution of temporal grid, better to be half length of dynamic frames
        }
        self.multires = [1, 2, 4, 8]  # multi resolution of voxel grid
        self.no_dx = False  # cancel the deformation of Gaussians' position
        self.no_grid = False  # cancel the spatial-temporal hexplane.
        self.no_ds = False  # cancel the deformation of Gaussians' scaling
        self.no_dr = False  # cancel the deformation of Gaussians' rotations
        self.no_do = True  # cancel the deformation of Gaussians' opacity
        self.no_dshs = True  # cancel the deformation of SH colors.
        self.empty_voxel = False  # useless
        self.grid_pe = 0  # useless, I was trying to add positional encoding to hexplane's features
        self.static_mlp = False  # useless
        self.apply_rotation = False  # useless
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        if not g.model_path.endswith(g.deform_type):
            g.model_path = os.path.join(
                os.path.dirname(os.path.normpath(g.model_path)),
                os.path.basename(os.path.normpath(g.model_path))
                + f"_{g.deform_type}",
            )
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.depth_ratio = 1.0
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.dataloader = False
        self.zerostamp_init = False
        self.custom_sampler = None
        self.iterations = 30_000
        self.coarse_iterations = 3000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 20_000
        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01
        self.grid_lr_init = 0.0016
        self.grid_lr_final = 0.00016

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0
        self.lambda_lpips = 0
        self.weight_constraint_init = 1
        self.weight_constraint_after = 0.2
        self.weight_decay_iteration = 5000
        self.opacity_reset_interval = 3000
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002
        self.pruning_from_iter = 500
        self.pruning_interval = 100
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005
        self.batch_size = 1
        self.add_point = False

        self.oneupSHdegree_step = 1000
        self.gt_alpha_mask_as_scene_mask = False
        self.gt_alpha_mask_as_dynamic_mask = False
        self.random_bg_color = False

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    if not args_cmdline.model_path.endswith(args_cmdline.deform_type):
        args_cmdline.model_path = os.path.join(
            os.path.dirname(os.path.normpath(args_cmdline.model_path)),
            os.path.basename(os.path.normpath(args_cmdline.model_path))
            + f"_{args_cmdline.deform_type}",
        )

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
