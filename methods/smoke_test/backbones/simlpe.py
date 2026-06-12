import copy

import numpy as np
import torch
from torch import nn
from einops.layers.torch import Rearrange


USED_JOINT_INDEXES = np.array(
    [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]
).astype(np.int64)


class MotionConfig:
    h36m_input_length = 50
    h36m_input_length_dct = 50
    h36m_target_length_train = 10
    h36m_target_length_eval = 25
    dim = 66


class MotionMLPConfig:
    hidden_dim = 66
    seq_len = 50
    num_layers = 48
    with_normalization = True
    spatial_fc_only = False
    norm_axis = "spatial"


class MotionFCConfig:
    in_features = 66
    out_features = 66
    with_norm = False
    activation = "relu"
    init_w_trunc_normal = False
    temporal_fc = False


class SiMLPeConfig:
    def __init__(self):
        self.motion = MotionConfig()
        self.motion_mlp = MotionMLPConfig()
        self.motion_fc_in = MotionFCConfig()
        self.motion_fc_out = MotionFCConfig()
        self.deriv_input = True
        self.deriv_output = True
        self.use_relative_loss = True
        self.pre_dct = False
        self.post_dct = False


def build_simlpe_config():
    """Standalone copy of dd_baseline_h36m/config.py values used by the backbone."""
    return copy.deepcopy(SiMLPeConfig())


class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class Spatial_FC(nn.Module):
    def __init__(self, dim):
        super(Spatial_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.arr0 = Rearrange("b n d -> b d n")
        self.arr1 = Rearrange("b d n -> b n d")

    def forward(self, x):
        x = self.arr0(x)
        x = self.fc(x)
        x = self.arr1(x)
        return x


class Temporal_FC(nn.Module):
    def __init__(self, dim):
        super(Temporal_FC, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLPblock(nn.Module):
    def __init__(self, dim, seq, use_norm=True, use_spatial_fc=False, layernorm_axis="spatial"):
        super().__init__()

        if not use_spatial_fc:
            self.fc0 = Temporal_FC(seq)
        else:
            self.fc0 = Spatial_FC(dim)

        if use_norm:
            if layernorm_axis == "spatial":
                self.norm0 = LN(dim)
            elif layernorm_axis == "temporal":
                self.norm0 = LN_v2(seq)
            elif layernorm_axis == "all":
                self.norm0 = nn.LayerNorm([dim, seq])
            else:
                raise NotImplementedError
        else:
            self.norm0 = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.fc.weight, gain=1e-8)
        nn.init.constant_(self.fc0.fc.bias, 0)

    def forward(self, x):
        x_ = self.fc0(x)
        x_ = self.norm0(x_)
        x = x + x_
        return x


class TransMLP(nn.Module):
    def __init__(self, dim, seq, use_norm, use_spatial_fc, num_layers, layernorm_axis):
        super().__init__()
        self.mlps = nn.Sequential(
            *[MLPblock(dim, seq, use_norm, use_spatial_fc, layernorm_axis) for i in range(num_layers)]
        )

    def forward(self, x):
        x = self.mlps(x)
        return x


def build_mlps(args):
    if hasattr(args, "seq_len"):
        seq_len = args.seq_len
    else:
        seq_len = None
    return TransMLP(
        dim=args.hidden_dim,
        seq=seq_len,
        use_norm=args.with_normalization,
        use_spatial_fc=args.spatial_fc_only,
        num_layers=args.num_layers,
        layernorm_axis=args.norm_axis,
    )


class siMLPe(nn.Module):
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        super(siMLPe, self).__init__()
        self.arr0 = Rearrange("b n d -> b d n")
        self.arr1 = Rearrange("b d n -> b n d")

        self.motion_mlp = build_mlps(self.config.motion_mlp)

        self.temporal_fc_in = config.motion_fc_in.temporal_fc
        self.temporal_fc_out = config.motion_fc_out.temporal_fc
        if self.temporal_fc_in:
            self.motion_fc_in = nn.Linear(
                self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct
            )
        else:
            self.motion_fc_in = nn.Linear(self.config.motion.dim, self.config.motion.dim)
        if self.temporal_fc_out:
            self.motion_fc_out = nn.Linear(
                self.config.motion.h36m_input_length_dct, self.config.motion.h36m_input_length_dct
            )
        else:
            self.motion_fc_out = nn.Linear(self.config.motion.dim, self.config.motion.dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.motion_fc_out.weight, gain=1e-8)
        nn.init.constant_(self.motion_fc_out.bias, 0)

    def forward(self, motion_input):
        if self.temporal_fc_in:
            motion_feats = self.arr0(motion_input)
            motion_feats = self.motion_fc_in(motion_feats)
        else:
            motion_feats = self.motion_fc_in(motion_input)
            motion_feats = self.arr0(motion_feats)

        motion_feats = self.motion_mlp(motion_feats)

        if self.temporal_fc_out:
            motion_feats = self.motion_fc_out(motion_feats)
            motion_feats = self.arr1(motion_feats)
        else:
            motion_feats = self.arr1(motion_feats)
            motion_feats = self.motion_fc_out(motion_feats)

        return motion_feats


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def _some_variables():
    parent = np.array(
        [0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13, 17, 18, 19, 20, 21, 20, 23,
         13, 25, 26, 27, 28, 29, 28, 31]
    ) - 1

    offset = np.array(
        [0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000, 0.000000, -442.894612,
         0.000000, 0.000000, -454.206447, 0.000000, 0.000000, 0.000000, 162.767078, 0.000000,
         0.000000, 74.999437, 132.948826, 0.000000, 0.000000, 0.000000, -442.894413, 0.000000,
         0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426, 0.000000, 0.000000,
         74.999948, 0.000000, 0.100000, 0.000000, 0.000000, 233.383263, 0.000000, 0.000000,
         257.077681, 0.000000, 0.000000, 121.134938, 0.000000, 0.000000, 115.002227, 0.000000,
         0.000000, 257.077681, 0.000000, 0.000000, 151.034226, 0.000000, 0.000000, 278.882773,
         0.000000, 0.000000, 251.733451, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         0.000000, 99.999627, 0.000000, 100.000188, 0.000000, 0.000000, 0.000000, 0.000000,
         0.000000, 257.077681, 0.000000, 0.000000, 151.031437, 0.000000, 0.000000, 278.892924,
         0.000000, 0.000000, 251.728680, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
         0.000000, 99.999888, 0.000000, 137.499922, 0.000000, 0.000000, 0.000000, 0.000000]
    )
    offset = offset.reshape(-1, 3)

    rotInd = [[5, 6, 4],
              [8, 9, 7],
              [11, 12, 10],
              [14, 15, 13],
              [17, 18, 16],
              [],
              [20, 21, 19],
              [23, 24, 22],
              [26, 27, 25],
              [29, 30, 28],
              [],
              [32, 33, 31],
              [35, 36, 34],
              [38, 39, 37],
              [41, 42, 40],
              [],
              [44, 45, 43],
              [47, 48, 46],
              [50, 51, 49],
              [53, 54, 52],
              [56, 57, 55],
              [],
              [59, 60, 58],
              [],
              [62, 63, 61],
              [65, 66, 64],
              [68, 69, 67],
              [71, 72, 70],
              [74, 75, 73],
              [],
              [77, 78, 76],
              []]

    expmapInd = np.split(np.arange(4, 100) - 1, 32)
    return parent, offset, rotInd, expmapInd


def fkl_torch(rotmat, parent, offset, rotInd, expmapInd):
    n = rotmat.data.shape[0]
    j_n = offset.shape[0]
    offset_t = torch.from_numpy(offset).float().to(rotmat.device)
    R = rotmat.view(n, j_n, 3, 3)
    rotations = [R[:, 0, :, :]]
    positions = [offset_t[0].unsqueeze(0).repeat(n, 1)]
    for i in np.arange(1, j_n):
        if parent[i] > 0:
            parent_rot = rotations[parent[i]]
            current_rot = torch.matmul(R[:, i, :, :], parent_rot)
            current_pos = torch.matmul(offset_t[i], parent_rot) + positions[parent[i]]
        else:
            current_rot = R[:, i, :, :]
            current_pos = offset_t[i].unsqueeze(0).repeat(n, 1)
        rotations.append(current_rot)
        positions.append(current_pos)
    return torch.stack(positions, dim=1)


def expmap2rotmat_torch(r):
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = torch.eye(3, 3).repeat(n, 1, 1).float().to(r.device) + torch.mul(
        torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
        (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
    return R


def rotmat2xyz_torch(rotmat):
    assert rotmat.shape[1] == 32
    parent, offset, rotInd, expmapInd = _some_variables()
    xyz = fkl_torch(rotmat, parent, offset, rotInd, expmapInd)
    return xyz


def expmap_to_xyz32(expmap):
    """Mirror DDH36MNPZBase._expmap_to_xyz32 for batched [B, T, 99] tensors."""
    if expmap.dim() != 3 or expmap.shape[-1] != 99:
        raise ValueError("Expected expmap shape [B, T, 99], got {}".format(tuple(expmap.shape)))

    batch_size, seq_len, _ = expmap.shape
    pose_info = expmap.reshape(batch_size * seq_len, 33, 3).clone()
    zeros = torch.zeros_like(pose_info[:, :2])
    pose_info = torch.cat([zeros, pose_info[:, 2:]], dim=1)
    pose_info = pose_info.reshape(-1, 3)
    pose_info = expmap2rotmat_torch(pose_info).reshape(batch_size * seq_len, 33, 3, 3)[:, 1:]
    xyz_info = rotmat2xyz_torch(pose_info)
    return xyz_info.reshape(batch_size, seq_len, 32, 3)


def expmap_to_simlpe_xyz66(expmap):
    """Convert Human3.6M expmap [B, T, 99] to original siMLPe xyz66 [B, T, 66]."""
    xyz_info = expmap_to_xyz32(expmap)
    joint_indexes = torch.tensor(USED_JOINT_INDEXES, dtype=torch.long, device=expmap.device)
    return xyz_info[:, :, joint_indexes, :].reshape(expmap.shape[0], expmap.shape[1], -1) / 1000.0


class SiMLPeMotionBackbone(nn.Module):
    """Adapter around the standalone siMLPe, preserving dd_baseline_h36m's xyz66 workflow."""

    def __init__(self, config=None):
        super().__init__()
        self.config = build_simlpe_config() if config is None else copy.deepcopy(config)
        self.model = siMLPe(self.config)

        dct_m, idct_m = get_dct_matrix(self.config.motion.h36m_input_length_dct)
        self.register_buffer("dct_m", torch.tensor(dct_m).float().unsqueeze(0))
        self.register_buffer("idct_m", torch.tensor(idct_m).float().unsqueeze(0))

    @property
    def input_len(self):
        return self.config.motion.h36m_input_length

    @property
    def output_len(self):
        return self.config.motion.h36m_target_length_train

    def forward(self, past_expmap, output_len=None):
        past_xyz = expmap_to_simlpe_xyz66(past_expmap)
        return self.forward_xyz66(past_xyz, output_len=output_len)

    def forward_xyz66(self, past_xyz, output_len=None):
        output_len = self.output_len if output_len is None else output_len
        if self.config.deriv_input:
            model_in = torch.matmul(self.dct_m[:, :, : self.config.motion.h36m_input_length], past_xyz)
        else:
            model_in = past_xyz.clone()

        pred = self.model(model_in)
        pred = torch.matmul(self.idct_m[:, : self.config.motion.h36m_input_length, :], pred)

        if self.config.deriv_output:
            pred = pred[:, :output_len] + past_xyz[:, -1:, :].repeat(1, output_len, 1)
        else:
            pred = pred[:, :output_len]
        return pred
