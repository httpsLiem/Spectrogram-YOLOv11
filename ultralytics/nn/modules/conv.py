# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "Conv",
    "ConvE",
    "RecycleConv",
    "Conv2",
    "LightConv",
    "SelectChannel",
    "DWConv",
    "DDWConv",
    "ConvOMN",
    "MaxConv",
    "GhostMaxConv",
    "GhostConvMax",
    "ConvHCA",
    "MaxConvHCA",
    "FilterMaxConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "GhostConvHCA",
    "LDConv",
    "LDConv1",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "AKCBAM",
    "CoordAtt",
    "HCoordAtt",
    "MaxHCoordAtt",
    "Fusion",
    "BiFPN_WConcat",
    "BiFPN_WConcat3",
    "BiFPN_WConcat2",
    "Concat",
    "RepConv",
    "Index",
    "GlobalContext",
    "MSCSpatialAttention",
    "BottleNect",
    "FGM",
    "ConvGN",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))
    
class ConvE(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.ELU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))

class ConvGN(nn.Module):
    """Standard convolution with GroupNorm instead of BatchNorm."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, gn_groups=32):
        """Initialize Conv layer with GroupNorm."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(gn_groups, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, group normalization, and activation to input tensor."""
        return self.act(self.gn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without normalization."""
        return self.act(self.conv(x))


################
# LDConv
################


# class RecycleConv(nn.Module):
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
#         self.c_ = c2 // 8
#         self.p = autopad(k, p, d)
#         self.s = s
#         self.conv = nn.Conv2d(c1//8, self.c_, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn_silu = nn.Sequential(
#             nn.BatchNorm2d(self.c_),
#             nn.SiLU()
#         )
#         self.trainable_scale = nn.Parameter(torch.ones(c1//8, 1, 1))  # Hệ số có thể học được
#         # Đăng ký saved_kernel như một tham số
#         self.saved_kernel = nn.Parameter(self.conv.weight.clone().detach(), requires_grad=True)
#         self.conv1 = Conv(self.c_, self.c_, k=1, s=1)
#
#     def forward(self, x):
#         chunks = torch.chunk(x, 8, dim=1)
#         processed_chunks = []
#
#         for i, c in enumerate(chunks):
#             if i == 0:
#                 out = self.bn_silu(self.conv(c))
#                 # Cập nhật saved_kernel trong forward nếu cần
#                 with torch.no_grad():
#                     self.saved_kernel.data = self.conv.weight.clone()
#             else:
#                 out = F.conv2d(c, self.saved_kernel * self.trainable_scale, padding=self.p, stride=self.s)
#                 out = self.bn_silu(out)
#
#             processed_chunks.append(out)
#
#         return torch.cat(processed_chunks, dim=1)

class RecycleConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.c_ = c2 // 8
        self.p = autopad(k, p, d)
        self.s = s
        self.conv = nn.Conv2d(c1//8, self.c_, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn_silu = nn.Sequential(
            nn.BatchNorm2d(self.c_),
            nn.SiLU()
        )
        self.trainable_scale = nn.Parameter(torch.ones(c1//8, 1, 1))  # Hệ số có thể học được theo kênh
        self.saved_kernel = None  # Biến lưu trọng số kernel
        self.conv1 = Conv(self.c_, self.c_, k=1, s=1)

    def forward(self, x):
        chunks = torch.chunk(x, 8, dim=1)  # Chia tensor đầu vào thành 4 phần theo chiều kênh
        # print(chunks[0].shape)
        # print(chunks[1].shape)
        processed_chunks = []

        for i, c in enumerate(chunks):
            if i == 0:
                out = self.bn_silu(self.conv(c))  # Xử lý khối đầu tiên
                # self.saved_kernel = self.conv.weight.clone()  # Lưu trọng số kernel của khối đầu tiên
            else:
                # out = F.conv2d(c, self.saved_kernel, padding=self.p, stride=self.s)
                # out = self.conv1(out)  # Chuẩn hóa và kích hoạt
                # Áp dụng kernel đã lưu vào c với trainable_scale
                out = F.conv2d(c, self.conv.weight.clone() * self.trainable_scale, padding=self.p, stride=self.s)
                out = self.bn_silu(out)  # Chuẩn hóa và kích hoạt
                # self.conv.weight.data = self.saved_kernel * self.trainable_scale  # Điều chỉnh kernel với hệ số có thể học được
                # out = self.bn_silu(self.conv(c))  # Chuẩn hóa và kích hoạt

            processed_chunks.append(out)
        # print(processed_chunks[0].shape)
        # print(processed_chunks[1].shape)

        return torch.cat(processed_chunks, dim=1)  # Nối lại thành tensor đầu ra



from einops import rearrange
class LDConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1),  stride=(num_param, 1), bias=bias),
                                  # nn.BatchNorm2d(outc),
                                  # nn.ReLU()
                                  nn.Sigmoid()
                                  )  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the LDConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)

        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset

class LDConv1(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv1, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(1, num_param),  stride=(1, num_param), bias=bias),
                                  nn.BatchNorm2d(outc),
                                  nn.SiLU()
                                  )  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the LDConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()

        # using Conv3d
        # y_offset = y_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)

        # using 1 × 1 Conv
        # y_offset = y_offset.permute(0,1,4,2,3), then, y_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)

        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)

        x_offset = rearrange(x_offset, 'b c h w n -> b c h (w n)')

        return x_offset
class SLDConv(nn.Module):
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(SLDConv, self).__init__()
        self.num_param = num_param
        self.stride = stride
        self.conv = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=(num_param, 1),  stride=(num_param, 1), bias=bias),
                                  # nn.BatchNorm2d(outc),
                                  nn.ReLU()
                                  )  # the conv adds the BN and SiLU to compare original Conv in YOLOv5.
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # N is num_param.
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # resampling the features based on the modified coordinates.
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # bilinear
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        out = self.conv(x_offset)

        return out

    # generating the inital sampled shapes for the LDConv with different sizes.
    def _get_p_n(self, N, dtype):
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int
        mod_number = self.num_param % base_int
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),
            torch.arange(0, base_int))
        p_n_x = torch.flatten(p_n_x)
        p_n_y = torch.flatten(p_n_y)
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),
                torch.arange(0, mod_number))

            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            p_n_x, p_n_y = torch.cat((p_n_x, mod_p_n_x)), torch.cat((p_n_y, mod_p_n_y))
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # no zero-padding
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),
            torch.arange(0, w * self.stride, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    #  Stacking resampled features in the row direction.
    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        b, c, h, w, n = x_offset.size()
        # using Conv3d
        # x_offset = x_offset.permute(0,1,4,2,3), then Conv3d(c,c_out, kernel_size =(num_param,1,1),stride=(num_param,1,1),bias= False)
        # using 1 × 1 Conv
        # x_offset = x_offset.permute(0,1,4,2,3), then, x_offset.view(b,c×num_param,h,w)  finally, Conv2d(c×num_param,c_out, kernel_size =1,stride=1,bias= False)
        # using the column conv as follow， then, Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias)

        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset

class ConvOMN(nn.Module):   ###lighter
    def __init__(self, c1, c2, k=3, s=2, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # g = c1
        # self.conv1 = Conv(c1, c2, (k, 1), s, g=math.gcd(c1, c2), d=d, act=act)
        self.conv = Conv(c1, c2, k, s, g=1, d=d, act=act)
        # self.conv2 = Conv(c1, c2, k, s, g=1, d=d, act=act)
        self.kz = k
        self.omn = BottleNect(c2)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        # x1 = torch.nn.functional.max_pool2d(x, kernel_size=self.kz, stride=2)
        # x2 = torch.nn.functional.max_pool2d(x, kernel_size=self.kz, stride=2)
        # x3 = torch.cat((x1, x2),1)
        # return self.conv2(x)
        # return self.conv2(self.conv1(x))
        return self.omn(self.conv(x))

class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class DDWConv(nn.Module):   #real
    def __init__(self, c1, c2, k=3, s=2, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # g = c1
        self.conv1 = Conv(c1, c2, k, s, g=8, d=d, act=act)
        # self.conv1 = Conv(c1, c2, k, s, g=1, d=d, act=act)
        self.kz = k
        self.conv2 = Conv(c2, c2, k=1, s=1)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        # x1 = torch.nn.functional.max_pool2d(x, kernel_size=self.kz, stride=2)
        # x2 = torch.nn.functional.max_pool2d(x, kernel_size=self.kz, stride=2)
        # x3 = torch.cat((x1, x2),1)
        # return self.conv1(x)
        return self.conv2(self.conv1(x))

# class DDWConv(nn.Module):
#     def __init__(self, c1, c2, k=3, s=2, d=1, act=True):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         # g = c1
#         self.conv1 = Conv(c1, c2, k, s, g=8, d=d, act=act)
#         # self.conv1 = Conv(c1, c2, k, s, g=1, d=d, act=act)
#         self.kz = k
#         self.conv2 = Conv(c2, c2, k=1, s=1)
#         self.msc = HCoordAtt(c2, c2, 32)
#
#     def forward(self, x):
#         """Apply 2 convolutions to input tensor."""
#         # x1 = torch.nn.functional.max_pool2d(x, kernel_size=self.kz, stride=2)
#         # x2 = torch.nn.functional.max_pool2d(x, kernel_size=self.kz, stride=2)
#         # x3 = torch.cat((x1, x2),1)
#         # return self.conv1(x)
#         return self.msc(self.conv2(self.conv1(x)))

# class MaxConv(nn.Module):
#     def __init__(self, c1, c2, k=3, s=2, d=1, act=True):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         # g = c1
#         # self.conv1 = Conv(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
#         self.kz = k
#         self.stride = s
#         self.dilation = d
#         self.conv2 = Conv(c1, c2, k=1, s=1)
#
#     def forward(self, x):
#         """Apply 2 convolutions to input tensor."""
#         x1 = torch.nn.functional.avg_pool2d(x, kernel_size=(self.kz,1), stride=self.stride, padding=(self.kz//2,0))
#         # x2 = torch.nn.functional.max_pool2d(x, kernel_size=self.kz, stride=2)
#         # x3 = torch.cat((x1, x2),1)
#         return self.conv2(x1)
class MaxConvHCA(nn.Module):   #MSCBam
    def __init__(self, c1, c2, k=3, s=2, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # g = c1
        # self.conv1 = Conv(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
        self.kz = k
        self.stride = s
        self.dilation = d
        self.conv2 = Conv(c1, c2, k=1, s=1)
        self.msc = HCoordAtt(c2, c2, 32)
        # self.msc = GlobalContext(c2)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        # x1 = torch.nn.functional.max_pool2d(x, kernel_size=(self.kz, 1), stride=self.stride,
        #                                     padding=(self.kz // 2, 0))
        x1 = torch.nn.functional.max_pool2d(x, kernel_size=(1,self.kz), stride=self.stride,
                                            padding=(0,self.kz // 2))
        # print('gc')
        # x1 = self.msc(x1)
        # x2 = torch.nn.functional.max_pool2d(x, kernel_size=self.kz, stride=2)
        # x3 = torch.cat((x1, x2),1)
        # return self.msc(self.conv2(x1))
        return self.msc(self.conv2(x1))
class FilterMaxConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # g = c1
        # self.conv1 = Conv(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
        self.kz = k
        self.stride = s
        self.dilation = d
        self.conv2 = Conv(c1, c2, k=k, s=s)
        self.msc = HCoordAtt(c2, c2, 32)
        # self.msc = GlobalContext(c2)
    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        # x1 = torch.nn.functional.max_pool2d(x, kernel_size=(self.kz, 1), stride=self.stride,
        #                                     padding=(self.kz // 2, 0))
        x1 = self.conv2(x)
        x2 = torch.nn.functional.max_pool2d(x1, kernel_size=(1,self.kz), stride=1, padding=(0, self.kz // 2))
        # print('gc')
        # x1 = self.msc(x1)
        # x2 = torch.nn.functional.max_pool2d(x, kernel_size=self.kz, stride=2)
        # x3 = torch.cat((x1, x2),1)
        # return self.msc(self.conv2(x1))
        return self.msc(x2)
class PartialConvMax(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # g = c1
        # self.conv1 = Conv(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
        self.kz = k
        self.stride = s
        self.dilation = d
        self.conv2 = Conv(c1, c1, k=k, s=s)
        self.fc = nn.Sequential(nn.Conv2d(c1*2, 8, 1, 1, 0, bias=True),
                                # nn.BatchNorm2d(8),
                                nn.ReLU())
        self.fc1 = nn.Sequential(nn.Conv2d(8, c2, kernel_size=1, stride=1, padding=0),
                                 nn.ReLU())
        # self.msc = HCoordAtt(c1, c1, 32)
        # self.msc = GlobalContext(c2)
    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        # x1 = torch.nn.functional.max_pool2d(x, kernel_size=(self.kz, 1), stride=self.stride,
        #                                     padding=(self.kz // 2, 0))
        x1 = self.conv2(x)
        x2 = torch.nn.functional.max_pool2d(x1, kernel_size=(1,self.kz), stride=1, padding=(0, self.kz // 2))
        x3 = torch.cat((x1, x2), 1)

        # print('gc')
        # x1 = self.msc(x1)
        # x2 = torch.nn.functional.max_pool2d(x, kernel_size=self.kz, stride=2)

        # return self.msc(self.conv2(x1))
        return self.fc1(self.fc(x3))

class ConvHCA(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # g = c1
        # self.conv1 = Conv(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
        self.kz = k
        self.stride = s
        self.dilation = d
        self.conv2 = Conv(c1, c2, k=k, s=s)
        # self.hca = HCoordAtt(c2, c2, 32)
        self.hca = SobelSpatialAttention(7)
    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        x1 = self.conv2(x)
        return self.hca(x1)
class GhostConvMax(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # g = c1
        # self.conv1 = Conv(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
        self.kz = k
        self.stride = s
        self.dilation = d
        self.conv2 = Conv(c1, c1, k=k, s=s, g=1)
        self.conv3 = Conv(c1*2, c2, k=1, s=1)
        self.hca = HCoordAtt(c2, c2, 32)
        # self.fc = nn.Sequential(nn.Conv2d(c1*2, 8, 1, 1, 0, bias=True),
        #                         # nn.BatchNorm2d(8),
        #                         nn.ReLU())
        # self.fc1 = nn.Sequential(nn.Conv2d(8, c2, kernel_size=1, stride=1, padding=0),
        #                          nn.ReLU())
        # self.msc = HCoordAtt(c1, c1, 32)
        # self.msc = GlobalContext(c2)
    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        # x1 = torch.nn.functional.max_pool2d(x, kernel_size=(self.kz, 1), stride=self.stride,
        #                                     padding=(self.kz // 2, 0))
        x1 = self.conv2(x)
        x2 = torch.nn.functional.max_pool2d(x1, kernel_size=(self.kz,1), stride=1, padding=(self.kz // 2, 0))
        x3 = torch.cat((x1, x2), 1)

        # print('gc')
        # x1 = self.msc(x1)
        # x2 = torch.nn.functional.max_pool2d(x, kernel_size=self.kz, stride=2)

        # return self.msc(self.conv2(x1))
        # return self.fc1(self.fc(x3))
        return self.hca(self.conv3(x3))
        # return self.conv3(x3)
class GhostMaxConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # g = c1
        # self.conv1 = Conv(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
        self.kz = k
        self.stride = s
        self.dilation = d
        self.conv2 = Conv(c1, c2//2, k=k, s=s, g=8)
        # self.conv3 = Conv(c2*2, c2, k=1, s=1)
        self.conv3 = Conv(c2//2, c2//2, k=1, s=1)
        self.hca = HCoordAtt(c2, c2, 32)
        # self.hca = SobelSpatialAttention(3)
        # self.fc = nn.Sequential(nn.Conv2d(c1*2, 8, 1, 1, 0, bias=True),
        #                         # nn.BatchNorm2d(8),
        #                         nn.ReLU())
        # self.fc1 = nn.Sequential(nn.Conv2d(8, c2, kernel_size=1, stride=1, padding=0),
        #                          nn.ReLU())
        # self.msc = HCoordAtt(c1, c1, 32)
        # self.msc = GlobalContext(c2)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        # x1 = torch.nn.functional.max_pool2d(x, kernel_size=(self.kz, 1), stride=self.stride,
        #                                     padding=(self.kz // 2, 0))
        x1 = self.conv3(self.conv2(x))
        # x1 = self.conv2(x)
        x2 = torch.nn.functional.max_pool2d(x1, kernel_size=(self.kz, 1), stride=1, padding=(self.kz // 2, 0))
        # x2 = torch.nn.functional.max_pool2d(x1, kernel_size=3, stride=1, padding=(3 //2))
        # x2 = torch.nn.functional.avg_pool2d(x1, kernel_size=3, stride=1, padding=(3 //2))
        x3 = torch.cat((x1, x2), 1)

        # print('gc')
        # x1 = self.msc(x1)
        # x2 = torch.nn.functional.max_pool2d(x, kernel_size=self.kz, stride=2)

        # return self.msc(self.conv2(x1))
        # return self.fc1(self.fc(x3))
        return self.hca(x3)
        # return self.hca(self.conv3(x3))


class MaxConv(nn.Module):   #MSCBam
    def __init__(self, c1, c2, k=3, s=2, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # g = c1
        # self.conv1 = Conv(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
        self.kz = k
        self.stride = s
        self.dilation = d
        self.conv2 = Conv(c1, c2//2, k=k, s=s, g=8)
        self.conv3 = Conv(c2//2, c2//2, k=1, s=1)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        x1 = self.conv3(self.conv2(x))
        x2 = torch.nn.functional.max_pool2d(x1, kernel_size=(self.kz, 1), stride=1, padding=(self.kz // 2, 0)) #main
        x3 = torch.cat((x1, x2), 1)
        return x3


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        # self.cv2 = Conv(c_, c_, 5, 1, None, c_, d=2, act=act)
        self.cv2 = Conv(c_, c_, 3, 1, None, 1, d=2, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)
class GhostConvHCA(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 3, 1, None, 1, d=2, act=act)
        self.hca = HCoordAtt(c2, c2, 32)
    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return self.hca(torch.cat((y, self.cv2(y)), 1))

class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
class SobelConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super(SobelConv, self).__init__()
        # Sobel kernels cho các góc 0°, 45°, 90°, ...
        angles = [0, 45, 90]  # Có thể thêm 135°, 180°, ...
        self.convs = nn.ModuleList()

        for angle in angles:
            # Tạo Sobel kernel cơ bản
            sobel_x = torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]], dtype=torch.float32)

            # Xoay kernel theo góc (đơn giản hóa: chỉ dùng sobel_x và sobel_y)
            if angle == 45:
                sobel_x = sobel_x + sobel_y  # Gần đúng cho góc 45°
            elif angle == 90:
                sobel_x = sobel_y  # Sobel Y cho 90°

            sobel_x = sobel_x.view(1, 1, 3, 3)
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False)
            conv.weight = nn.Parameter(sobel_x.repeat(out_channels, 1, 1, 1), requires_grad=True)
            self.convs.append(conv)
    def forward(self, x):
        # Áp dụng Sobel cho từng góc và cộng kết quả
        outputs = [conv(x) for conv in self.convs]
        return sum(outputs)  # TopK (ở đây đơn giản hóa bằng phép cộng)
class SobelSpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.sobel = SobelConv(in_channels=2, out_channels=2)
        self.cv1 = nn.Conv2d(2, 1, 1, padding=1//2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(self.sobel(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1))))

class MSCSpatialAttention(nn.Module):
    """Spatial-attention module."""
    def __init__(self, c1, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2d(2, 1, (31,31), padding=(31//2,31//2), bias=False),
            # nn.Sigmoid()
            nn.ReLU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(2, 1, (3, 3), padding=(1,1), bias=False),
            # nn.Sigmoid()
            nn.ReLU()
        )
        # self.cv2 = nn.Sequential(
        #     nn.Conv2d(2, 1, (1,1), padding=(0, 0), bias=False),
        #     # nn.Sigmoid()
        #     nn.ReLU()
        # )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(c1, c1, 1, 1, 0, bias=True)
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU()
    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        x1 = torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)
        # print('1')
        x2 = self.cv1(x1)
        x3 = self.cv2(x1)
        x4 = x * x2
        x5 = x * x3

        x6 = x4+x5
        # x6 = x4 + x5 + x

        x7 = self.pool(x6)
        x8 = self.act(self.fc(x7))
        x9 = self.act(self.fc(x7))

        x10 = x4 * x8
        x11 = x5 * x9

        return x10 + x11 + x

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))
class AKSpatialAttention(nn.Module):
    """Spatial-attention module."""
    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = LDConv(2, 1, 7, 1)
        # self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1))
class SAKSpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        # assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        # padding = 3 if kernel_size == 7 else 1
        self.cv1 = LDConv(2, 1, 7, 1)
        self.cv2 = LDConv(2, 1, 15, 1)
        # self.cv = nn.Conv2d(2, 1, 1, padding=0, bias=False)
        # self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        x1 = torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)

        x2 = self.cv1(x1)
        x3 = self.cv2(x1)
        x5 = x * x2
        x6 = x * x3

        return x5+x6
class AKSESpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, channels: int) -> None:
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        self.cv1 = LDConv(channels, int(channels*0.5), 7, 1)
        self.cv2 = nn.Conv2d(int(channels*0.5), channels, 1, padding=0, bias=False)
        # self.bn = nn.BatchNorm2d(channels)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act((self.cv2(self.cv1(x))))
class SAKAM(nn.Module):
    """Spatial-attention module."""
    def __init__(self, c1, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        self.cv1 = LDConv(2, 1, 7, 1)
        self.cv2 = LDConv(2, 1, 15, 1)
        # self.cv1 = nn.Sequential(
        #     SLDConv(2, 10, 7, 1),
        #     nn.Conv2d(10, 1, 1, padding=0, bias=False),
        #     # nn.Sigmoid()
        #     nn.ReLU()
        # )
        # self.cv2 = nn.Sequential(
        #     SLDConv(2, 10, 15, 1),
        #     nn.Conv2d(10, 1, 1, padding=0, bias=False),
        #     # nn.Sigmoid()
        #     nn.ReLU()
        # )

        # self.cv1 = nn.Sequential(
        #     nn.Conv2d(2, 1, 7, padding=7//2, bias=False),
        #     # nn.Sigmoid()
        #     nn.ReLU()
        # )
        # self.cv2 = nn.Sequential(
        #     nn.Conv2d(2, 1, 15, padding=15//2, bias=False),
        #     # nn.Sigmoid()
        #     nn.ReLU()
        # )

        self.channel_attention = ECAAttention(c1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(c1, c1, 1, 1, 0, bias=True)
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU()
    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        x1 = torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)

        x2 = self.cv1(x1)
        x3 = self.cv2(x1)
        x4 = x * x2
        x5 = x * x3

        x6 = x4+x5
        # x6 = x4 + x5 + x

        x7 = self.pool(x6)
        x8 = self.act(self.fc(x7))
        x9 = self.act(self.fc(x7))

        # x8 = self.channel_attention(x6)
        # x9 = self.channel_attention(x6)


        # x0 = self.act(self.fc(x7))

        x10 = x4 * x8
        x11 = x5 * x9

        # x00 = x * x0
        # return x10 + x11
        # return x10 + x11 + x00
        return x10 + x11 + x
class SAKAM1(nn.Module):
    """Spatial-attention module."""
    def __init__(self, c1, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        self.cv1 = SLDConv(2, 1, 3, 1)
        self.cv2 = SLDConv(2, 1, 7, 1)
        self.cv3 = SLDConv(2, 1, 15, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(c1, c1, 1, 1, 0, bias=True)
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU()
        self.channel_attention = nn.Sequential(
            # nn.Linear(c1, int(c1 / 4)),
            nn.Conv2d(c1, int(c1 / 4), 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            # nn.Linear(int(c1 / 4), c1)
        )
        self.up = nn.Sequential(
            nn.Conv2d(int(c1 / 4), c1, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        x1 = torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)

        x2 = self.cv1(x1)
        x3 = self.cv2(x1)
        x3_1 = self.cv2(x1)
        x4 = x * x2
        x5 = x * x3
        x5_1 = x * x3_1

        x6 = x4 + x5 + x5_1
        x7 = self.pool(x6)
        # x8 = self.up(self.channel_attention(x7))
        # x9 = self.up(self.channel_attention(x7))
        # x9_1 = self.up(self.channel_attention(x7))
        x8 = self.act(self.fc(x7))
        x9 = self.act(self.fc(x7))
        x9_1 = self.act(self.fc(x7))

        x10 = x4 * x8
        x11 = x5 * x9
        x11_1 = x5_1 * x9_1
        # return x10 + x11
        return x10 + x11 + x11_1
class ECAAttention(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, c1, k_size=3):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        # self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.ReLU()
    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return y
        # return x * y.expand_as(x)
class AKCBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        # self.channel_attention = ChannelAttention(c1)
        # self.spatial_attention = AKSpatialAttention(7)
        # self.spatial_attention = SAKSpatialAttention(7)
        self.spatial_attention = SAKAM(c1, 7)
        # self.spatial_attention = SAKAM1(c1, 7)
        # self.spatial_attention = AKSESpatialAttention(c1)
    def forward(self, x):
        """Applies the forward pass through C1 module."""
        # return self.spatial_attention(self.channel_attention(x))
        # return self.channel_attention(self.spatial_attention(x))
        return self.spatial_attention(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):  # inp: number of input channel, oup: number of output channel
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        # print('yeah')
        return out
class CoordTran(nn.Module):
    def __init__(self, inp, oup, reduction=32):  # inp: number of input channel, oup: number of output channel
        super(CoordTran, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h

        return out
# class HCoordAtt(nn.Module):
#     def __init__(self, inp, oup, reduction=32):  # inp: number of input channel, oup: number of output channel
#         super(HCoordAtt, self).__init__()
#         # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         self.stdpool_w = GlobalStdPool2d()
#
#         mip = max(8, inp // reduction)
#
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#
#         self.conv_stdw = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         identity = x
#
#         _, _, h, w = x.size()
#         x_stdw = self.stdpool_w(x)
#         # print(x_stdw.shape)
#         x_w = self.pool_w(x)
#
#         y = torch.cat([x_stdw, x_w], dim=3)
#         # print(y.shape)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)
#
#         x_stdw, x_w = torch.split(y, [w, w], dim=3)
#
#         a_stdw = self.conv_stdw(x_stdw).sigmoid()
#         # print(a_stdw.shape)
#         a_w = self.conv_w(x_w).sigmoid()
#         # print(a_w.shape)
#         out = identity * a_w * a_stdw
#         return out

# class HCoordAtt(nn.Module):   #############main
#     def __init__(self, inp, oup, reduction=32):  # inp: number of input channel, oup: number of output channel
#         super(HCoordAtt, self).__init__()
#         # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         # self.pool_w = GlobalStdPool2d()
#
#         mip = max(8, inp // reduction)
#
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#         # self.relu = nn.ReLU()
#         # self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         identity = x
#         # print('2')
#         _, _, h, w = x.size()
#         # x_h = self.pool_h(x)
#         x_w = self.pool_w(x)
#         # print(x_w.shape)
#
#         x_w = self.conv1(x_w)
#         x_w = self.bn1(x_w)
#         x_w = self.act(x_w)
#
#         x_w = x_w
#
#         a_w = self.conv_w(x_w).sigmoid()
#         # a_w = self.relu(self.conv_w(x_w))
#         # print(x_w.shape)
#         # print(a_w.shape)
#         out = identity * a_w
#
#         return out
class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
class HCoordAtt(nn.Module):   #############main
    def __init__(self, inp, oup, reduction=32):  # inp: number of input channel, oup: number of output channel
        super(HCoordAtt, self).__init__()
        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.cv1 = nn.Conv2d(2, 1, 3, padding=3 // 2, bias=False)
        self.act = nn.Sigmoid()
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # self.pool_w = GlobalStdPool2d()
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(oup)
        self.act = nn.Sigmoid()
        # self.relu = nn.ReLU()
        # self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        # self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        # print('2')
        _, _, h, w = x.size()
        x_pooled = self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
        # x_h = self.pool_h(x)
        x_w = self.pool_w(x_pooled)
        #
        # x_w = self.conv1(x_w)
        # x_w = self.bn1(x_w)
        # x_w = self.act(x_w)

        # a_w = self.relu(self.conv_w(x_w))
        # print(x_w.shape)
        # print(x_w.shape)
        # out = identity * x_pooled * x_w
        out = identity * x_w

        return out

# Định nghĩa StdPool với đầu ra giống nn.AdaptiveAvgPool2d((1, W))
class GlobalStdPool2d(nn.Module):
    def __init__(self):
        super(GlobalStdPool2d, self).__init__()

    def forward(self, x):
        # print('1')
        # x shape: [N, C, H, W]
        mean = torch.mean(x, dim=2, keepdim=True)  # Tính trung bình trên chiều H, giữ W
        # mean shape: [N, C, 1, W]
        variance = torch.mean((x - mean) ** 2, dim=2, keepdim=True)  # Tính phương sai trên H
        # variance shape: [N, C, 1, W]
        std = torch.sqrt(variance + 1e-5)  # Tính độ lệch chuẩn
        # std shape: [N, C, 1, W]
        return std
#
# # Sử dụng StdPool
# std_pool = GlobalStdPool2d()
# x = torch.randn(2, 64, 32, 32)  # Đầu vào: [N, C, H, W] = [2, 64, 32, 32]
# output = std_pool(x)
# print(output.shape)  # torch.Size([2, 64, 1, 32])


# class HCoordAtt(nn.Module): ##################+max pool
#     def __init__(self, inp, oup, reduction=32):  # inp: number of input channel, oup: number of output channel
#         super(HCoordAtt, self).__init__()
#         # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         self.pool_m = nn.AdaptiveMaxPool2d((1, None))
#
#         mip = max(8, inp // reduction)
#
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#
#         # self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_m = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         identity = x
#
#         _, _, h, w = x.size()
#         # x_h = self.pool_h(x)
#         x_w = self.pool_w(x)
#         x_m = self.pool_m(x)
#
#         x_o = x_w + x_m
#         # print(x_w.shape)
#
#
#         x_o = self.conv1(x_o)
#         x_o = self.bn1(x_o)
#         x_o = self.act(x_o)
#
#         a_w = self.conv_w(x_o).sigmoid()
#         # print(a_w.shape)
#         out = identity * a_w
#
#         return out


# class HCoordAtt(nn.Module): ##################select max pool
#     def __init__(self, inp, oup, reduction=32):  # inp: number of input channel, oup: number of output channel
#         super(HCoordAtt, self).__init__()
#         # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#         self.pool_m = nn.AdaptiveMaxPool2d((1, None))
#
#         mip = max(8, inp // reduction)
#
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#
#         # self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_m = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         identity = x
#
#         _, _, h, w = x.size()
#         # x_h = self.pool_h(x)
#         x_w = self.pool_w(x)
#         x_m = self.pool_m(x)
#
#         # print(x_w.shape)
#
#
#         x_w = self.conv1(x_w)
#         x_w = self.bn1(x_w)
#         x_w = self.act(x_w)
#
#         x_m = self.conv1(x_m)
#         x_m = self.bn1(x_m)
#         x_m = self.act(x_m)
#
#         a_w = self.conv_w(x_w).sigmoid()
#         a_m = self.conv_w(x_m).sigmoid()
#         # print(a_w.shape)
#         out = identity * a_w * a_m
#         return out

class MaxHCoordAtt(nn.Module): ##################max pool
    def __init__(self, inp, oup, reduction=32):  # inp: number of input channel, oup: number of output channel
        super(MaxHCoordAtt, self).__init__()
        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool_m = nn.AdaptiveMaxPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        # self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        # self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_m = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        _, _, h, w = x.size()
        x_m = self.pool_m(x)


        x_m = self.conv1(x_m)
        x_m = self.bn1(x_m)
        x_m = self.act(x_m)

        a_m = self.conv_m(x_m).sigmoid()
        # print(a_w.shape)
        out = identity * a_m
        return out
class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input."""

    def __init__(self, index=0):
        """Returns a particular index of the input."""
        super().__init__()
        self.index = index

    def forward(self, x):
        """
        Forward pass.

        Expects a list of tensors as input.
        """
        return x[self.index]

class WeightedSpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))

class Fusion(nn.Module):
    def __init__(self, inc_list, fusion='bifpn', c1=128) -> None:
        super().__init__()

        assert fusion in ['weight', 'adaptive', 'concat', 'bifpn', 'SChannel', 'SChannel_real','ESChannel','SChannel_new', 'GSchannel', 'SChannel_promax', 'SChannel_AK', 'SChannel_F']
        self.fusion = fusion

        if self.fusion == 'bifpn':
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        elif self.fusion == 'SChannel':
            self.sc = SelectChannel(c1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Conv2d(c1, c1, 1, 1, 0, bias=True)
            self.act = nn.ReLU()
        elif self.fusion == 'SChannel_new':
            self.sc = SelectChannel(c1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(nn.Conv2d(c1, 8, 1, 1, 0, bias=True),
                                    # nn.BatchNorm2d(8),
                                    nn.ReLU())
            self.fc1 = nn.Sequential(nn.Conv2d(8, c1, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU())
            # self.act = nn.Sigmoid()
        elif self.fusion == 'SChannel_promax':
            self.wsp = WeightedSpatialAttention(3)
            # self.hcoorattn = HCoordAtt(c1, c1, 32)
            # self.epsilon = 1e-4
            # self.pool = nn.AdaptiveAvgPool2d(1)
            # self.fc = nn.Sequential(nn.Conv2d(c1, 8, 1, 1, 0, bias=True),
            #                         # nn.BatchNorm2d(8),
            #                         nn.ReLU())
            # self.fc1 = nn.Sequential(nn.Conv2d(8, c1, kernel_size=1, stride=1, padding=0),
            #                          nn.ReLU())
            # self.act = nn.ReLU()
            self.gsc = GCT(c1)
        elif self.fusion == 'SChannel_real':
            self.sab = WeightedSpatialAttention(3)
            self.aka = AKSpatialAttention(7)
            # self.coorattn = CoordAtt(c1, c1, 32)
            self.epsilon = 1e-4
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(nn.Conv2d(c1, 8, 1, 1, 0, bias=True),
                                    # nn.BatchNorm2d(8),
                                    nn.ReLU())
            self.fc11 = nn.Sequential(nn.Conv2d(8, c1, kernel_size=1, stride=1, padding=0))
            self.fc12 = nn.Sequential(nn.Conv2d(8, c1*2, kernel_size=1, stride=1, padding=0))
            self.fc13 = nn.Sequential(nn.Conv2d(8, c1*3, kernel_size=1, stride=1, padding=0))
            self.softmax = nn.Softmax(dim=1)
            self.gate2 = GCT(c1*2)
            self.gate3 = GCT(c1 * 3)
            # self.act = nn.Sigmoid()
        elif self.fusion == 'SChannel_F':
            self.sab = WeightedSpatialAttention(3)
            self.epsilon = 1e-4
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc3 = nn.Sequential(nn.Conv2d(c1, c1 * 3, 1, 1, 0, bias=True))
            self.fc2 = nn.Sequential(nn.Conv2d(c1, c1*2, 1, 1, 0, bias=True))
            self.softmax = nn.Softmax(dim=1)
            self.gsc2 = GCT(c1*2)
            self.gsc3 = GCT(c1*3)
        elif self.fusion == 'SChannel_AK':
            self.ak = AKSpatialAttention(7)
            self.epsilon = 1e-4
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(nn.Conv2d(c1, 8, 1, 1, 0, bias=True),
                                    # nn.BatchNorm2d(8),
                                    nn.ReLU())
            self.fc1 = nn.Sequential(nn.Conv2d(8, c1, kernel_size=1, stride=1, padding=0),
                                     nn.ReLU())
            self.act = nn.ReLU()
            self.gsc = GCT(c1)

        elif self.fusion == 'ESChannel':
            self.sab = WeightedSpatialAttention(3)
            # self.epsilon = 1e-4
            # self.pool = nn.AdaptiveAvgPool2d(1)
            # self.fc3 = nn.Sequential(nn.Conv2d(c1, c1 * 3, 1, 1, 0, bias=True))
            # self.fc2 = nn.Sequential(nn.Conv2d(c1, c1 * 2, 1, 1, 0, bias=True))
            # self.softmax = nn.Softmax(dim=1)
            # self.gsc = GCT(c1)
            self.gsc2 = GCT(c1 * 2)
            self.gsc3 = GCT(c1 * 3)
        else:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])

            if self.fusion == 'adaptive':
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)
    def forward(self, x):
        # for i in range(len(x)):
            # print(f"x[{i}] has {x[i].shape[1]} channels")  # In số lượng kênh
            # print(f"x[{i}] has {x[i].shape} channels")  # In số lượng kênh

        if self.fusion in ['weight', 'adaptive']:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == 'weight':
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'concat':
            return torch.cat(x, dim=1)
        elif self.fusion == 'bifpn':
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0)+ self.epsilon)
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'SChannel':
            c1 = self.act(self.fc(self.pool(x[0])))
            c2 = self.act(self.fc(self.pool(x[1])))
            x1 = x[0] * c1
            x2 = x[1] * c2
            return x1 + x2
            # c_list = [self.act(self.fc(self.pool(x_i))) for x_i in x]  # Tạo danh sách các c tự động
            # x_weighted = [x_i * c for x_i, c in zip(x, c_list)]  # Nhân từng phần tử của x với c tương ứng
            # return sum(x_weighted)  # Cộng các giá trị lại
        elif self.fusion == 'SChannel_new':
            c_list = [self.fc1(self.fc(self.pool(x_i))) for x_i in x]  # Tạo danh sách các c tự động
            x_weighted = [x_i * c for x_i, c in zip(x, c_list)]  # Nhân từng phần tử của x với c tương ứng
            return sum(x_weighted)  # Cộng các giá trị lại
        elif self.fusion == 'SChannel_promax':
            wsp_list = [self.wsp(x_i) for x_i in x]  # Áp dụng wsp và clone từng phần tử x[i]
            # wsp_list = [self.hcoorattn(x_i) for x_i in x]
            c_list = [self.gsc(wsp_i) for wsp_i in wsp_list]  # Tạo danh sách các c tự động
            ############################################################################################################
            # wsp_list = [self.wsp(x_i) for x_i in x]  # Áp dụng wsp và clone từng phần tử x[i]
            # # c_list = [self.fc1(self.fc(self.pool(wsp_i))) for wsp_i in wsp_list]  # Tạo danh sách các c tự động
            # c_list = [self.gsc(wsp_i) for wsp_i in wsp_list]  # Tạo danh sách các c tự động
            # x_weighted1 = [wsp_i * c for wsp_i, c in zip(wsp_list, c_list)]
            # # Tính trung bình có trọng số bằng cách cộng tất cả x_weighted
            # return sum(x_weighted1)
            #######################################################################################################
            # c_list = [self.fc1(self.fc(self.pool(wsp_i))) for wsp_i in wsp_list]  # Tạo danh sách các c tự động
            # c_list = [self.gsc(x_i) for x_i in x]  # Tạo danh sách các c tự động
            # return sum(c_list)
            # output = [wsp_i + c for wsp_i, c in zip(x, c_list)]
            output = [wsp_i + c for wsp_i, c in zip(wsp_list, c_list)]
            return sum(output)
            # Tính trung bình có trọng số bằng cách cộng tất cả x_weighted

        elif self.fusion == 'SChannel_F':
            wsp_list = [self.sab(x_i) for x_i in x]  # Áp dụng wsp và clone từng phần tử x[i]
            x_weightedsum = sum(wsp_list)  #1
            z = self.pool(x_weightedsum) #1
            #########################################################################################
            # z = self.pool(torch.cat(wsp_list, dim=1))
            if (len(wsp_list) == 2):
                a_b = self.fc2(z)  ##1
                # a_b = self.softmax(a_b)
                a_b = self.gsc2(a_b)
                chunks = torch.chunk(a_b, len(wsp_list), dim=1)  # 3 chunk, mỗi chunk (batch, 128)
                output = [chunk * wsp_list[i] for i, chunk in enumerate(chunks)]  # Nhân trọng số
                return sum(output)
            elif (len(wsp_list) == 3):
                a_b = self.fc3(z)   ##1
                # a_b = self.softmax(a_b)
                a_b = self.gsc3(a_b)
                chunks = torch.chunk(a_b, len(wsp_list), dim=1)  # 3 chunk, mỗi chunk (batch, 128)
                output = [chunk * wsp_list[i] for i, chunk in enumerate(chunks)]  # Nhân trọng số
                return sum(output)
        elif self.fusion == 'SChannel_AK':
            wsp_list = [self.wsp(x_i) for x_i in x]  # Áp dụng wsp và clone từng phần tử x[i]
            # wsp_list = [wsp / (sum(torch.stack(wsp_list))+ self.epsilon) for wsp in wsp_list]
            # Nhân từng phần tử x[i] với trọng số tương ứng
            # x_weighted = sum([wsp_i * x_i for wsp_i, x_i in zip(wsp_list, x)])
            # x_weighted = sum(wsp_list)
            # c_list = [self.fc1(self.fc(self.pool(x_weighted))) for _ in x]  # Tạo danh sách các c tự động
            # x_weighted1 = [wsp_i * c for wsp_i, c in zip(wsp_list, c_list)]
            # # Tính trung bình có trọng số bằng cách cộng tất cả x_weighted
            # return sum(x_weighted1)
            #######################################################################################################
            c_list = [self.fc1(self.fc(self.pool(wsp_i))) for wsp_i in wsp_list]  # Tạo danh sách các c tự động
            # c_list = [self.gsc(wsp_i) for wsp_i in wsp_list]  # Tạo danh sách các c tự động
            x_weighted1 = [wsp_i * c for wsp_i, c in zip(wsp_list, c_list)]
            # Tính trung bình có trọng số bằng cách cộng tất cả x_weighted
            return sum(x_weighted1)
        elif self.fusion == 'SChannel_real':
            wsp_list = [self.sab(x_i) for x_i in x]  # Áp dụng wsp và clone từng phần tử x[i]
            # wsp_list = [self.coorattn(x_i) for x_i in x]  # Áp dụng wsp và clone từng phần tử x[i]
            # wsp_list = [self.aka(x_i) for x_i in x]  # Áp dụng wsp và clone từng phần tử x[i]
            # x_weighted = [wsp_i * x_i for wsp_i, x_i in zip(wsp_list, x)]
            x_weightedsum = sum(wsp_list)
            #########################################################################################
            s = self.pool(x_weightedsum)
            z = self.fc(s)
            batch = z.size(0)
            if (len(wsp_list) == 2):
                # a_b = self.fc12(z).view(batch, 2, 128, 1, 1)
                # a_b = self.softmax(a_b)
                # a_b = self.gate(a_b)
                # output = [a_b[:, i] * x_i for i, x_i in zip(range(len(a_b)), wsp_list)]
                #######################################################################################################
                a_b = self.fc12(z)
                # a_b = self.softmax(a_b)
                a_b = self.gate2(a_b)
                chunks = torch.chunk(a_b, len(wsp_list), dim=1)  # 3 chunk, mỗi chunk (batch, 128)
                output = [chunk * wsp_list[i] for i, chunk in enumerate(chunks)]  # Nhân trọng số
                # print(output.shape)
                #######################################################################################################
                # output = [a_b[:, i] * wsp_i + x_i for i, wsp_i, x_i in zip(range(len(a_b)), wsp_list, x)]
                # print(a_b[:,1].shape)
                # print(len(x_weighted), len(a_b))
                # a_b = [self.softmax(self.fc11(z)) for _ in range(len(x_weighted))]
                # output = [a_b[i] * x_i for i, x_i in zip(range(len(a_b)), x_weighted)]
                return sum(output)
            elif (len(wsp_list) == 3):
                # a_b = self.fc13(z).view(batch, 3, 128, 1, 1)
                # a_b = self.softmax(a_b)
                # # a_b = self.gate(a_b)
                # output = [a_b[:, i] * x_i for i, x_i in zip(range(len(a_b)), wsp_list)]
                # output = [a_b[:, i] * wsp_i + x_i for i, wsp_i, x_i in zip(range(len(a_b)), wsp_list, x)]
                #######################################################################################################
                a_b = self.fc13(z)
                # a_b = self.softmax(a_b)
                a_b = self.gate3(a_b)
                chunks = torch.chunk(a_b, len(wsp_list), dim=1)  # 3 chunk, mỗi chunk (batch, 128)
                output = [chunk * wsp_list[i] for i, chunk in enumerate(chunks)]  # Nhân trọng số
                # print(a_b.shape)
                #######################################################################################################
                # print(a_b[:,1].shape)
                # print(len(x_weighted), len(a_b))
                # a_b = [self.softmax(self.fc11(z)) for _ in range(len(x_weighted))]
                # output = [a_b[i] * x_i for i, x_i in zip(range(len(a_b)), x_weighted)]
                return sum(output)
            #########################################################################################
            # a_b = [self.softmax(self.fc11(self.fc(self.pool(x_weightedsum)))) for _ in range(len(x_weighted))]
            # output = [a_b[i] * x_i for i, x_i in zip(range(len(a_b)), x_weighted)]
            ##############################################################################################
            # print(a_b[1].shape)
            # print(len(x_weighted), len(a_b))
            # return sum(output)
        elif self.fusion == 'ESChannel':
            ######################################################################################################### 1
            # wsp_list = [self.sab(x_i) for x_i in x]  # Áp dụng wsp và clone từng phần tử x[i]
            # x_weightedsum = sum(wsp_list)  # 1
            # z = self.pool(x_weightedsum)  #
            # z = self.pool(torch.cat(wsp_list, dim=1))
            # if (len(x) == 2):
            #     a_b = torch.cat(x, dim=1)  ##1
            #     a_b_fr = a_b
            #     # a_b = self.softmax(a_b)
            #     a_b = self.gsc2(a_b)
            #     a_b_sa = self.sab(a_b_fr)
            #     chunks = torch.chunk(a_b, len(x), dim=1)  # 3 chunk, mỗi chunk (batch, 128)
            #     chunks_sa = torch.chunk(a_b_sa, len(x), dim=1)
            #     output = [chunk + chunk_sa for chunk, chunk_sa in zip(chunks, chunks_sa)]  # Nhân trọng số
            #     return sum(output)
            # elif (len(x) == 3):
            #     a_b = torch.cat(x, dim=1)  ##1
            #     a_b_fr = a_b
            #     a_b = self.gsc3(a_b)
            #     a_b_sa = self.sab(a_b_fr)
            #     chunks = torch.chunk(a_b, len(x), dim=1)  # 3 chunk, mỗi chunk (batch, 128)
            #     chunks_sa = torch.chunk(a_b_sa, len(x), dim=1)
            #     output = [chunk + chunk_sa for chunk, chunk_sa in zip(chunks, chunks_sa)]
            #     return sum(output)
            #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 0.988
            if (len(x) == 2):
                a_b = torch.cat(x, dim=1)  ##1
                # a_b = self.softmax(a_b)
                a_b = self.gsc2(a_b)
                chunks = torch.chunk(a_b, len(x), dim=1)  # 3 chunk, mỗi chunk (batch, 128)
                output = [chunk + self.sab(x[i]) for i, chunk in enumerate(chunks)]  # Nhân trọng số
                return sum(output)
            elif (len(x) == 3):
                a_b = torch.cat(x, dim=1)  ##1
                # a_b = self.fc3(z)  ##1
                # a_b = self.softmax(a_b)
                a_b = self.gsc3(a_b)
                chunks = torch.chunk(a_b, len(x), dim=1)  # 3 chunk, mỗi chunk (batch, 128)
                output = [chunk + self.sab(x[i]) for i, chunk in enumerate(chunks)]  # Nhân trọng số
                return sum(output)
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ 0.988
           ########################################################################################################## 2
            # wsp_list = [self.sab(x_i) for x_i in x]  # Áp dụng wsp và clone từng phần tử x[i]
            # x_weightedsum = sum(wsp_list)
            # # z = self.pool(x_weightedsum)
            # #########################################################################################
            # # z = self.pool(torch.cat(wsp_list, dim=1))
            # if (len(wsp_list) == 2):
            #     a_b = self.fc2(x_weightedsum)
            #     # a_b = self.softmax(a_b)
            #     a_b = self.gsc2(a_b)
            #     chunks = torch.chunk(a_b, len(wsp_list), dim=1)  # 3 chunk, mỗi chunk (batch, 128)
            #     output = [chunk * wsp_list[i] for i, chunk in enumerate(chunks)]  # Nhân trọng số
            #     return sum(output)
            # elif (len(wsp_list) == 3):
            #     a_b = self.fc3(x_weightedsum)
            #     # a_b = self.fc3(z)  ##1
            #     # a_b = self.softmax(a_b)
            #     a_b = self.gsc3(a_b)
            #     chunks = torch.chunk(a_b, len(wsp_list), dim=1)  # 3 chunk, mỗi chunk (batch, 128)
            #     output = [chunk * wsp_list[i] for i, chunk in enumerate(chunks)]  # Nhân trọng số
            #     return sum(output)
            ########################################################################################################### 3
            # wsp_list = [self.sab(x_i) for x_i in x]  # Áp dụng wsp và clone từng phần tử x[i]
            # x_weightedsum = sum(wsp_list)
            # gsc_list = [self.gsc(x_weightedsum) for _ in wsp_list]
            # x_gated = [gsc_i * wsp_i for gsc_i, wsp_i in zip(gsc_list, wsp_list)]
            # output = [gsc_i + x_i for gsc_i, x_i in zip(x_gated, x)]
            # return sum(output)

# class Fusion1(nn.Module):
#     def __init__(self, inc_list, fusion='bifpn') -> None:
#         super().__init__()
#
#         assert fusion in ['weight', 'adaptive', 'concat', 'bifpn', 'new']
#         self.fusion = fusion
#
#         if self.fusion == 'bifpn':
#             self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
#             self.relu = nn.ReLU()
#             self.epsilon = 1e-4
#         else:
#             self.pool = nn.AdaptiveAvgPool2d(1)
#             self.act = nn.Sigmoid()
#             self.fusion_conv = nn.ModuleList([nn.Conv2d(inc, inc, 1, 1, 0, bias=True) for inc in inc_list])
#             # self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])
#             if self.fusion == 'adaptive':
#                 self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)
#
#     def forward(self, x):
#         if self.fusion in ['weight', 'adaptive', 'new']:
#             for i in range(len(x)):
#                 x[i] = self.fusion_conv[i](x[i])
#         if self.fusion == 'weight':
#             return torch.sum(torch.stack(x, dim=0), dim=0)
#         elif self.fusion == 'adaptive':
#             fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
#             x_weight = torch.split(fusion, [1] * len(x), dim=1)
#             return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
#         elif self.fusion == 'new':
#             fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
#             x_weight = torch.split(fusion, [1] * len(x), dim=1)
#             return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
#         elif self.fusion == 'concat':
#             return torch.cat(x, dim=1)
#         elif self.fusion == 'bifpn':
#             fusion_weight = self.relu(self.fusion_weight.clone())
#             fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0))
#
#             return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
class SelectChannel(nn.Module):
    def __init__(self, c1):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(c1, c1, 1, 1, 0, bias=True)
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU()
    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        c1 = self.act(self.fc(self.pool(x[0])))
        c2 = self.act(self.fc(self.pool(x[1])))

        x1 = x[0] * c1
        x2 = x[1] * c2
        return x1+x2

class BiFPN_WConcat(nn.Module):
    def __init__(self, inc_list, dimension=1):
        super(BiFPN_WConcat, self).__init__()
        self.d = dimension
        # self.relu = nn.SiLU()
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
    def forward(self, x):
        # w = self.relu(self.w.clone())
        w = self.w.clone()
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion [fusion_weight[i] * x[i] for i in range(len(x))]
        x = [weight[i] * x[i] for i in range(len(x))]
        # return torch.cat(x, self.d)
        return channel_shuffle(torch.cat(x, self.d),4)

class BiFPN_WConcat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_WConcat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.relu = nn.ReLU()
        # self.relu = nn.Sigmoid()
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.relu(self.w.clone())
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        # return torch.cat(x, self.d)
        return channel_shuffle(torch.cat(x, self.d), 4)

class BiFPN_WConcat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_WConcat3, self).__init__()
        self.d = dimension
        self.relu = nn.ReLU()
        # self.relu = nn.Sigmoid()
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.relu(self.w.clone())
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        # return torch.cat(x, self.d)
        return channel_shuffle(torch.cat(x, self.d),4)
def channel_shuffle(x, groups=2):  ##shuffle channel
    # RESHAPE----->transpose------->Flatten
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out

import torch.nn.functional as F
from timm.layers.create_act import create_act_layer, get_act_layer
from timm.layers.helpers import make_divisible
from timm.layers.mlp import ConvMlp
from timm.layers.norm import LayerNorm2d


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
        norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        gate = 1. + torch.tanh(embedding * norm + self.beta)
        # print('cool')
        return x * gate
        # return gate
class GlobalContext(nn.Module):

    def __init__(self, channels, use_attn=True, fuse_add=False, fuse_scale=True, init_last_zero=False,
                 rd_ratio=1. / 8, rd_channels=None, rd_divisor=1, act_layer=nn.ReLU, gate_layer='sigmoid'):
        super(GlobalContext, self).__init__()
        act_layer = get_act_layer(act_layer)

        self.conv_attn = nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None

        if rd_channels is None:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        if fuse_add:
            self.mlp_add = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_add = None
        if fuse_scale:
            self.mlp_scale = ConvMlp(channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d)
        else:
            self.mlp_scale = None

        self.gate = create_act_layer(gate_layer)
        self.init_last_zero = init_last_zero
        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_attn is not None:
            nn.init.kaiming_normal_(self.conv_attn.weight, mode='fan_in', nonlinearity='relu')
        if self.mlp_add is not None:
            nn.init.zeros_(self.mlp_add.fc2.weight)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)  # (B, 1, H * W)
            attn = F.softmax(attn, dim=-1).unsqueeze(3)  # (B, 1, H * W, 1)
            context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean(dim=(2, 3), keepdim=True)

        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x

        return x
class BottleNect(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 63
        pad = ker // 2
        self.in_conv = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
                    nn.GELU()
                    )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        # self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,pad), stride=1, groups=dim)
        # self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(pad,0), stride=1, groups=dim)
        # self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=3, padding=3//2, stride=1, groups=dim)
        self.sa = SpatialAttention(3)
        self.act = nn.ReLU()

        ### sca ###
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        ### fca ###
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)

        ### fca ###
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        # print(x_fft.dtype)  # Kiểm tra kiểu dữ liệu sau FFT
        # print(x_att.dtype)
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2,-1), norm='backward')
        x_fca = torch.abs(x_fca)

        ### fca ###
        ### sca ###
        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca
        ### sca ###
        x_sca = self.fgm(x_sca)

        # out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        # out = self.act(x_sca) + self.sa(out)
        out = self.act(x_sca) + x
        # out = self.act(out)
        # print(x.dtype)  # Kiểm tra kiểu dữ liệu đầu vào
        #
        # print(x_fca.dtype)  # Kiểm tra kiểu dữ liệu sau khi lấy abs

        # print("omn")
        return out
        # return self.out_conv(out)
class FGM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1)

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        # res = x.clone()
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta
from timm.models.layers import DropPath
class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        # self.dim_conv3 = dim // n_div
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        # self.dim_conv3_1 = self.dim_conv3 // 2
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.partial_conv3_1 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1,  1, dilation=1, bias=False)
        self.partial_conv3_2 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, padding= 5//2, dilation=2, bias=False)

        # self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, dilation=1, bias=False)
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        # x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        # # x1, x2 = torch.split(x, [self.partial_conv3.weight.size(1), x.size(1) - self.partial_conv3.weight.size(1)],
        # #                      dim=1)
        # x1 = self.partial_conv3(x1)
        # x = torch.cat((x1, x2), 1)
        ######################################
        x1, x2 = torch.split(x, [self.partial_conv3.weight.size(1), x.size(1) - self.partial_conv3.weight.size(1)],
                             dim=1)
        x1_1 = self.partial_conv3_1(x1)
        x1_2 = self.partial_conv3_2(x1_1)
        x = torch.cat((x1_1, x1_2, x2), 1)
        return x
class Faster_Block(nn.Module):
    def __init__(self,
                 inc,   #input
                 dim,   # output
                 n_div= 4, #divide partical conv #original = 4
                 mlp_ratio=2,
                 drop_path=0.1, #regularization
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div
        self.dim_div = (dim//n_div) * 2 + (dim-dim//n_div)
        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(self.dim_div, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ] # block after PConv

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        ) #PConv or call spatial_mixing

        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None: #only for input != output
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x