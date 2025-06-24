# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, ConvE, DWConv, GhostConv, LightConv, ConvGN, RepConv, autopad, LDConv, LDConv1, AKCBAM, HCoordAtt, GlobalContext, MSCSpatialAttention, GCT, Faster_Block, CBAM, SpatialAttention, ChannelAttention
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "SAPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "Cross_AKConv",
    "BottleneckCSP",
    "BottleneckX_CBam",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C3MSCk2",
    "XCBAM2C2f",
    "C3k2GC",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    "S2CrossConvDilated",
    "Mix_SPPF",
    "SELayer",
    "SaELayer",
)

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        #self.m = nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(3 // 2, 0))
    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class SaELayer(nn.Module):
    def __init__(self, in_channel, reduction=32):
        super(SaELayer, self).__init__()
        assert in_channel>=reduction and in_channel%reduction==0,'invalid in_channel in SaElayer'
        self.reduction = reduction
        self.cardinality=4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #cardinality 1
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel,in_channel//self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 2
        self.fc2 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 3
        self.fc3 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 4
        self.fc4 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_channel//self.reduction*self.cardinality, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y1 = self.fc1(y)
        y2 = self.fc2(y)
        y3 = self.fc3(y)
        y4 = self.fc4(y)
        y_concate = torch.cat([y1,y2,y3,y4],dim=1)
        y_ex_dim = self.fc(y_concate).view(b,c,1,1)

        return x * y_ex_dim.expand_as(x)
    
 ##############################################################   
# class SPPF(nn.Module):
#     """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

#     def __init__(self, c1, c2, k=5):
#         """
#         Initializes the SPPF layer with given input/output channels and kernel size.
#         This module is equivalent to SPP(k=(5, 9, 13)).
#         """
#         super().__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = ConvE(c1, c_, 1, 1)  
#         self.se_layer = SELayer(c_, reduction=16)  # SELayer
#         self.cv2 = ConvE(c_ * 4, c2, 1, 1)  
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.conv1x1_pooled = ConvE(c_, c_, 1, 1)  # 1x1 Conv cho pooled outputs

#     def forward(self, x):
#         """Forward pass through SPPF layer."""
#         # Áp dụng cv1 lên input
#         x_cv1 = self.cv1(x)  # Đầu ra của cv1, shape: [batch_size, c_, H, W]
        
#         # Nhánh cho concatenation: Áp dụng SELayer lên x_cv1
#         y_se = self.se_layer(x_cv1)  # Tensor sẽ dùng để concatenate
        
#         # Nhánh cho max-pooling: Sử dụng trực tiếp x_cv1 (không qua SELayer)
#         y = [y_se]  # Khởi tạo danh sách với tensor đã qua SELayer cho concatenation
#         y.extend(self.m(x_cv1) for _ in range(3))  # Áp dụng max-pooling 3 lần trên x_cv1
        
#         # Áp dụng conv1x1_pooled và SELayer cho tất cả các tensor
#         y = [self.conv1x1_pooled(l) for l in y]  # Áp dụng 1x1 Conv
#         y = [self.se_layer(l) for l in y]  # Áp dụng SELayer
        
#         # Concatenate các tensor
#         y = torch.cat(y, 1)  # Kết hợp 4 tensor, tạo tensor có 4*c_ kênh
        
#         # Đưa qua cv2
#         return self.cv2(y)
 ##############################################################   


##############################################################

# class SPPF(nn.Module):
#     """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

#     def __init__(self, c1, c2, k=5):
#         """
#         Initializes the SPPF layer with given input/output channels and kernel size.

#         This module is equivalent to SPP(k=(5, 9, 13)).
#         """
#         super().__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)  
#         self.se_layer = SELayer(c_, reduction=16)  # Thêm SELayer
#         self.cv2 = Conv(c_ * 4, c2, 1, 1)  
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.conv1x1_pooled = Conv(c_, c_, 1, 1)  # Lớp 1x1 Conv cho pooled outputs


#     def forward(self, x):
#         """Forward pass through SPPF layer."""
#         y = [self.cv1(x)]
#         y = self.se_layer(y[0])  # Áp dụng SELayer vào đầu ra của cv1
#         y = [y]  # Đặt lại y thành danh sách để có thể sử dụng extend
#         y.extend(self.m(y[-1]) for _ in range(3))  # Thực hiện MaxPool
        
#         #y = [self.se_layer(l) for l in y]  # Áp dụng SELayer
#         # Áp dụng 1x1 Conv và SELayer cho các đầu ra từ MaxPooling
#         y = [self.conv1x1_pooled(l) for l in y]  # Áp dụng 1x1 Conv
#         y = [self.se_layer(l) for l in y]  # Áp dụng SELayer
        
#         # Kết hợp đầu ra của cv1 đã qua SELayer với các đầu ra từ MaxPooling
#         y = torch.cat(y, 1)  # Kết hợp đầu ra của cv1 và các đầu ra từ MaxPooling
        
#         return self.cv2(y)  # Trả về đầu ra cuối cùng



class Mix_SPPF(nn.Module):
    def __init__(self, c1, c2, k=5, dropout_rate=0.3):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)  
        
        # Thêm SELayer sau 1x1 conv đầu tiên
        self.se_layer_after_cv1 = SELayer(c_)  # SELayer sau cv1
        
        # Thêm 1x1 conv trước SELayer
        self.conv1x1_se = Conv(c_, c_, 1, 1)  # 1x1 conv trước SELayer
        self.se_layer = SELayer(c_)  # SELayer
        
        self.dwconv = Conv(c_, c_, 3, 1, g=c_)  # 3x3 depthwise conv
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1)  # Average pooling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)  # Max pooling
        self.dropout = nn.Dropout(dropout_rate)  # Thêm Dropout
        self.conv1x1_final = Conv(c_ * 2, c2, 1, 1)  # 1x1 conv cuối cùng
        
        # Thêm 1x1 conv mới sau nhánh 1
        self.conv1x1_after_branch1 = Conv(c_, c_, 1, 1)  # 1x1 conv mới

    def forward(self, x):
        x = self.cv1(x)  # Thực hiện 1x1 conv đầu tiên
        x = self.se_layer_after_cv1(x)  # Áp dụng SELayer sau cv1
        
        # Nhánh 1: Thêm 1x1 conv, sau đó là 1x1 conv mới và depthwise conv
        branch1 = self.conv1x1_se(x)  # 1x1 conv trước SELayer
        branch1 = self.conv1x1_after_branch1(branch1)  # Áp dụng 1x1 conv mới
        branch1 = self.dwconv(branch1)  # Nhánh 1: 3x3 depthwise conv

        # Nhánh 2: Average pooling và Max pooling
        avg = self.avgpool(x)  # Average pooling
        branch2 = self.maxpool(avg)  # Max pooling

        # Kết hợp các nhánh
        branch2 = nn.functional.interpolate(branch2, size=branch1.shape[2:], mode='bilinear', align_corners=False)
        combined = torch.cat((branch1, branch2), dim=1)  # Kết hợp các nhánh

        combined = self.dropout(combined)  # Áp dụng Dropout
        return self.conv1x1_final(combined) 



class SAPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer with Dilated Convolutions."""

    def __init__(self, c1, c2, k=3):
        """Initializes the SPPF layer with dilated convolutions."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 3, c2, 1, 1)

        # Dilated convolutions with different dilation rates
        self.dilated_conv1 = Conv(c_, c_, k, 1, d=1)
        self.dilated_conv2 = Conv(c_, c_, k, 1, d=2)
        # self.dilated_conv3 = Conv(c_, c_, k, 1, d=10)

    def forward(self, x):
        """Forward pass through the modified SPPF block."""
        y = [self.cv1(x)]
        y.append(self.dilated_conv1(y[-1]))
        y.append(self.dilated_conv2(y[-1]))
        # y.append(self.dilated_conv3(y[-1]))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        # self.m = nn.ModuleList(LightBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.n = n
        # self.m = nn.ModuleList(Cross_AKConv(self.c, self.c, shortcut, e=1.0) for _ in range(n))
        # self.m = nn.ModuleList(AKCBAM(self.c) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        # print(self.n)
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class XCBAM2C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.n = n
        self.m = nn.ModuleList(
            nn.Sequential(*(BottleneckX_CBam(c_, c_, shortcut, g, k=3, e=1.0) for _ in range(n)))
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return x + self.cv2(torch.cat(y, 1))
class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
class C3GC(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))
        self.gc = GlobalContext(c_)
    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        # print(('gc'))
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.gc(self.cv2(x))), 1))

class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = MSCSpatialAttention(self.c_)


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))
class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)
class BottleneckX_CBam(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # self.cv1 = Faster_Block(c1, c_)
        # self.cv4 = Faster_Block(c_, c2)
        # self.cv1 = CBAM(c1, 7)
        # self.cv1 = AKCBAM(c1, 3)
        # self.cv1 = SpatialAttention(7)
        # self.cv2 = Conv(c1, c_, 1, 1, g=1)
        # self.cv3 = Conv(c_, c2, 1, 1, g=1)
        # self.cv4 = SpatialAttention(7)
        self.cv1 = Faster_Block(c1, c2)
        # self.cv2 = Faster_Block(c_, c2)
        # self.cv4 = CBAM(c2, 3)
        # self.cv4 = AKCBAM(c2, 3)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        # print('y')
        # return x + self.cv4(self.cv1(x)) if self.add else self.self.cv4(self.cv1(x))
        return x + self.cv1(x) if self.add else self.cv1(x)
        # return x + self.cv4(self.cv3(self.cv2(self.cv1(x)))) if self.add else self.cv4(self.cv3(self.cv2(self.cv1(x))))

class BottleneckX_PConv(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # self.cv1 = Faster_Block(c1, c1)
        self.cv1 = Conv(c1, c1, 3, 1, g=8)
        self.cv2 = Conv(c1, c_, 1, 1, g=g)
        self.cv3 = Conv(c_, c2, 1, 1, g=g)
        # self.cv4 = Faster_Block(c2, c2)
        self.cv4 = Conv(c2, c2, 3, 1, g=8)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        # print('y')
        return x + self.cv4(self.cv3(self.cv2(self.cv1(x)))) if self.add else self.cv4(self.cv3(self.cv2(self.cv1(x))))
class LightBottleneck(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c1, k[0], 1, g=1)
        # self.cv2 = Conv(c1//2, c1 // 2, k[1], 1, g=1)
        self.add = shortcut and c1 == c2
        self.pool = nn.AdaptiveAvgPool2d(1)
        mip = max(8, 2* c1 // 32)
        self.conv1 = nn.Sequential(nn.Conv2d(2* c1, mip, kernel_size=1, stride=1, padding=0), nn.ReLU())
        self.act = h_swish()
        self.conv2 = nn.Conv2d(mip, 2* c1, kernel_size=1, stride=1, padding=0)
        self.conv3 = Conv(2* c1, c2, 1, 1, g=8)
    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        x1 = self.cv1(x)
        # x2 = self.cv2(x1)
        z = torch.cat((x, x1),1)
        y = self.pool(z)
        y = self.conv1(y)
        a_y = self.act(self.conv2(y))
        out = z * a_y
        out = self.conv3(out)
        return x + out if self.add else out
        # return x + self.fc1(self.fc(self.pool(torch.cat((x1, x2),1)))) if self.add else self.fc1(self.fc(self.pool(torch.cat((x1, x2),1))))
class SLBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(Conv(c1, c_, 1, 1), Conv(c_, c_, k[0], 1))
        self.cv2 = nn.Sequential(Conv(c_, c_, k[1], 1), Conv(c_, c2, 1, 1))
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
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
class LightBottleneck1(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c1//2, k[0], 1, g=1)
        self.cv2 = Conv(c1//2, c1 // 2, k[1], 1, g=1)
        self.add = shortcut and c1 == c2
        self.pool = nn.AdaptiveAvgPool2d(1)
        mip = max(8, 2*c1 // 32)
        self.conv1 = nn.Sequential(nn.Conv2d(2*c1, mip, kernel_size=1, stride=1, padding=0), nn.ReLU())
        self.act = h_swish()
        self.conv2 = nn.Conv2d(mip, 2*c1, kernel_size=1, stride=1, padding=0)
        self.conv3 = Conv(2*c1, c2, 1, 1, g=8)
    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        z = torch.cat((x, x1, x2),1)
        y = self.pool(z)
        y = self.conv1(y)
        a_y = self.act(self.conv2(y))
        out = z * a_y
        out = self.conv3(out)
        return x + out if self.add else out
        # return x + self.fc1(self.fc(self.pool(torch.cat((x1, x2),1)))) if self.add else self.fc1(self.fc(self.pool(torch.cat((x1, x2),1))))
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
# class BottleNect(nn.Module):
#     def __init__(self, dim) -> None:
#         super().__init__()
#
#         ker = 63
#         pad = ker // 2
#         self.in_conv = nn.Sequential(
#                     nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
#                     nn.GELU()
#                     )
#         self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
#         self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,pad), stride=1, groups=dim)
#         self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(pad,0), stride=1, groups=dim)
#         self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
#         self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)
#
#         self.act = nn.ReLU()
#
#         ### sca ###
#         self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#         self.pool = nn.AdaptiveAvgPool2d((1,1))
#
#         ### fca ###
#         self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
#         self.fac_pool = nn.AdaptiveAvgPool2d((1,1))
#         self.fgm = FGM(dim)
#
#     def forward(self, x):
#         out = self.in_conv(x)
#
#         ### fca ###
#         x_att = self.fac_conv(self.fac_pool(out))
#         x_fft = torch.fft.fft2(out, norm='backward')
#         # print(x_fft.dtype)  # Kiểm tra kiểu dữ liệu sau FFT
#         # print(x_att.dtype)
#         x_fft = x_att * x_fft
#         x_fca = torch.fft.ifft2(x_fft, dim=(-2,-1), norm='backward')
#         x_fca = torch.abs(x_fca)
#
#         ### fca ###
#         ### sca ###
#         x_att = self.conv(self.pool(x_fca))
#         x_sca = x_att * x_fca
#         ### sca ###
#         x_sca = self.fgm(x_sca)
#
#         # out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
#         out = x + self.dw_11(out) + x_sca
#         out = self.act(out)
#         # print(x.dtype)  # Kiểm tra kiểu dữ liệu đầu vào
#         #
#         # print(x_fca.dtype)  # Kiểm tra kiểu dữ liệu sau khi lấy abs
#
#         # print("omn")
#         return self.out_conv(out)
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
        out = self.act(x_sca)
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
class MSBottleneck(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.msc = MSCSpatialAttention(c2)
    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        # print("MSC")
        # return x + self.msc(self.cv2(self.cv1(x))) if self.add else self.msc(self.cv2(self.cv1(x)))
        return x + self.cv2(self.cv1(self.msc(x))) if self.add else self.cv2(self.cv1(self.msc(x)))
class Cross_AKConv(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = LDConv(c1, c_, 5, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class DualChannelHConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, shortcut=False, g=1, k=3, e=0.5, s=1 ):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1_1 = Conv(c1, c2, (1, 1), 1)
        self.cv3_1 = Conv(c1, c_, (3, 1), 1)
        self.cv7_1 = Conv(c1, c_, (3, 1), 1)
        self.se = nn.Sequential(
            Conv(2 * c_, c_, 1, act=nn.ReLU()),
            Conv(c_, c2, 1, act=nn.ReLU()))
        self.hca = HCoordAtt(c2, c2,32)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""


        x3 = self.cv3_1(x)

        x7 = self.cv7_1(x)

        cat = torch.cat((x3, x7), dim=1)
        output = self.se(cat)
        output = self.hca(output)
        return x + output if self.add else output
class CrossConvDilated(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, shortcut=False, g=1, k=3, e=0.5, s=1 ):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1_1 = Conv(c1, c_, (1, k), (1, s), d=1)
        self.cv1_2 = Conv(c_, c2, (k, 1), (s, 1), d=1)
        self.cv2_1 = Conv(c1, c_, (1, k), (1, s), d=2)
        self.cv2_2 = Conv(c_, c2, (k, 1), (s, 1), d=2)
        self.cv5_1 = Conv(c1, c_, (1, k), (1, s), d=3)
        self.cv5_2 = Conv(c_, c2, (k, 1), (s, 1), d=3)
        self.squeeze = Conv(3 * c2, c2, 1)
        self.fc = nn.Sequential(nn.Conv2d(3 * c2, max(3*c2/32,8), 1, 1, 0, bias=True),
                                # nn.BatchNorm2d(8),
                                nn.ReLU())
        self.fc1 = nn.Sequential(nn.Conv2d(max(3*c2/32,8), c2, kernel_size=1, stride=1, padding=0),
                                 nn.ReLU())
        self.add = shortcut and c1 == c2
    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""
        x1 = self.cv1_1(x)
        x1 = self.cv1_2(x1)
        x2 = self.cv2_1(x)
        x2 = self.cv2_2(x2)

        x5 = self.cv5_1(x)
        x5 = self.cv5_2(x5)

        output = torch.cat((x1, x2, x5), dim=1)
        # output = channel_shuffle(output, 4)
        output = self.fc1(self.fc(output))

        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        return x + output if self.add else output
class S2DenseCrossConvDilated(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, shortcut=False, g=1, k=3, e=0.5, s=1 ):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # self.cv1_1 = Conv(c1, c_, (1, k), (1, s), d=1)
        # self.cv1_2 = Conv(c_, c2, (k, 1), (s, 1), d=1)
        # self.cv2_1 = Conv(c1, c_, (1, k), (1, s), d=2)
        # self.cv2_2 = Conv(c_, c2, (k, 1), (s, 1), d=2)
        self.cv1_1 = Conv(c1, c_, (k, 1), (1, s), d=1)
        self.cv1_2 = Conv(c_, c2, (1, k), (s, 1), d=1)
        self.cv2_1 = Conv(c1, c_, (k, 1), (1, s), d=2)
        self.cv2_2 = Conv(c_, c2, (1, k), (s, 1), d=2)
        self.squeeze = Conv(3 * c2, c2, 1)
        self.add = shortcut and c1 == c2

        self.gct = GCT(c2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(c1, 8, 1, 1, 0, bias=True),
                                # nn.BatchNorm2d(8),
                                nn.ReLU())
        self.fc1 = nn.Sequential(nn.Conv2d(8, c2, kernel_size=1, stride=1, padding=0),
                                 nn.ReLU())
        self.act = nn.ReLU()
        # self.act = nn.Sigmoid()
    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""
        x1_1 = self.cv1_1(x)
        x1_2 = self.cv1_2(x1_1)
        x2_1 = self.cv2_1(x1_2)
        x2_2 = self.cv2_2(x2_1)

        x6 = x1_2 + x2_2

        x7 = self.pool(x6)
        c1 = self.fc1(self.fc(x7))
        c2 = self.fc1(self.fc(x7))


        x1_o = x1_2 * c1
        x2_o = x2_2 * c2
        # x1_1 = self.gct(x1)
        # x2_1 = self.gct(x2)

        output = x1_o + x2_o
        # print('msc')
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        return x + output if self.add else output
class S2CrossConvDilated(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, shortcut=False, g=1, k=3, e=0.5, s=1 ):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # self.cv1_1 = Conv(c1, c_, (1, k), (1, s), d=1)
        # self.cv1_2 = Conv(c_, c2, (k, 1), (s, 1), d=1)
        # self.cv2_1 = Conv(c1, c_, (1, k), (1, s), d=2)
        # self.cv2_2 = Conv(c_, c2, (k, 1), (s, 1), d=2)
        self.cv1_1 = Conv(c1, c_, (k, 1), (1, s), d=1)
        self.cv1_2 = Conv(c_, c2, (1, k), (s, 1), d=1)
        self.cv2_1 = Conv(c1, c_, (k, 1), (1, s), d=2)
        self.cv2_2 = Conv(c_, c2, (1, k), (s, 1), d=2)
        self.squeeze = Conv(3 * c2, c2, 1)
        self.add = shortcut and c1 == c2

        self.gct = GCT(c2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(c1, 8, 1, 1, 0, bias=True),
                                nn.BatchNorm2d(8),
                                nn.ReLU())
        self.fc1 = nn.Sequential(nn.Conv2d(8, c2, kernel_size=1, stride=1, padding=0),
                                 # nn.ReLU()
                                 nn.Softmax())
        # self.act = nn.ReLU()
        # self.act = nn.Sigmoid()
    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""
        x1 = self.cv1_1(x)
        x1 = self.cv1_2(x1)
        x2 = self.cv2_1(x)
        x2 = self.cv2_2(x2)

        x6 = x1 + x2

        x7 = self.pool(x6)
        c1 = self.fc1(self.fc(x7))
        c2 = self.fc1(self.fc(x7))


        x1_1 = x1 * c1
        x2_1 = x2 * c2
        # x1_1 = self.gct(x1)
        # x2_1 = self.gct(x2)

        output = x1_1 + x2_1
        # print('msc')
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        return x + output if self.add else output
class DenseCross(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=0.5, s=1 ):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1_1 = Conv(c1, c_, (k, 1), (1, 1), d=1)
        self.cv1_2 = Conv(c_, c2, (1, k), (1, 1), d=1)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""
        batch, channels, _, _ = x.shape
        # Multi-branch feature extraction
        x1_1 = self.cv1_1(x)
        x1_2 = self.cv1_2(x1_1+x)
        x1 = x1_1 + x1_2

        return x + x1 if self.add else x1
class S2CrossConvDilatedDenseReal(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, shortcut=False, g=1, k=3, e=0.5, s=1 ):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # self.cv1_1 = Conv(c1, c_, (1, k), (1, s), d=1)
        # self.cv1_2 = Conv(c_, c2, (k, 1), (s, 1), d=1)
        # self.cv2_1 = Conv(c1, c_, (1, k), (1, s), d=2)
        # self.cv2_2 = Conv(c_, c2, (k, 1), (s, 1), d=2)
        self.cv1_1 = Conv(c1, c_, (k, 1), (1, s), d=1)
        self.cv1_2 = Conv(c_, c2, (1, k), (s, 1), d=1)
        self.cv2_1 = Conv(c1, c_, (k, 1), (1, s), d=2)
        self.cv2_2 = Conv(c_, c2, (1, k), (s, 1), d=2)
        self.squeeze = Conv(3 * c2, c2, 1)
        self.add = shortcut and c1 == c2

        self.gct = GCT(c2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(c1, 8, 1, 1, 0, bias=True),
                                # nn.BatchNorm2d(8),
                                nn.ReLU())
        self.fc1 = nn.Sequential(nn.Conv2d(8, c2*2, kernel_size=1, stride=1, padding=0))
                                 # nn.ReLU()
                                 # nn.Softmax(dim=1))

        self.softmax = nn.Softmax(dim=1)
        # self.act = nn.ReLU()
        # self.act = nn.Sigmoid()
    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""
        batch, channels, _, _ = x.shape
        # print('1')
        # Multi-branch feature extraction
        x1 = self.cv1_1(x)
        x1 = self.cv1_2(x1)
        x2 = self.cv2_1(x1)
        x2 = self.cv2_2(x2)

        u = x1 + x2

        # Channel-wise attention

        s = self.pool(u)
        z = self.fc(s)
        batch = z.size(0)
        a_b = self.fc1(z).view(batch, 2, channels, 1, 1)
        a_b = self.softmax(a_b)

        # Selective feature reweighting
        output = a_b[:, 0] * x1 + a_b[:, 1] * x2
        return x + output if self.add else output
class S2CrossConvDilatedReal(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, shortcut=False, g=1, k=3, e=0.5, s=1 ):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # self.cv1_1 = Conv(c1, c_, (1, k), (1, s), d=1)
        # self.cv1_2 = Conv(c_, c2, (k, 1), (s, 1), d=1)
        # self.cv2_1 = Conv(c1, c_, (1, k), (1, s), d=2)
        # self.cv2_2 = Conv(c_, c2, (k, 1), (s, 1), d=2)
        self.cv1_1 = Conv(c1, c_, (k, 1), (1, s), d=1)
        self.cv1_2 = Conv(c_, c2, (1, k), (s, 1), d=1)
        self.cv2_1 = Conv(c1, c_, (k, 1), (1, s), d=2)
        self.cv2_2 = Conv(c_, c2, (1, k), (s, 1), d=2)
        self.squeeze = Conv(3 * c2, c2, 1)
        self.add = shortcut and c1 == c2

        self.gct = GCT(c2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(c1, 8, 1, 1, 0, bias=True),
                                # nn.BatchNorm2d(8),
                                nn.ReLU())
        self.fc1 = nn.Sequential(nn.Conv2d(8, c2*2, kernel_size=1, stride=1, padding=0))
                                 # nn.ReLU()
                                 # nn.Softmax(dim=1))

        self.softmax = nn.Softmax(dim=1)
        # self.act = nn.ReLU()
        # self.act = nn.Sigmoid()
    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""
        batch, channels, _, _ = x.shape

        # Multi-branch feature extraction
        x1 = self.cv1_1(x)
        x1 = self.cv1_2(x1)
        x2 = self.cv2_1(x)
        x2 = self.cv2_2(x2)

        u = x1 + x2

        # Channel-wise attention

        s = self.pool(u)
        z = self.fc(s)
        batch = z.size(0)
        a_b = self.fc1(z).view(batch, 2, channels, 1, 1)
        a_b = self.softmax(a_b)

        # Selective feature reweighting
        output = a_b[:, 0] * x1 + a_b[:, 1] * x2
        return x + output if self.add else output
class SCrossConvDilated(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, shortcut=False, g=1, k=3, e=0.5, s=1 ):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1_1 = Conv(c1, c_, (1, k), (1, s), d=1)
        self.cv1_2 = Conv(c_, c2, (k, 1), (s, 1), d=1)
        self.cv2_1 = Conv(c1, c_, (1, k), (1, s), d=2)
        self.cv2_2 = Conv(c_, c2, (k, 1), (s, 1), d=2)
        self.cv5_1 = Conv(c1, c_, (1, k), (1, s), d=5)
        self.cv5_2 = Conv(c_, c2, (k, 1), (s, 1), d=5)
        self.squeeze = Conv(3 * c2, c2, 1)
        self.add = shortcut and c1 == c2

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(c1, c1, 1, 1, 0, bias=True)
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU()
    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""
        x1 = self.cv1_1(x)
        x1 = self.cv1_2(x1)

        x2 = self.cv2_1(x)
        x2 = self.cv2_2(x2)

        x5 = self.cv5_1(x)
        x5 = self.cv5_2(x5)

        x6 = x1+x2+x5

        x7 = self.pool(x6)
        c1 = self.act(self.fc(x7))
        c2 = self.act(self.fc(x7))
        c5 = self.act(self.fc(x7))

        x1_1 = x1 * c1
        x2_1 = x2 * c2
        x5_1 = x5 * c5

        output = x1_1 + x2_1 + x5_1

        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        return x + output if self.add else output
class RepCross(torch.nn.Module):
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False) -> None:
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv5 = Conv(c1, c_, (1, k), (1, s))
        self.cv6 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        # self.conv = self.cv2(self.cv1)
        # self.cv3 = Conv(c1, c_, (1, 5), (1, s),padding=(0, 5//2))
        # self.cv4 = Conv(c_, c2, (5, 1), (s, 1), g=g,padding=(5//2,0))
        self.cv1 = Conv(c1, c_, (1, 7), (1, s))
        self.cv2 = Conv(c_, c2, (7, 1), (s, 1), g=g)
        # self.conv1 = self.cv5(self.cv6)
        self.add = shortcut and c1 == c2
        # self.act = nn.SiLU()

    def forward(self, x):
        attn_0 = self.cv1(x)
        attn_2 = self.cv5(x)
        x0 = x + attn_0 + attn_2
        attn_0 = self.cv2(x0)
        attn_2 = self.cv6(x0)
        # attn_1 = self.cv3(x)
        # attn_1 = self.cv4(attn_1)
        attn = x0 + attn_0 + attn_2
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        return x + attn if self.add else attn
class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            # C3k(self.c, self.c, 2, shortcut, g) if c3k else RepBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
            # C3k(self.c, self.c, 2, shortcut, g) if c3k else LightBottleneck1(self.c, self.c, shortcut, g) for _ in range(n)
            # C3k(self.c, self.c, 2, shortcut, g) if c3k else BottleneckX_CBam(self.c, self.c, shortcut, g) for _ in range(n)
            # C3k(self.c, self.c, 2, shortcut, g) if c3k else AKCBAM(self.c) for _ in range(n)
        )
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*(DenseCross(c_, c_, shortcut, g, k=k, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*(BottleneckX_CBam(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*(LightBottleneck1(c_, c_, shortcut, g, k=((k, 1), (1, k)), e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*(LightBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# class C3k2GC(C2f):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""
#     def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
#         """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(
#             # C3k(self.c, self.c, 2, shortcut, g) if c3k else Cross_AKConv(self.c, self.c, shortcut, g) for _ in range(n)
#             C3kGC(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
#             # C3k(self.c, self.c, 2, shortcut, g) if c3k else AKCBAM(self.c) for _ in range(n)
#         )
# class C3kGC(C3GC):
#     """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
#
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
#         """Initializes the C3k module with specified channels, number of layers, and configurations."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
#         self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
#         # self.m = nn.Sequential(*(Cross_AKConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
class C3k2GC(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            # C3k(self.c, self.c, 2, shortcut, g) if c3k else Cross_AKConv(self.c, self.c, shortcut, g) for _ in range(n)
            C3kGC(self.c, self.c, 2, shortcut, g) if c3k else BottleNect(self.c) for _ in range(n)
            # C3k(self.c, self.c, 2, shortcut, g) if c3k else AKCBAM(self.c) for _ in range(n)
        )
class C3kGC(C3GC):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(BottleNect(self.c) for _ in range(n)))
        # self.m = nn.Sequential(*(Cross_AKConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
class C3MSCk2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            # C3MSCk(self.c, self.c, 2, shortcut, g) if c3k else Cross_AKConv(self.c, self.c, shortcut, g) for _ in range(n)
            # C3MSCk(self.c, self.c, 2, shortcut, g) if c3k else S2CrossConvDilated(self.c, self.c, shortcut, g) for _ in range(n)
            # C3MSCk(self.c, self.c, 2, shortcut, g) if c3k else S2CrossConvDilatedReal(self.c, self.c, shortcut, g) for _ in range(n)
            C3MSCk(self.c, self.c, 2, shortcut, g) if c3k else DenseCross(self.c, self.c, shortcut, g) for _ in range(n)

            # C3MSCk(self.c, self.c, 2, shortcut, g) if c3k else S2DenseCrossConvDilated(self.c, self.c, shortcut, g) for _ in range(n)
            # C3MSCk(self.c, self.c, 2, shortcut, g) if c3k else BottleneckX_CBam(self.c, self.c, shortcut, g) for _ in range(n)
            # C3MSCk(self.c, self.c, 2, shortcut, g) if c3k else AKCBAM(self.c) for _ in range(n)
        )
class C3MSCk(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*(S2DenseCrossConvDilated(c_, c_, shortcut, g, k=3, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*(S2CrossConvDilated(c_, c_, shortcut, g, k=3, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*(S2CrossConvDilatedReal(c_, c_, shortcut, g, k=3, e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(DenseCross(c_, c_, shortcut, g, k=3, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*(BottleneckX_CBam(c_, c_, shortcut, g, k=3, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*(Cross_AKConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
        # self.pe = nn.Sequential(nn.AdaptiveAvgPool2d((None, 1)), nn.AdaptiveAvgPool2d((1, None)))


    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, attn_ratio=0.5, reduction_ratio=1):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.key_dim = int(self.head_dim * attn_ratio)
#         self.scale = self.key_dim**-0.5
#         nh_kd = self.key_dim * num_heads
#         h = dim + nh_kd * 2
#         self.qkv = Conv(dim, h, 1, act=False)
#         self.proj = Conv(dim, dim, 1, act=False)
#         self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
        
#         # New parameters for sequence reduction
#         self.reduction_ratio = reduction_ratio
#         self.linear_k = nn.Linear(self.key_dim * num_heads, self.key_dim * num_heads // reduction_ratio)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         N = H * W
#         qkv = self.qkv(x)
#         q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
#             [self.key_dim, self.key_dim, self.head_dim], dim=2
#         )

#         # Sequence reduction for the key tensor
#         if self.reduction_ratio > 1:
#             k = k.view(B, self.num_heads, self.key_dim, N // self.reduction_ratio, self.reduction_ratio)
#             k = k.permute(0, 1, 3, 2, 4).contiguous()  # Rearranging dimensions
#             k = k.view(B, self.num_heads, N // self.reduction_ratio, -1)  # Shape: (B, num_heads, N/R, C*R)
#             k = self.linear_k(k)  # Linear transformation to reduce dimensions

#         attn = (q.transpose(-2, -1) @ k) * self.scale
#         attn = attn.softmax(dim=-1)
#         x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
#         x = self.proj(x)
#         return x

class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


######################################################################

# class PSABlock(nn.Module):
#     """
#     PSABlock class implementing a Position-Sensitive Attention block for neural networks.

#     This class encapsulates the functionality for applying multi-head attention and a
#     depthwise-pointwise feed-forward network with optional shortcut connections.

#     Attributes:
#         attn (Attention): Multi-head attention module.
#         ffn (nn.Sequential): Enhanced feed-forward network module.
#         add (bool): Flag indicating whether to add shortcut connections.

#     Methods:
#         forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.
#     """

#     def __init__(self, c, attn_ratio=0.5, num_heads=4, expansion=2, shortcut=True) -> None:
#         """Initializes the PSABlock with attention and enhanced feed-forward layers."""
#         super().__init__()

#         self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        
#         mid = c * expansion
#         self.ffn = nn.Sequential(
#             # 1×1 pointwise expand
#             Conv(c, mid, k=1),
#             # 3×3 depthwise conv
#             Conv(mid, mid, k=3, p=1, g=mid),
#             nn.BatchNorm2d(mid),
#             nn.GELU(),
#             # 1×1 pointwise project
#             Conv(mid, c, k=1),
#             nn.BatchNorm2d(c),
#         )

#         self.add = shortcut

#     def forward(self, x):
#         """Executes a forward pass through PSABlock, applying attention and enhanced FFN."""
#         x = x + self.attn(x) if self.add else self.attn(x)
#         x = x + self.ffn(x)  if self.add else self.ffn(x)
#         return x

######################################################################

class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(self, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """Load the model and weights from torchvision."""
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """Forward pass through the model."""
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y
