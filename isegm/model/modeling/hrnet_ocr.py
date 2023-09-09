import os
import numpy as np
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from .ocr import SpatialOCR_Module, SpatialGather_Module
from .resnetv1b import BasicBlockV1b, BottleneckV1b
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from isegm.utils.feat_vis import save_image

relu_inplace = True

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

################################### decoder mlp ############################################
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
############################################################################################## 

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        # self.attn = External_attention(dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.bn = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        
        self.act = act_layer()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W)) 

        return x

class OverlapPatchEmbed_HR(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size_h=224, img_size_w=224, patch_size=7, stride=1, in_chans=3, embed_dim=768):
        super().__init__()
        
        patch_size = to_2tuple(patch_size)

        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        return x, H, W

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method,multi_scale_output=True,
                 norm_layer=nn.BatchNorm2d, align_corners=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.norm_layer = norm_layer
        self.align_corners = align_corners

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=relu_inplace)

    def _check_branches(self, num_branches, num_blocks, num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(num_channels[branch_index] * block.expansion),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride,
                            downsample=downsample, norm_layer=self.norm_layer))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index],
                                norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(in_channels=num_inchannels[j],
                                  out_channels=num_inchannels[i],
                                  kernel_size=1,
                                  bias=False),
                        self.norm_layer(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          kernel_size=3, stride=2, padding=1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          kernel_size=3, stride=2, padding=1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3),
                                nn.ReLU(inplace=relu_inplace)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=self.align_corners)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

class HighResolutionNet(nn.Module):
    def __init__(self, width, num_classes, ocr_width=256, small=False,
                 norm_layer=nn.BatchNorm2d, align_corners=True, embed_dims=[18,36,72,144],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer_trans=nn.LayerNorm,
                 depths=[1,1,1,1], sr_ratios=[8, 4, 2, 1]):
        super(HighResolutionNet, self).__init__()
        self.norm_layer = norm_layer
        self.width = width
        self.ocr_width = ocr_width
        self.align_corners = align_corners

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = norm_layer(64)
        self.relu = nn.ReLU(inplace=relu_inplace)

        num_blocks = 2 if small else 4

        stage1_num_channels = 64
        self.layer1 = self._make_layer(BottleneckV1b, 64, stage1_num_channels, blocks=num_blocks)
        stage1_out_channel = BottleneckV1b.expansion * stage1_num_channels

        self.stage2_num_branches = 2
        num_channels = [width, 2 * width]
        num_inchannels = [
            num_channels[i] * BasicBlockV1b.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_inchannels)
        self.stage2, pre_stage_channels = self._make_stage(
            BasicBlockV1b, num_inchannels=num_inchannels, num_modules=1, num_branches=self.stage2_num_branches,
            num_blocks=2 * [num_blocks], num_channels=num_channels)

        self.stage3_num_branches = 3
        num_channels = [width, 2 * width, 4 * width]
        num_inchannels = [
            num_channels[i] * BasicBlockV1b.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_inchannels)
        self.stage3, pre_stage_channels = self._make_stage(
            BasicBlockV1b, num_inchannels=num_inchannels,
            num_modules=3 if small else 4, num_branches=self.stage3_num_branches,
            num_blocks=3 * [num_blocks], num_channels=num_channels)

        self.stage4_num_branches = 4
        num_channels = [width, 2 * width, 4 * width, 8 * width]
        num_inchannels = [
            num_channels[i] * BasicBlockV1b.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_inchannels)
        self.stage4, pre_stage_channels = self._make_stage(
            BasicBlockV1b, num_inchannels=num_inchannels, num_modules=2 if small else 3,
            num_branches=self.stage4_num_branches,
            num_blocks=4 * [num_blocks], num_channels=num_channels)

        last_inp_channels = np.int_(np.sum(pre_stage_channels))
        if self.ocr_width > 0:
            ocr_mid_channels = 2 * self.ocr_width
            ocr_key_channels = self.ocr_width

            self.conv3x3_ocr = nn.Sequential(
                nn.Conv2d(last_inp_channels, ocr_mid_channels,
                          kernel_size=3, stride=1, padding=1),
                norm_layer(ocr_mid_channels),
                nn.ReLU(inplace=relu_inplace),
            )
            self.ocr_gather_head = SpatialGather_Module(num_classes)

            self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                     key_channels=ocr_key_channels,
                                                     out_channels=ocr_mid_channels,
                                                     scale=1,
                                                     dropout=0.05,
                                                     norm_layer=norm_layer,
                                                     align_corners=align_corners)
            self.cls_head = nn.Conv2d(
                ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

            self.aux_head = nn.Sequential(
                nn.Conv2d(last_inp_channels, last_inp_channels,
                          kernel_size=1, stride=1, padding=0),
                norm_layer(last_inp_channels),
                nn.ReLU(inplace=relu_inplace),
                nn.Conv2d(last_inp_channels, num_classes,
                          kernel_size=1, stride=1, padding=0, bias=True)
            )
        else:
            self.cls_head = nn.Sequential(
                nn.Conv2d(last_inp_channels, last_inp_channels,
                          kernel_size=3, stride=1, padding=1),
                norm_layer(last_inp_channels),
                nn.ReLU(inplace=relu_inplace),
                nn.Conv2d(last_inp_channels, num_classes,
                          kernel_size=1, stride=1, padding=0, bias=True)
            )
            self.aux_head = nn.Sequential(
                nn.Conv2d(last_inp_channels, last_inp_channels,
                          kernel_size=1, stride=1, padding=0),
                norm_layer(last_inp_channels),
                nn.ReLU(inplace=relu_inplace),
                nn.Conv2d(last_inp_channels, num_classes,
                          kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.patch_embed_HR1 = OverlapPatchEmbed_HR(img_size_h=80, img_size_w=120, patch_size=3, stride=1, in_chans=18,
                                              embed_dim=18)
        self.patch_embed_HR2 = OverlapPatchEmbed_HR(img_size_h=40, img_size_w=60, patch_size=3, stride=1, in_chans=36,
                                              embed_dim=36)                            
        self.patch_embed_HR3 = OverlapPatchEmbed_HR(img_size_h=20, img_size_w=30, patch_size=3, stride=1, in_chans=72,
                                              embed_dim=72)
        self.patch_embed_HR4 = OverlapPatchEmbed_HR(img_size_h=10, img_size_w=15, patch_size=3, stride=1, in_chans=144,
                                              embed_dim=144)
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur], norm_layer=norm_layer_trans,
            sr_ratio=sr_ratios[0])
        self.norm1 = norm_layer_trans(18)

        cur += depths[0]
        self.block2 = Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur], norm_layer=norm_layer_trans,
            sr_ratio=sr_ratios[1])
        self.norm2 = norm_layer_trans(36)

        cur += depths[1]
        self.block3 = Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur], norm_layer=norm_layer_trans,
            sr_ratio=sr_ratios[2])
        self.norm3 = norm_layer_trans(72)

        cur += depths[2]
        self.block4 = Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur], norm_layer=norm_layer_trans,
            sr_ratio=sr_ratios[3])
        self.norm4 = norm_layer_trans(144)

        ############# edge detection ##################
        sobel_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype='float32')
        sobel_kernel = sobel_kernel.reshape((1,1,3,3))
        self.edge_conv2d = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.edge_conv2d.weight.data = torch.from_numpy(sobel_kernel)

        self.conv1_tf = nn.Conv2d(3,18,kernel_size=3,stride=4,padding=1,bias=False)
        self.conv2_tf = nn.Conv2d(18,36,kernel_size=3,stride=2,padding=1,bias=False)
        self.conv3_tf = nn.Conv2d(36,72,kernel_size=3,stride=2,padding=1,bias=False)
        self.conv4_tf = nn.Conv2d(72,144,kernel_size=3,stride=2,padding=1,bias=False)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False),
                        self.norm_layer(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=relu_inplace)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels,
                                  kernel_size=3, stride=2, padding=1, bias=False),
                        self.norm_layer(outchannels),
                        nn.ReLU(inplace=relu_inplace)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride,
                            downsample=downsample, norm_layer=self.norm_layer))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_stage(self, block, num_inchannels,
                    num_modules, num_branches, num_blocks, num_channels,
                    fuse_method='SUM',
                    multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output,
                                     norm_layer=self.norm_layer,
                                     align_corners=self.align_corners)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x, interactions, additional_features=None):
        feats, feats_add = self.compute_hrnet_feats(x, interactions, additional_features)

        if self.ocr_width > 0:
            out_aux = self.aux_head(feats)
            feats = self.conv3x3_ocr(feats)

            context = self.ocr_gather_head(feats, out_aux)
            feats = self.ocr_distri_head(feats, context)
            out = self.cls_head(feats)

            out_aux_edge = self.edge_conv2d(out)
            return [out, out_aux, out_aux_edge]
        else:
            return [self.cls_head(feats), self.aux_head(feats_add)]

    def compute_hrnet_feats(self, x, interactions, additional_features):
        x = self.compute_pre_stage_features(x, additional_features)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_num_branches):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_num_branches):
            if self.transition2[i] is not None:
                if i < self.stage2_num_branches:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_num_branches):
            if self.transition3[i] is not None:
                if i < self.stage3_num_branches:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        return self.aggregate_hrnet_features(x, interactions)

    def compute_pre_stage_features(self, x, additional_features):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if additional_features is not None:
            x = x + additional_features
        x = self.conv2(x)
        x = self.bn2(x)
        return self.relu(x)

    def aggregate_hrnet_features(self, x, interactions):

        x_tf1 = self.conv1_tf(interactions)
        x_tf2 = self.conv2_tf(x_tf1)
        x_tf3 = self.conv3_tf(x_tf2)
        x_tf4 = self.conv4_tf(x_tf3)
        x[0] = x[0] + x_tf1
        x[1] = x[1] + x_tf2
        x[2] = x[2] + x_tf3
        x[3] = x[3] + x_tf4

        ax0_h, ax0_w = x[0].size(2), x[0].size(3)
        ax1 = F.interpolate(x[1], size=(ax0_h, ax0_w),
                           mode='bilinear', align_corners=self.align_corners)
        ax2 = F.interpolate(x[2], size=(ax0_h, ax0_w),
                           mode='bilinear', align_corners=self.align_corners)
        ax3 = F.interpolate(x[3], size=(ax0_h, ax0_w),
                           mode='bilinear', align_corners=self.align_corners)
        addb = torch.cat([x[0], ax1, ax2, ax3], 1)

        B = x[0].shape[0]

        # stage 1
        x[0], H0, W0 = self.patch_embed_HR1(x[0])
        x[0] = self.block1(x[0], H0, W0)
        x[0] = self.norm1(x[0])
        x[0] = x[0].reshape(B, H0, W0, -1).permute(0, 3, 1, 2).contiguous() # 经过 stage 1 后的y特征图

        # stage 2
        x[1], H1, W1 = self.patch_embed_HR2(x[1])
        x[1] = self.block2(x[1], H1, W1)
        x[1] = self.norm2(x[1])
        x[1] = x[1].reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous() # 经过 stage 2 后的y特征图

        # stage 3
        x[2], H2, W2 = self.patch_embed_HR3(x[2])
        x[2] = self.block3(x[2], H2, W2)
        x[2] = self.norm3(x[2])
        x[2] = x[2].reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous() # 得到 stage 3 后的z特征图
        
        # stage 4
        x[3], H3, W3 = self.patch_embed_HR4(x[3])
        x[3] = self.block4(x[3], H3, W3)
        x[3] = self.norm4(x[3])
        x[3] = x[3].reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous() # 得到 stage 4 后的w特征图
        
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=self.align_corners)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=self.align_corners)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w),
                           mode='bilinear', align_corners=self.align_corners)

        return torch.cat([x[0], x1, x2, x3], 1),addb

    def load_pretrained_weights(self, pretrained_path=''):
        model_dict = self.state_dict()

        if not os.path.exists(pretrained_path):
            print(f'\nFile "{pretrained_path}" does not exist.')
            print('You need to specify the correct path to the pre-trained weights.\n'
                  'You can download the weights for HRNet from the repository:\n'
                  'https://github.com/HRNet/HRNet-Image-Classification')
            exit(1)
        pretrained_dict = torch.load(pretrained_path, map_location={'cuda:0': 'cpu'})
        pretrained_dict = {k.replace('last_layer', 'aux_head').replace('model.', ''): v for k, v in
                           pretrained_dict.items()}

        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
