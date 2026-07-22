import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Backbone
from .positional_encoding import PositionalEncodingsFixed
from .ope import OPEModule
from .transformer import TransformerEncoder
from .regression_head import DensityMapRegressor

from .ViT_Encoder import CLIPVisionTransformer as vit
from .ViT_Encoder import VPTCLIPVisionTransformer as vpt
from .ViT_Encoder_adaption import SPTCLIPVisionTransformer as spt
from .Text_Encoder import CLIPTextEncoder

from timm.models.layers import trunc_normal_

def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        # module.weight 用于初始化的权重张量
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=0, flag=True):
        super(UpConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
        if flag:
            self.gn = nn.GroupNorm(8, out_channels)
        else:
            self.gn = nn.GroupNorm(1, out_channels)
        self.gelu = nn.GELU()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.flag = flag

    def forward(self, trg):
        trg = self.conv(trg)
        if self.flag:
            trg = self.up(self.gelu(self.gn(trg)))
        return trg

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                               padding=0, bias=False)
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        return out

# denseblock-new
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_channels * 4
        self.trans = TransitionBlock(in_channels*2, out_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return self.trans(torch.cat([x, out], 1))

# 交叉注意力跳跃连接
class CrossAttentionSkipConnection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 深度可分离卷积减少计算量
        self.depthwise_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise_conv = nn.Conv2d(channels, channels, 1)
        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, query, key_value):
        """
        简化版交叉注意力，使用深度可分离卷积
        """
        # 计算注意力权重
        attn = torch.sigmoid(
            self.depthwise_conv(key_value) +
            F.interpolate(query, size=key_value.shape[2:], mode='bilinear')
        )

        # 应用注意力
        attended_value = key_value * attn

        # 上采样并融合
        out = F.interpolate(attended_value, size=query.shape[2:], mode='bilinear')
        return self.norm(query + self.pointwise_conv(out))

class Counter(nn.Module):
    def __init__(self, args):
        super(Counter,self).__init__()

        self.v = args.v
        self.enc = args.enc

        embed_dims = 512
        proj_dims = 64
        # 添加全连接的线性层，其中embed是input，proj_dim是output
        # 数学公式就是一个y=wx+b
        self.t_proj = nn.Linear(embed_dims, proj_dims)
        self.v_proj = nn.Linear(embed_dims, proj_dims)

        """
        prototype函数
        reduction = 8 所以输出的图像维度是48*48，当修改为16时，变成24*24
        """
        self.backbone = Backbone(
            'resnet18', pretrained=True, dilation=False, reduction=16,
            swav=True, requires_grad=False
        )

        '''
        超参数emb_dim = 128 256 512
        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, emb_dim, kernel_size=1
        )
        '''
        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, 512, kernel_size=1
        )

        '''
        超参数emb_dim = 128 256 512
        self.pos_emb = PositionalEncodingsFixed(emb_dim)
        '''
        self.pos_emb = PositionalEncodingsFixed(512)

        '''
        超参数 num_objects=1 2 3 默认是3
        emb_dim = 128 256 512
        num_iterative_steps=3, emb_dim=256, kernel_dim=3, num_objects=3, num_heads=8,
        reduction=16, layer_norm_eps=1e-5, mlp_factor=8, norm_first=True
        '''
        self.ope = OPEModule(
            3, 512, 3, 3, 8,
            16, 1e-5, 8, True, nn.GELU, True
        )

        '''
        超参数emb_dim
        self.aux_heads = nn.ModuleList([
            DensityMapRegressor(emb_dim, reduction)
            for _ in range(num_ope_iterative_steps - 1)
        ])
        
        self.regression_head = DensityMapRegressor(emb_dim, reduction)
        '''
        self.aux_heads = nn.ModuleList([
            DensityMapRegressor(512, 16)
            for _ in range(3 - 1)
        ])

        self.regression_head = DensityMapRegressor(512, 16)

        '''
        超参数
        self.encoder = TransformerEncoder(
                num_encoder_layers, emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, norm
            )
        '''
        self.encoder = TransformerEncoder(
            3, 512, 8, 0.1, 1e-5,
            8, True, nn.GELU, True
        )

        # Sequential是一个容器，容器里面添加不同的操作
        self.proj = nn.Sequential(
            nn.Conv2d(768, proj_dims, 1), # 卷积操作
            nn.GroupNorm(8, proj_dims), # GroupNorm 通常用于小批量数据或需要更稳定归一化的情况。
            nn.GELU(), # 激活函数
            nn.UpsamplingBilinear2d(scale_factor=2)  # 双线性插值操作 缩放因子2
        )

        #上采样的尺度因子：4
        self.proj1 = nn.Sequential(
            nn.Conv2d(768, proj_dims, 1),
            nn.GroupNorm(8, proj_dims),
            nn.GELU(),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

        # 上采样的尺度因子：8
        self.proj2 = nn.Sequential(
            nn.Conv2d(768, proj_dims, 1),
            nn.GroupNorm(8, proj_dims),
            nn.GELU(),
            nn.UpsamplingBilinear2d(scale_factor=8)
        )

        # segment-aware 解码器:解码器的作用是从低分辨率的特征图逐步恢复高分辨率的输出。具体来说：
        # 通过逐层操作将proj_dims转化为1
        self.decoder = nn.ModuleList([
                                    # 解码器的第一层，input: proj_dims+1 output: proj_dims 卷积核大小 3 步长1
                                    UpConv(proj_dims+1, proj_dims, 3, 1),
                                    # UpConv(proj_dims+2, proj_dims, 3, 1),
                                    # 解码器的第二层
                                    UpConv(proj_dims, proj_dims, 3, 1),
                                    # 解码器的第三层
                                    UpConv(proj_dims, proj_dims, 3, 1),
                                    # 解码器的第四层
                                    UpConv(proj_dims, proj_dims, 3, 1),
                                    # 解码器的第五层
                                    UpConv(proj_dims, 1, 1, flag=False)
                                ])

        # self.block1 = BottleneckBlock(proj_dims, proj_dims)
        # self.block2 = BottleneckBlock(proj_dims, proj_dims)
        # self.block3 = BottleneckBlock(proj_dims, proj_dims)

        # 初始化注意力权重
        # self.attn_weight = nn.Parameter(torch.ones(1, 1, 24, 24))
        self.attn_weight1 = nn.Parameter(torch.ones(1, 1, 24, 24))
        self.attn_weight2 = nn.Parameter(torch.ones(1, 1, 24, 24))
        # 初始化注意力偏差
        self.attn_bias = nn.Parameter(torch.zeros(1, 1, 24, 24))

        # 替代att_map转化为count_map
        # self.conv = nn.Conv2d(1, 1, kernel_size=1)  # 输入输出通道均为1
        # 采用vit作为图像的特征提取：效果bad
        # self.vit = vit(pretrained=args.MODEL.pretrain + 'ViT-B-16.pt')
        # self.input_proj_vit = nn.Conv2d(
        #     512, 256, kernel_size=1
        # )
        self.init_weights()

        if args.enc == "spt":
            # 自定义token和patch-size传入SPTCLIPVisionTransformer
            # 应该就是spt视觉编码器 类似与 ViT
            # num_tokens：10 patch_size：16
            self.v_enc = spt(pretrained=args.MODEL.pretrain+'ViT-B-16.pt', num_tokens=args.num_tokens, patch_size=args.patch_size)
            self.v_enc.init_weights()
        elif args.enc == "vpt":
            self.v_enc = vpt(pretrained=args.MODEL.pretrain+'ViT-B-16.pt')
            self.v_enc.init_weights()
        else:
            raise NotImplementedError

        # t是文本特征提取器
        self.t_enc = CLIPTextEncoder(pretrained=args.MODEL.pretrain+'ViT-B-16.pt', embed_dim=embed_dims)
        # 为文本特征提取器统一的初始化权重
        self.t_enc.init_weights()

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # 当用model类的调用的是这个方法  v是image feature  tokenized_text是文本的token
    # v.shape torch.Size([4, 3, 384, 384])
    def forward(self, v, tokenized_text):

        """
        添加 prototype
        """

        # 选择用vit来作为prototype提取方法
        # backbone_features = self.vit(v)
        # src = self.input_proj_vit(backbone_features[-1])
        backbone_features = self.backbone(v) # torch.Size([4, 896, 48, 48])
        src = self.input_proj(backbone_features)  # torch.Size([4, 256, 24, 24])  # 通过卷积来减少通道数
        bs, c, h, w = src.size()
        pos_emb = self.pos_emb(bs, h, w, src.device).flatten(2).permute(2, 0, 1) # 设置位置编码用于transformer的encoder提取
        src = src.flatten(2).permute(2, 0, 1)
        image_features = self.encoder(src, pos_emb, src_key_padding_mask=None, src_mask=None)

        '''
        超参数prototype= 128 256 512
        '''
        f_e = image_features.permute(1, 2, 0).reshape(-1, 512, h, w) # [4,256,24,24]

        all_prototypes = self.ope(f_e, pos_emb) # [3,27,4,256] [L, num_objects, kernel_dim^2,emb_dim]: 其中L是总原型数

        # predicted_dmaps = None
        outputs = list()
        for i in range(all_prototypes.size(0)):
            # reshape(bs, 3, 3, 3, -1):将卷积核参数重组为空间形式（kernel_dim * kernel_dim）
            # flatten(0,2) 是将前3维（bs,num_objects,emb_dim）合并为一维，得到 (bs*num_objects*emb_dim, kernel_dim, kernel_dim)。
            # [:,None,...]:在第二维插入一个维度，变成（bs*num_objects*emb_dim,1，kernel_dim,kernel_dim）
            '''
            超参数 num_objects = 1 2 3
            kernel_dim = 1 3 5
            prototypes = all_prototypes[i, ...].permute(1, 0, 2).reshape(
                bs, num_objects, self.kernel_dim, self.kernel_dim, -1
            ).permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...]
            '''
            prototypes = all_prototypes[i, ...].permute(1, 0, 2).reshape(
                bs, 3, 3, 3, -1
            ).permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...]

            # 为什么要迭代3次，并在通道维度上合并，是因为上一行代码：原形的reshape时设置了num_objects
            '''
            超参数 num_objects = prototype number
            self.emb_dim = 128 256 512
            response_maps = F.conv2d(
                torch.cat([f_e for _ in range(num_objects)], dim=1).flatten(0, 1).unsqueeze(0),
                prototypes,
                bias=None,
                padding=self.kernel_dim // 2,
                groups=prototypes.size(0)
            ).view(
                bs, num_objects, self.emb_dim, h, w
            ).max(dim=1)[0]
            '''
            response_maps = F.conv2d(
                torch.cat([f_e for _ in range(2)], dim=1).flatten(0, 1).unsqueeze(0),
                prototypes,
                bias=None,
                padding= 3 // 2,
                groups=prototypes.size(0)
            ).view(
                bs, 3, 512, h, w
            ).max(dim=1)[0]

            # LOCA的decoder process
            if i == all_prototypes.size(0) - 1:
                predicted_dmaps_prototype = self.regression_head(response_maps)  # torch.Size([4, 1, 24, 24])
                # predicted_dmaps = F.interpolate(self.regression_head(response_maps), size=(24, 24), mode='bilinear',
                #                                            align_corners=False)
                # predicted_dmaps = self.regression_head(response_maps)
            else:
                predicted_dmaps_prototype = self.aux_heads[i](response_maps)
            outputs.append(predicted_dmaps_prototype)

        '''
        以下的操作是将文本插入到图像特征
        '''
        # batch大小
        B = v.size(0)

        t = []
        # 对输入的 tokenized text（分词后的文本）进行处理，并将处理后的结果存储在一个列表t中。
        # 循环的是batch_size大小
        for tt in tokenized_text:
            # tt是tokenized_text的token_id  每条文本的长度是77
            # tt的长度是dataset.py下的设置了多少条的token  该len(tt)是11
            # 通过token id输入到CLIPTextEncoder模型中生成该文本的embedding（语义信息）
            # len(tt) = 11
            _t = self.t_enc(tt) # 因为tokenized_text.size() [11, 512]
            _t = _t / _t.norm(dim=-1, keepdim=True)  # 对最后一维的数字进行归一化
            _t = _t.mean(dim=0) # 对第一维数字进行求均值
            _t /= _t.norm() # 对全局特征进行归一化
            t.append(_t) # 将每次一次batch得到的text_embedding输入到数组中
        _t = torch.stack(t) # torch.stack() 用于将多个张量沿着一个新的维度进行堆叠。
        
        if self.enc == "vpt":
            # 将图像输入到visual encode
            v = self.v_enc(v)
        elif self.enc == "spt":
            # 利用文本和视觉图像合并得到处理后的混合特征的图像
            v = self.v_enc(v, _t.unsqueeze(1)) # 得到是一个图像特征的 元组
        else:
            raise NotImplementedError


        # 这个v是处理后的混合特征图像
        # 先对输出特征4维变为3维，然后在进行全连接操作，输出维度为64，然后再进行3维转化为4维，单独对文本特征进行全连接处理输出维度64
        # proj_v: torch.Size([4, 64, 24, 24])
        proj_v, _t = self.d3_to_d4(self.v_proj(self.d4_to_d3(v[-1]))), self.t_proj(_t)

        # 爱因斯坦求和计算方式的目的是针对两个张量的维度不对应，该公式是将_t（bc）的通道维度和proj_v(bchw)的通道维度进行矩阵乘法得到维度bhw，
        # 其实attn_map就是论文中得到的S
        attn_map = torch.einsum('bc,bchw->bhw', _t, proj_v).unsqueeze(1)  # torch.Size([4, 1, 24, 24])

        # learning affine transformation将输出的（目标域）S转化为S'（count map）
        #  self.attn_weight.expand(B, -1, -1, -1) :添加batch，为了跟attn_map的batch维度一致
        # self.attn_weight.expand(B, -1, -1, -1) 需要被训练的参数权重W
        # self.attn_bias.expand(B, -1, -1, -1) 需要被训练的参数偏置B

        # affine_attn_map = self.attn_weight.expand(B, -1, -1, -1) * attn_map + self.attn_bias.expand(B, -1, -1, -1) # torch.Size([4, 1, 24, 24])

        # 4.19 采用非线性映射（如Sigmoid或Softmax）
        # affine_attn_map = torch.sigmoid(self.attn_weight.expand(B, -1, -1, -1) * attn_map + self.attn_bias.expand(B, -1, -1, -1))

        # 4.17 采用卷积映射到计数图
        # affine_attn_map = self.conv(attn_map)

        # 4.19 乘法和加法结合的映射方式
        affine_attn_map = self.attn_weight1.expand(B, -1, -1, -1) * attn_map + \
                          self.attn_weight2.expand(B, -1, -1, -1) * attn_map.pow(2) + \
                          self.attn_bias.expand(B, -1, -1, -1)

        # print(proj_v.shape) # torch.Size([4, 64, 24, 24])
        # print(affine_attn_map.shape) # torch.Size([4, 1, 24, 24])
        '''
        视觉编码器（visual和text混合）得到的visual feature 与 attn_map（视觉和文本的爱因斯坦求和）的可学习的仿射映射 进行拼接
        '''
        x = torch.cat([proj_v, affine_attn_map], dim=1)

        # 尝试将x和LOCA模型的response map进行合并 2.28
        # x = torch.cat([x, predicted_dmaps], dim=1)

        # print(x.shape) # torch.Size([4, 65, 24, 24])
        # print(predicted_dmaps.shape) # torch.Size([4, 256, 48, 48]])

        for i, d in enumerate(self.decoder):
            if i==1:
                # 采用dense_block特征提取
                # x = d(self.block1(x + self.proj(v[-2]) * F.interpolate(affine_attn_map, scale_factor=2)))
                # v是text-aware感知的vit模型得到的结果，和affine_attn_map映射后的count_map进行上采用
                x = d(x + self.proj(v[-2]) * F.interpolate(affine_attn_map, scale_factor=2))
            elif i==2:
                # x = d(self.block2(x + self.proj1(v[-3]) * F.interpolate(affine_attn_map, scale_factor=4)))
                x = d(x + self.proj1(v[-3]) * F.interpolate(affine_attn_map, scale_factor=4))
            elif i==3:
                # x = d(self.block3(x + self.proj2(v[-4]) * F.interpolate(affine_attn_map, scale_factor=8)))
                x = d(x + self.proj2(v[-4]) * F.interpolate(affine_attn_map, scale_factor=8))
            else:
                x = d(x)


        # 这个解码器其实就是segment aware用于提升模型的泛化能力
        # for i, d in enumerate(self.decoder):
        #     if i==1:
        #         # v是text-aware感知的vit模型得到的结果，和affine_attn_map映射后的count_map进行上采用
        #         x = d(x + self.proj(v[-2]) * F.interpolate(affine_attn_map, scale_factor=2))
        #     elif i==2:
        #         x = d(x + self.proj1(v[-3]) * F.interpolate(affine_attn_map, scale_factor=4))
        #     elif i==3:
        #         x = d(x + self.proj2(v[-4]) * F.interpolate(affine_attn_map, scale_factor=8))
        #     else:
        #         x = d(x)

        # 修改为交叉注意力跳跃连接 no-bad
        # for i, d in enumerate(self.decoder):
        #     if i==1:
        #         skip_feat = self.proj(v[-2]) * F.interpolate(affine_attn_map, scale_factor=2)
        #         x = self.cross_attn1(x, skip_feat)
        #         x = d(x)
        #     elif i==2:
        #         skip_feat = self.proj1(v[-3]) * F.interpolate(affine_attn_map, scale_factor=4)
        #         x = self.cross_attn2(x, skip_feat)
        #         x = d(x)
        #     elif i==3:
        #         skip_feat = self.proj2(v[-4]) * F.interpolate(affine_attn_map, scale_factor=8)
        #         x = self.cross_attn3(x, skip_feat)
        #         x = d(x)
        #     else:
        #         x = d(x)

        return x, F.interpolate(affine_attn_map, scale_factor=16), affine_attn_map, outputs
        # return x, F.interpolate(affine_attn_map, scale_factor=16), affine_attn_map, predicted_dmaps

    def d3_to_d4(self, t):
        b, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(b, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)
