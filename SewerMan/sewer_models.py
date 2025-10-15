# import torch
# import timm


# model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
# model.reset_classifier(num_classes=17)  # 修改分类头

# a = torch.rand(2,3,224,224)
# B = model(a)
# print(B.shape)

# def zhong2023_multilabel(num_classes, **kwargs):
#     assert num_classes > 1
#     model = timm.create_model('swin_base_patch4_window7_224', pretrained=False)
#     model.reset_classifier(num_classes=17)  # 修改分类头
#
#     return model

# import torch
# import timm

# model = timm.create_model('mobilenetv3_large_100', pretrained=False)
# model.reset_classifier(num_classes=17)  # 修改分类头

# a = torch.rand(2,3,224,224)
# B = model(a)
# print(B.shape)

# def zhong2023_multilabel(num_classes, **kwargs):
#     assert num_classes > 1
#
#     model = timm.create_model('mobilenetv3_large_100', pretrained=False)
#     model.reset_classifier(num_classes=17)  # 修改分类头
#
#     return model

# import torch
# import timm
#
#
#
# def zhong2023_multilabel(num_classes, **kwargs):
#     assert num_classes > 1
#
#     model = timm.create_model('tf_efficientnetv2_s', pretrained=False)
#     model.reset_classifier(num_classes=17)  # 修改分类头
#
#     return model


# import torch
# import timm
#
# # model = timm.create_model('mobilenetv3_large_100', pretrained=False)
# # model.reset_classifier(num_classes=17)  # 修改分类头
#
# # a = torch.rand(2,3,224,224)
# # B = model(a)
# # print(B.shape)
#
# def zhong2023_multilabel(num_classes, **kwargs):
#     assert num_classes > 1
#
#     model = timm.create_model('vit_base_patch16_224', pretrained=False)
#     model.reset_classifier(num_classes=17)  # 修改分类头
#
#     return model

# import torch
# import timm
#
# # model = timm.create_model('mobilenetv3_large_100', pretrained=False)
# # model.reset_classifier(num_classes=17)  # 修改分类头
#
# # a = torch.rand(2,3,224,224)
# # B = model(a)
# # print(B.shape)
#
# def zhong2023_multilabel(num_classes, **kwargs):
#     assert num_classes > 1
#
#     model = timm.create_model('mambaout_small.in1k', pretrained=False)
#     model.reset_classifier(num_classes=17)  # 修改分类头
#
#     return model

from torch.hub import load_state_dict_from_url
import torch
import torch.nn as nn

from multilabel_models.graph_layers import GraphConvolution, ConvGraphCombination
import math
import torch.nn.functional as F
from timm.models.layers import DropPath

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        """_summary_

        Args:
            v (_type_): bs, n_img, dim
            l (_type_): bs, n_text, dim
            attention_mask_v (_type_, optional): _description_. bs, n_img
            attention_mask_l (_type_, optional): _description_. bs, n_text

        Returns:
            _type_: _description_
        """
        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()
        bsz, tgt_len, _ = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        # mask vison for language
        if attention_mask_v is not None:
            attention_mask_v = (
                attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights_l.masked_fill_(attention_mask_v, float("-inf"))

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        # mask language for vision
        if attention_mask_l is not None:
            attention_mask_l = (
                attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights.masked_fill_(attention_mask_l, float("-inf"))
        attn_weights_v = attn_weights.softmax(dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l

class BiAttentionBlock(nn.Module):
    def __init__(
        self,
        v_dim,
        l_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        cfg=None,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(
            v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l
        )
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l

    # def forward(self, v:List[torch.Tensor], l, attention_mask_v=None, attention_mask_l=None)

class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        # 修改权重形状为 [num_class, hidden_dim] 而不是 [1, num_class, hidden_dim]
        self.weight = nn.Parameter(torch.Tensor(num_class, hidden_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_class))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        for i in range(self.num_class):
            self.weight.data[i].uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data[i].uniform_(-stdv, stdv)

    def forward(self, x):
        # x: [B, num_class, hidden_dim]
        # weight: [num_class, hidden_dim]
        x = (x * self.weight.unsqueeze(0)).sum(-1)  # [B, num_class]
        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)
        return x


class EnhancedQuerySelector(nn.Module):
    def __init__(self, hidden_dim, num_query):
        super().__init__()
        self.num_query = num_query

        # 投影层（修正缩进：统一4空格）
        self.text_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU()
        )
        self.img_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU()
        )

        # 可学习参数（修正缩进：对齐父级）
        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.text_weight = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.diversity_lambda = nn.Parameter(torch.tensor(0.1))

    def forward(self, image_features, text_features):
        # 特征投影（修正缩进：方法内统一4空格）
        text_compressed = self.text_proj(text_features) + text_features[..., :64]
        img_compressed = self.img_proj(image_features) + image_features[..., :64]

        # 权重计算（修正缩进：嵌套块+4空格）
        text_weights = torch.sigmoid(self.text_weight(text_features)).softmax(dim=1)
        weighted_text = text_compressed * text_weights

        # 相似度计算（修正缩进：操作符对齐）
        logits = torch.einsum("bic,bjc->bij",
                              img_compressed,
                              weighted_text)  # 多行对齐
        logits = logits / (self.temperature.abs() + 1e-6)

        # 选择逻辑（修正缩进：保持方法调用对齐）
        scores = logits.softmax(dim=1).max(-1).values
        topk_idx = self._differentiable_sampling(logits, scores)
        return topk_idx

    def _differentiable_sampling(self, logits, scores):
        batch_size, num_img = scores.shape
        device = scores.device

        # Gumbel采样（修正缩进：列表操作+4空格）
        prob = scores / (scores.sum(dim=1, keepdim=True) + 1e-6)
        selected = [torch.multinomial(prob, 1)]

        # 多样性选择（修正缩进：循环内8空格）
        for _ in range(1, self.num_query):
            # 距离计算（修正缩进：嵌套12空格）
            gathered = torch.gather(
                logits, 1,  # 参数对齐
                torch.cat(selected, dim=1).unsqueeze(-1).expand(-1, -1, logits.size(-1))
            )
            diff = torch.cdist(
                logits.float().view(batch_size, num_img, -1),
                gathered.float().view(batch_size, len(selected), -1),
                p=2
            ).mean(dim=-1)

            # 组合得分（修正缩进：操作符对齐）
            combined = (scores.log() +
                        self.diversity_lambda * diff).exp()
            mask = torch.zeros_like(combined).scatter(
                1,
                torch.cat(selected, dim=1),
                -float('inf')
            )
            new_idx = (combined + mask).argmax(dim=1, keepdim=True)
            selected.append(new_idx)

        return torch.cat(selected, dim=1)

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

import numpy as np


import torch
import clip


device = "cuda" if torch.cuda.is_available() else "cpu"
# 模型选择['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']，对应不同权重
#model, preprocess = clip.load("model_data/ViT-B-32.pt", device=device)  # 载入模型


text_language = ["Cracks, breaks, and collapses", "Surface damage", "Production error",
                 "Deformation","Displaced joint","Intruding sealing material","Roots","Inﬁltration","Settled deposits",
                 "Attached deposits","Obstacle","Branch pipe","Chiseled connection","Drilled connection",
                 "Lateral reinstatement cuts","Connection with transition proﬁle","Connection with construction changes",
                 "Unknown", "Concrete", "Plastic", "Lining", "Vitrified clay", "Iron", "Brickwork", "Other"]

defect_categories = text_language[0:17]
material_categories = text_language[17:]

template_1 = [f"a photo of {defect}" for defect in defect_categories] + material_categories
template_2 = [f"This sewer image contains  {defect} defect" for defect in defect_categories] + material_categories

# 正确：直接使用列表，不要拼接成字符串
text = clip.tokenize(template_1).to(device)


import torchvision.models as models
import math
def Resnet50():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features  # 获取fc层的输入特征数
    model.fc = torch.nn.Linear(num_ftrs, 8)  # 修改fc层为输出17个特征

    weights = torch.load('model_data/resnet50_stl_material.pth')
    model_state_dict = weights['state_dict']
    model.load_state_dict(model_state_dict)
    return model


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.block = block
        self.groups = groups
        self.base_width = width_per_group

        # 224,224,3 -> 112,112,64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 112,112,64 -> 56,56,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 56,56,64 -> 56,56,256
        self.layer1 = self._make_layer(block, 64, layers[0])

        # 56,56,256 -> 28,28,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0]).to("cuda")

        # 28,28,512 -> 14,14,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1]).to("cuda")

        # 14,14,1024 -> 7,7,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2]).to("cuda")

        # 7,7,2048 -> 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 2048 -> num_classes
        self.fc_1 = nn.Linear(512 * block.expansion, num_classes)



        self.line1 = nn.Linear(512,256).to("cuda")
        self.line2 = nn.Linear(256, 512).to("cuda")
        self.line3 = nn.Linear(512, 1024).to("cuda")


        self.convgcn1 = ConvGraphCombination(nn.Conv2d(1, 256, kernel_size=1, stride=1, bias=False).to("cuda"), nn.Tanh())
        self.convgcn2 = ConvGraphCombination(nn.Conv2d(1, 512, kernel_size=1, stride=1, bias=False).to("cuda"), nn.Tanh())
        self.convgcn3 = ConvGraphCombination(nn.Conv2d(1, 1024, kernel_size=1, stride=1, bias=False).to("cuda"), nn.Tanh())

        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(4, stride=4)

        self.inputto_512 = nn.Conv2d(3584, 512, kernel_size=1)

        self.biatt = BiAttentionBlock(512, 512, 512, 4, dropout=0.1)

        self.qfc = GroupWiseLinear(17, 512)

        self.QuerySelector = EnhancedQuerySelector(512, 17)
        self.res_scale = nn.Parameter(torch.ones(1) * 0.2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.model1, self.preprocess = clip.load("model_data/ViT-B-32.pt", device=device)
        self.resnet = Resnet50()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # Conv_block
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # identity_block
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)



    def forward(self, x):

        b, c, h, w = x.shape
        text_fea = self.model1(text)

        text_lab = text_fea[0:17]
        text_promot = text_fea[17:]

        MaterialLabels = self.resnet(x)
        MaterialLabels = torch.sigmoid(MaterialLabels)

        idx = torch.argmax(MaterialLabels, dim=1)


        query_promot = torch.empty(b,512,dtype=torch.float32).to('cuda')
        for i in range (0,b):
            #query.append(text_prom[idx[i]])
            #[i] = text_prom[idx[i]])
            #query = torch.cat((query[i], text_prom[idx[i]]), dim=0)
            query_promot[i] = text_promot[idx[i]]

        #print(query_promot.shape) b C

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        query_promot1 = self.line1(query_promot).unsqueeze(1).to("cuda")
        x = self.convgcn1(x.to("cuda"), query_promot1)

        x1 = self.layer2(x)
        query_promot2 = self.line2(query_promot1).to("cuda")
        x1 = self.convgcn2(x1.to("cuda"), query_promot2)

        x2 = self.layer3(x1)
        query_promot3 = self.line3(query_promot2).to("cuda")
        x2 = self.convgcn3(x2.to("cuda"), query_promot3)


        x3 = self.layer4(x2)

        x2 = self.pool1(x2)
        x1 = self.pool2(x1)

        src = torch.cat((x1, x2, x3), dim=1)

        src = self.inputto_512(src)

        image_fea = src.flatten(2).permute(2, 0, 1)

        text_lab = text_lab.unsqueeze(1).repeat(1, b, 1)
        #mask = self.generate_mask()
        #text_lab = self.apply_text_mask(text_lab, mask)



        text_lab = text_lab.permute(1, 0, 2).to(torch.float32)
        image_fea = image_fea.permute(1, 0, 2).to(torch.float32)
        image_fea_up, text_fea_up = self.biatt(image_fea, text_lab)

        S_Query = self.QuerySelector(image_fea_up, text_fea_up)
        expanded_indices = S_Query.unsqueeze(-1).expand(-1, -1, image_fea_up.size(-1))
        selected_features = torch.gather(image_fea_up, 1, expanded_indices)


        # 6. 自注意力增强
        d_k = selected_features.size(-1)
        self_att = (selected_features @ selected_features.transpose(-2, -1)) / (d_k ** 0.5)
        self_att = self_att.softmax(dim=-1)
        self_att = (self_att @ selected_features) + selected_features * self.res_scale
        #self_att = self.norm3(self_att)

        # 7. 图像-文本交互
        attn_image = (self_att @ image_fea_up.transpose(-2, -1)) / (d_k ** 0.5)
        attn_image = attn_image.softmax(dim=-1)
        attn_image = (attn_image @ image_fea_up) + self_att * self.res_scale
        #attn_image = self.norm4(attn_image)

        # 8. 最终文本增强
        attn_text = (attn_image @ text_fea_up.transpose(-2, -1)) / (d_k ** 0.5)
        attn_text = attn_text.softmax(dim=-1)
        attn_text = (attn_text @ text_fea_up) + attn_image * self.res_scale

        x = self.qfc(attn_text)

        return x

    def freeze_backbone(self):
        backbone = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = False

    def Unfreeze_backbone(self):
        backbone = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = True


def resnet50_image(pretrained=False, progress=True, num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'], model_dir='./model_data',
                                              progress=progress)
        model.load_state_dict(state_dict)

    if num_classes != 1000:
        model.fc_1 = nn.Linear(3584, num_classes)
    return model


# a = torch.rand(2,3,224,224).to('cuda')
# at = resnet50_image(num_classes=17).to('cuda')
# B = at(a)
# print(B.shape)

def zhong2023_multilabel(num_classes, **kwargs):
    assert num_classes > 1
    model =resnet50_image(num_classes=17)

    return model

