import torch
from torch import Tensor
import torch.nn as nn
from resnet._internally_replaced_utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
import numpy as np
from torch.nn import functional as F
from sklearn.mixture import GaussianMixture
from torchvision import utils as vutils
from resnet.utils import visualize_cam


import warnings
warnings.filterwarnings('ignore')

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
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

    def forward(self, x: Tensor) -> Tensor:
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


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        xfmer_hidden_size = 1024,
        xfmer_layer = 2,
        pool = 'avg',
        p_threshold = 0.5
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.xfmer_hidden_size = xfmer_hidden_size
        self.xfmer_layer = xfmer_layer
        self.reduce = nn.Conv2d(2048, self.xfmer_hidden_size, kernel_size=1, stride=1, padding=0, bias=False)
        # self.reduce = nn.Linear(2048, self.xfmer_hidden_size, bias=False)
        # self.reduce = nn.Identity()

        self.xfmer_dropout = nn.Dropout(0.1)
        self.xfmer = Transformer_head(hidden_size=self.xfmer_hidden_size,layers=self.xfmer_layer, inter=False)
      
        self.x2_pos_embed = get_2d_sincos_pos_embed(self.xfmer_hidden_size, 16) #x2 regular grid
        self.x2_pos_embed = torch.from_numpy(self.x2_pos_embed).float().unsqueeze(0)

        self.pool = pool
        if self.pool == 'max':
            self.max1d = nn.MaxPool1d(2)

        if self.pool == 'cat':
            self.xfmer_layernorm = nn.LayerNorm(self.xfmer_hidden_size)
            self.xfmer_fc = nn.Linear(self.xfmer_hidden_size*2, num_classes)
        else:
            self.xfmer_fc = nn.Sequential(
                nn.LayerNorm(self.xfmer_hidden_size),
                nn.Linear(self.xfmer_hidden_size, num_classes)
            )  
        
        self.p_threshold = p_threshold

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(mean=0.0, std=0.01)
            #     if m.bias is not None:
            #         m.bias.data.zero_()
        for m in self.xfmer.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x1, x2, img1_grid, id: Tensor) -> Tensor:
        # See note [TorchScript super()]
        img1 = x1
        img2 = x2

        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)

        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)

        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)

        x1_layer4 = x1
        x2_layer4 = x2

        x1_patch_layer4 = torch.flatten(x1_layer4, start_dim=2).permute(0, 2, 1) # bs, 256, 2048
        x2_patch_layer4 = torch.flatten(x2_layer4, start_dim=2).permute(0, 2, 1)

        x1 = self.reduce(x1)
        x2 = self.reduce(x2)

        x1_patch = torch.flatten(x1, start_dim=2).permute(0, 2, 1) # bs, 256, 1024
        x2_patch = torch.flatten(x2, start_dim=2).permute(0, 2, 1)

        # gmm
        x1_avg = torch.mean(x1_patch_layer4, dim=-1) #bs, 256
        x2_avg = torch.mean(x2_patch_layer4, dim=-1) 

        # prob1 = torch.zeros_like(x1_avg)
        # prob2 = torch.zeros_like(x2_avg)

        x1_avg_max, _ = torch.max(x1_avg, dim=1) #bs
        x1_avg_min, _ = torch.min(x1_avg, dim=1)
        x1_avg = (x1_avg - x1_avg_min.unsqueeze(-1)) / (x1_avg_max.unsqueeze(-1) - x1_avg_min.unsqueeze(-1))

        x2_avg_max, _ = torch.max(x2_avg, dim=1)
        x2_avg_min, _ = torch.min(x2_avg, dim=1)
        x2_avg = (x2_avg - x2_avg_min.unsqueeze(-1)) / (x2_avg_max.unsqueeze(-1) - x2_avg_min.unsqueeze(-1))

        
        # pred: 1-black，0-eye
        pred1 = (x1_avg < self.p_threshold)      
        pred2 = (x2_avg < self.p_threshold)  # bs, 256
        pred = torch.cat([pred1, pred2], dim=-1) # bs, 512


        # visualize
        '''
        vis_bs = 0
        # mask = torch.zeros(1,1,16*16)
        # indice = indices1[vis_bs][:int(16*16*1)]
        # mask[:,:,indice] = 1
        mask = 1-pred1[vis_bs].unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
        # print(mask.size())
        mask = mask.reshape(1,1,16,16)
        mask = F.interpolate(mask, scale_factor=32, mode='bilinear', align_corners=True)
        mask = mask.squeeze(0).permute(1,2,0)

        # mask = torch.rand(512,512,1)
        # img = F.interpolate(img1, scale_factor=1/32, mode='bilinear', align_corners=True)
        heatmap, result = visualize_cam(mask, img1[vis_bs])
        # print(heatmap.size(), result.size())
        path = './hybrid_vis_high_activation/'
        print(id)
        vutils.save_image(result, path+id'./mask.jpg')
        '''
        # print(id)

        for vis_bs in range(img1.size(0)):
            name = id[vis_bs].split('_')[0]
            print(img1[vis_bs].size(), pred1[vis_bs].size())
            self.vis_threshold(img1[vis_bs], pred1[vis_bs], name+'_1')
            self.vis_threshold(img2[vis_bs], pred2[vis_bs], name+'_2')

        
        



        # out_patch = torch.cat([x1_patch, x2_patch], dim=1) #bs, 512, 2048

        # generate x1 aligned pos embedding 
        # img1_grid (bs,512,512,2)
        bs, feat_dim, g_size, _ = x1.size()
        # feat_dim = self.xfmer_hidden_size
        # print(img1_grid.shape, img1_grid.max(),img1_grid.min()) # bs, 16,16,2

        x1_grid = F.interpolate(img1_grid.permute(0,3,1,2), scale_factor=1/32, mode='bilinear', align_corners=True)
        x1_grid = x1_grid * np.mean(np.array(range(g_size))) + np.mean(np.array(range(g_size))) # bs,2,16,16
        # print(x1_grid)

        x1_grid_embed = torch.ones(bs,g_size**2,feat_dim).cuda()
        for b in range(bs):
            x1_grid_bs = x1_grid[b].reshape([2,1,g_size,g_size])
            x1_grid_embed_bs = get_2d_sincos_pos_embed(embed_dim=feat_dim, grid_size=g_size, grid=x1_grid_bs.cpu().detach().numpy()) # 256,2048
            x1_grid_embed[b] = torch.from_numpy(x1_grid_embed_bs)
        # print(x1_grid_embed[0])

        self.x2_pos_embed = self.x2_pos_embed.to(x1.device)
        x1_patch = x1_patch + x1_grid_embed #bs,256,1024
        x2_patch = x2_patch + self.x2_pos_embed

        # sorted_x1_grid_embed = torch.gather(x1_grid_embed, 1, indices1.unsqueeze(-1).expand(bs, g_size**2, feat_dim))
        # sorted_x2_pos_embed = torch.gather(self.x2_pos_embed.repeat(bs,1,1), 1, indices2.unsqueeze(-1).expand(bs, g_size**2, feat_dim))

        # sel_x1_grid_embed = sorted_x1_grid_embed[:,:num]
        # sel_x2_pos_embed = sorted_x2_pos_embed[:,:num]

        # x1_patch = x1_patch + sel_x1_grid_embed #bs,256,1024
        # x2_patch = x2_patch + sel_x2_pos_embed

        # print(x1_patch.size(), indices1.size())
        # sorted_x1 = torch.gather(x1_patch, 1, indices1.unsqueeze(-1).expand(bs, g_size**2, feat_dim))
        # sorted_x2 = torch.gather(x2_patch, 1, indices2.unsqueeze(-1).expand(bs, g_size**2, feat_dim))

        # total = x1_avg.size()[1]
        # num = int(self.r * total)

        # out1_patch = sorted_x1[:,:num]
        # out2_patch = sorted_x2[:,:num]

        out_patch = torch.cat([x1_patch, x2_patch], dim=1) #bs, 256, 1024


        # x_patch = torch.cat([x1_patch, x2_patch], dim=1) # bs, 512, 1024
        # sorted_x = torch.gather(x_patch, 1, indices.unsqueeze(-1).expand(bs, 2*g_size**2, feat_dim))

        # total = x_avg.size()[1] # 512
        # num = int(self.r * total)

        # out_patch = sorted_x[:,:num]

        # print(out_patch.size())
        # out_patch = torch.cat([x1_patch, x2_patch], dim=1) #bs, 256, 1024

        out_patch = self.xfmer_dropout(out_patch)

        out_patch = self.xfmer((out_patch, pred))

        # rest (1-k)% token
        # out1_patch = torch.cat([out_patch[:,:num], sorted_x1[:,num:]],dim=1)
        # out2_patch = torch.cat([out_patch[:,num:], sorted_x2[:,num:]],dim=1)

        # out_patch = torch.cat([out1_patch, out2_patch], dim=1)

        # print(self.r, out_patch.size())
        if self.pool == 'avg':         
            out = torch.mean(out_patch, dim=1)
        elif self.pool == 'max':
            out1_patch, out2_patch = out_patch.chunk(2,dim=1) #bs,256,2048
            '''
            pred_v1 = pred1.unsqueeze(-1).expand(bs, g_size**2, feat_dim) # bs, 256, 2048
            out1_patch = out1_patch * ~pred_v1
                        
            pred_v2 = pred2.unsqueeze(-1).expand(bs, g_size**2, feat_dim) # bs, 256, 2048
            out2_patch = out2_patch * ~pred_v2 # remove black embedding

            out1 = torch.sum(out1_patch, dim=1)/torch.sum((~pred1), dim=1, keepdim=True) #bs,2048
            out2 = torch.sum(out2_patch, dim=1)/torch.sum((~pred2), dim=1, keepdim=True)
            '''
            out1 = torch.mean(out1_patch, dim=1)
            out2 = torch.mean(out2_patch, dim=1)
            out = torch.cat([out1.unsqueeze(-1),out2.unsqueeze(-1)],dim=-1)
            out = self.max1d(out)
            out = out.flatten(start_dim=1)
        elif self.pool == 'cat':
            out1_patch, out2_patch = out_patch.chunk(2,dim=1)
            out1 = torch.mean(out1_patch, dim=1) #bs,2048
            out2 = torch.mean(out2_patch, dim=1)   
            out1 = self.xfmer_layernorm(out1)
            out2 = self.xfmer_layernorm(out2)
            out = torch.cat([out1,out2],dim=-1)

        out = self.xfmer_fc(out)
        return out, x1_avg, x2_avg


    def forward(self, x1, x2, img1_grid, id: Tensor) -> Tensor:
        return self._forward_impl(x1, x2, img1_grid, id)

    def vis_threshold(self, img1, pred1, id):
        # vis_bs = 0
        # mask = torch.zeros(1,1,16*16)
        # indice = indices1[vis_bs][:int(16*16*1)]
        # mask[:,:,indice] = 1
        mask = 1-pred1.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
        # print(mask.size())
        mask = mask.reshape(1,1,16,16)
        mask = F.interpolate(mask, scale_factor=32, mode='bilinear', align_corners=True)
        mask = mask.squeeze(0).permute(1,2,0)

        # mask = torch.rand(512,512,1)
        # img = F.interpolate(img1, scale_factor=1/32, mode='bilinear', align_corners=True)
        heatmap, result = visualize_cam(mask, img1)
        # print(heatmap.size(), result.size())
        path = './hybrid_vis_high_activation_0.1/'
        vutils.save_image(result, path+id+'.jpg')

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict,False) #这里改成False了
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)



def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet50_32x32(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 9], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)



def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)



def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)



def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)



def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)



def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, ipt):
        x, mask = ipt
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        mask1 = mask.unsqueeze(1).expand(B, N, N)
        mask1 = mask1.unsqueeze(1).repeat(1,self.num_heads,1,1)
        attn = attn.data.masked_fill_(mask1.byte(),-float('inf'))

        attn = attn.softmax(dim=-1)

        # mask2 = ~mask.unsqueeze(-1).expand(B, N, N)
        # mask2 = mask2.unsqueeze(1).repeat(1,self.num_heads,1,1)
        # attn = attn * mask2

        # diag_ones = torch.diag_embed(mask) # bs, 512, 512
        # diag_ones = diag_ones.unsqueeze(1).repeat(1,self.num_heads,1,1)
        # attn = attn + diag_ones

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        # mask2 = ~mask.unsqueeze(-1).expand(B, N, C)
        # x = x * mask2

        x = self.proj_drop(x)
        return x

class Attention_inter(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) #B,heads,N,C//heads
        # q1, q2 = q[:,:,:N//2,:], q[:,:,N//2:,:]
        # k1, k2 = k[:,:,:N//2,:], k[:,:,N//2:,:]
        # v1, v2 = v[:,:,:N//2,:], v[:,:,N//2:,:]
        f1 = x[:,:N//2,:]
        f2 = x[:,N//2:,:]

        qkv1 = self.qkv(f1).reshape(B, N//2, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv(f2).reshape(B, N//2, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q1, k1, v1 = qkv1.unbind(0)   # make torchscript happy (cannot use tensor as tuple) #B,heads,N,C//heads
        q2, k2, v2 = qkv2.unbind(0)   # make torchscript happy (cannot use tensor as tuple) #B,heads,N,C//heads

        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)

        x1 = (attn1 @ v2).transpose(1, 2).reshape(B, N//2, C)
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)

        x2 = (attn2 @ v1).transpose(1, 2).reshape(B, N//2, C)
        x2 = self.proj(x2)
        x2 = self.proj_drop(x2)

        x = torch.cat([x1,x2],dim=1)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, inter=False):
        super().__init__()
        self.inter = inter
        self.norm1 = norm_layer(dim)

        if inter:
            self.attn = Attention_inter(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, ipt):
        x, mask = ipt
        B, N, C = x.shape

        x = x + self.drop_path(self.attn((self.norm1(x),mask)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return (x, mask)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if not isinstance(drop, tuple):
            drop_probs = (drop, drop)
        # drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def get_2d_sincos_pos_embed(embed_dim=2048, grid_size=8, grid=None, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if grid is None:
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0) # 2, grid_size, grid_size

        grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed



def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Transformer_head(nn.Module):
    def __init__(self, hidden_size=1024, layers=2, inter=False):
        super().__init__()
        blocks = []
        for i in range(layers):
            blocks.append(Block(hidden_size, 16, inter=inter))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, ipt):
        ipt = self.blocks(ipt)
        x, mask = ipt
        return x



def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

