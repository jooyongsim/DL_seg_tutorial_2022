from typing import List, Optional, Dict
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision
import torchvision.transforms.functional as TF

from utils import IntermediateLayerGetter

__all__ = [
    "UNET",
    "DeepLabV3",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "deeplabv3_mobilenet_v3_large",
]

class DConv(nn.Module):
  def __init__(self, in_channel, out_channel):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel,3,1,1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel,3,1,1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
    )
  def forward(self, x):
    return self.conv(x)

class UNET(nn.Module):
  """Constructs a DeepLabV3 model with a ResNet-101 backbone.

  Args:
      in_channels (int) : The number of channels of input images
      num_classes (int): The number of classes
      aux_loss (bool, optional): False, we do not use auxia auxiliary classifier
      features (list): The number of featueres in down paths
  """
  def __init__(self, in_channels = 3, num_classes = 21, aux_loss = False, features=[64,128,256,512]):
    super().__init__()
    self.downs = nn.ModuleList()
    for feature in features:
      self.downs.append(DConv(in_channels, feature))
      in_channel = feature
    self.pool = nn.MaxPool2d(2,2)
    self.bottleneck = DConv(features[-1], features[-1]*2)
    self.ups = nn.ModuleList()
    features = features[::-1]
    for feature in features:
      self.ups.append(
          nn.ModuleList([
                         nn.ConvTranspose2d(feature*2,feature,2,2),
                         DConv(feature*2, feature),
          ])
      )
    self.head = nn.Conv2d(features[-1], num_classes, kernel_size=1)

  def forward(self,x):
    # down path
    skips = []
    for down in self.downs:
      x = down(x)
      skips.append(x)
      x = self.pool(x)
    x = self.bottleneck(x)
    skips = skips[::-1]

    # up path
    for ind, up in enumerate(self.ups):
      x = up[0](x)
      if x.shape != skips[ind].shape:
          x = TF.resize(x, size=skips[ind].shape[2:])
      x = torch.cat((skips[ind],x),dim=1)
      x = up[1](x)
    x = self.head(x)
    
    result = OrderedDict()
    result["out"] = x
    
    return result

class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)


class DeepLabV3(nn.Module):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    __constants__ = ["aux_classifier"]

    def __init__(self, backbone: nn.Module, classifier: nn.Module, aux_classifier: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        return result


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


def _deeplabv3_resnet(
    backbone: torchvision.models.resnet.ResNet,
    num_classes: int,
    aux: Optional[bool],
) -> DeepLabV3:
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux else None
    classifier = DeepLabHead(2048, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier)


def _deeplabv3_mobilenetv3(
    backbone: torchvision.models.mobilenetv3.MobileNetV3,
    num_classes: int,
    aux: Optional[bool],
) -> DeepLabV3:
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    out_inplanes = backbone[out_pos].out_channels
    aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    aux_inplanes = backbone[aux_pos].out_channels
    return_layers = {str(out_pos): "out"}
    if aux:
        return_layers[str(aux_pos)] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(aux_inplanes, num_classes) if aux else None
    classifier = DeepLabHead(out_inplanes, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier)


def deeplabv3_resnet50(
    in_channels: int = 3,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        in_channel (int) : The number of channels of input images
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """

    backbone = torchvision.models.resnet.resnet50(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

    return model


def deeplabv3_resnet101(
    in_channels: int = 3,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        in_channel (int) : The number of channels of input images
        num_classes (int): The number of classes
        aux_loss (bool, optional): If True, include an auxiliary classifier
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """

    backbone = torchvision.models.resnet.resnet101(pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)
    return model


def deeplabv3_mobilenet_v3_large(
    in_channels: int = 3,
    num_classes: int = 21,
    aux_loss: Optional[bool] = None,
    pretrained_backbone: bool = True,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a MobileNetV3-Large backbone.

    Args:
        in_channel (int) : The number of channels of input images
        num_classes (int): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        pretrained_backbone (bool): If True, the backbone will be pre-trained.
    """

    backbone = torchvision.models.mobilenetv3.mobilenet_v3_large(pretrained=pretrained_backbone, dilated=True)
    model = _deeplabv3_mobilenetv3(backbone, num_classes, aux_loss)

    return model
