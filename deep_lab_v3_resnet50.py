from torch import nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import deeplabv3_resnet50

def replace_batchnorm_with_groupnorm(model):
    for name, module in list(model.named_children()):
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            gn = nn.GroupNorm(num_groups=32, num_channels=num_features)
            setattr(model, name, gn)
        else:
            replace_batchnorm_with_groupnorm(module) 
    return model

def custom_DeepLabv3(out_channel):
    model = deeplabv3_resnet50(weights="DEFAULT", progress=True)
    model.classifier = DeepLabHead(1024*1024, out_channel)

    model = replace_batchnorm_with_groupnorm(model)

    return model
