from src.datasets import *
from torch.utils.data import DataLoader
from src.networks import *
from torchvision import models


def create_dataloader(dataset, root_dir, idx_file, gt_file, image_t, batch_size):
    match dataset:
        case "test":
            ds = TestDataSet(root_dir, idx_file, transform=image_t)
            return DataLoader(ds, batch_size=batch_size, num_workers=4)
        case "soft_siamese":
            ds = SiameseDataSet(root_dir, idx_file, gt_file, ds_key="fov", transform=image_t)
        case "binary_siamese":
            ds = SiameseDataSet(root_dir, idx_file, gt_file, ds_key="sim", transform=image_t)
        case _:
            raise ValueError(f"{dataset} is not valid dataset!")
    return DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=True)


def create_msls_dataloader(dataset, root_dir, cities, transform, batch_size):
    match dataset:
        case "binary_MSLS":
            ds = MSLSDataSet(root_dir, cities, ds_key="sim", transform=transform)
        case "soft_MSLS":
            ds = MSLSDataSet(root_dir, cities, ds_key="fov", transform=transform)
        case _:
            raise ValueError(f"{dataset} is not a valid dataset!")
    return DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=True)


def get_backbone(name):
    match name:
        case "resnet18":
            backbone = models.resnet18(pretrained=True)
            backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
            output_dim = 2048
        case "resnet34":
            backbone = models.resnet34(pretrained=True)
            backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
            output_dim = 2048
        case "resnet152":
            backbone = models.resnet152(pretrained=True)
            backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
            output_dim = 2048
        case "resnet50":
            backbone = models.resnet50(pretrained=True)
            backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
            output_dim = 2048
        case "resnext":
            backbone = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
            backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
            output_dim = 2048
        case "densenet161":
            backbone = models.densenet161(pretrained=True).features
            output_dim = 2208
        case "densenet121":
            backbone = models.densenet121(pretrained=True).features
            output_dim = 2208
        case "vgg16":
            backbone = models.vgg16(pretrained=True).features
            output_dim = 512
        case _:
            raise ValueError(f"{name} is not a valid backbone!")
    return backbone, output_dim


def create_model(name, pool, last_layer=None, norm=None, p_gem=3, mode="siamese"):
    backbone, output_dim = get_backbone(name)
    layers = len(list(backbone.children()))

    if last_layer is None:
        last_layer = layers
    elif "densenet" in name:
        last_layer = last_layer * 2
    elif "vgg" in name:
        last_layer = last_layer * 8 - 2

    aux = 0
    for layer in backbone.children():
        if aux < layers - last_layer:
            print(aux, layer._get_name(), "IS FROZEN")
            for p in layer.parameters():
                p.requires_grad = False
        else:
            print(aux, layer._get_name(), "IS TRAINED")
        aux += 1

    if mode == "siamese":
        return SiameseNet(backbone, pool, norm=norm, p=p_gem)
    else:
        return BaseNet(backbone, pool, norm=norm, p=p_gem)
