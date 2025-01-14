import torch.nn as nn
import torch
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def __repr__(self):
        return f"{self.__class__.__name__}(p='{self.p.data.tolist()[0]:.4f}', eps='{str(self.eps)}')"

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    @staticmethod
    def gem(x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class BaseNet(nn.Module):
    def __init__(self, backbone, global_pool=None, norm=None, p=3):
        super(BaseNet, self).__init__()
        self.backbone = backbone
        n = 0
        for name, param in self.backbone.named_parameters():
            n = param.size()[0]
        self.feature_length = n
        match global_pool:
            case "max":
                self.pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
            case "avg":
                self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            case "GeM":
                self.pool = GeM(p=p)
            case _:
                self.pool = None
        self.norm = norm

    def forward(self, x0):
        out = self.backbone.forward(x0)
        out = self.pool.forward(out).squeeze(-1).squeeze(-1)
        if self.norm == "L2":
            out = nn.functional.normalize(out)
        return out


class SiameseNet(BaseNet):
    def __init__(self, backbone, global_pool=None, norm=None, p=3):
        super(SiameseNet, self).__init__(backbone, global_pool, norm=norm, p=p)

    def forward_single(self, x0):
        return super(SiameseNet, self).forward(x0)

    def forward(self, x0, x1):
        out0 = super(SiameseNet, self).forward(x0)
        out1 = super(SiameseNet, self).forward(x1)
        return out0, out1
