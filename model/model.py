from .model_util import *
#from .seg_model import DeeplabMulti
from .pspnet import DeeplabMulti

from scipy.ndimage.filters import gaussian_filter
import functools
import numpy as np
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

pspnet_specs = {
    'n_classes': 19,
    'input_size': (713, 713),
    'block_config': [3, 4, 23, 3],
}

class SharedEncoder(nn.Module):
    def __init__(self, initialization=None, bn_clr=False):
        super(SharedEncoder, self).__init__()
        self.n_classes = pspnet_specs['n_classes']
        self.bn_clr = bn_clr
        self.initialization = initialization

        model_seg = DeeplabMulti(pretrained=True, num_classes=self.n_classes, initialization = self.initialization, bn_clr =self.bn_clr)

        self.layer0 = nn.Sequential(model_seg.conv1, model_seg.bn1, model_seg.relu, model_seg.maxpool)
        self.layer1 = model_seg.layer1
        self.layer2 = model_seg.layer2
        self.layer3 = model_seg.layer3
        self.layer4 = model_seg.layer4
        if self.bn_clr:
            self.bn_pretrain = model_seg.bn_pretrain
        self.final1 = model_seg.layer5
        self.final2 = model_seg.layer6

    def forward(self, x):
        #inp_shape = x.shape[2:]

        x = self.layer0(x)
        # [2, 64, 65, 129]
        x = self.layer1(x)
        x = self.layer2(x)
        shared_shallow = x
        #4*512*33*65

        x = self.layer3(x)
        pred1 = self.final1(x)

        shared_seg = self.layer4(x)
        if self.bn_clr:
            shared_seg = self.bn_pretrain(shared_seg)
        pred2 = self.final2(shared_seg)

        return shared_shallow, pred1, pred2, shared_seg

    def get_1x_lr_params_NOscale(self):
        b = []

        b.append(self.layer0)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        b = []
        b.append(self.final1.parameters())
        if self.bn_clr:
            b.append(self.bn_pretrain.parameters())
        b.append(self.final2.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': 1 * learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]


class Classifier(nn.Module):
    def __init__(self, inp_shape):
        super(Classifier, self).__init__()
        n_classes = pspnet_specs['n_classes']
        self.inp_shape = inp_shape

        # PSPNet_Model = PSPNet(pretrained=True)

        self.dropout = nn.Dropout2d(0.1)
        self.cls = nn.Conv2d(512, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.cls(x)
        x = F.upsample(x, size=self.inp_shape, mode='bilinear')
        return x


class Discriminator_MS(nn.Module):
    def __init__(self):
        super(Discriminator_MS, self).__init__()
        # FCN classification layer
        self.dim = 64
        self.n_layer = 4
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.num_scales = 3
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(3, dim, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dim = 64

        self.cnn_x = nn.Sequential(Conv2dBlock(3, self.dim, 3, 1, 1, norm='none',activation='lrelu', bias=False),
                                 Conv2dBlock(self.dim, 2 * self.dim, 4, 2, 1, norm='none',activation='lrelu', bias=False),
                                 Conv2dBlock(2 * self.dim, 4 * self.dim, 4, 2, 1, norm='none',activation='lrelu', bias=False))
        self.glob = nn.Sequential(Conv2dBlock(4*self.dim, 4*self.dim, 4, 2, 1,norm='none',activation='lrelu', bias=False),
                                 Conv2dBlock(4 * self.dim, 8 * self.dim, 4, 2, 1, norm='none',activation='lrelu', bias=False),
                                 Conv2dBlock(8 * self.dim, 8 * self.dim, 4, 2, 1, norm='none',activation='lrelu', bias=False),
                                 nn.Conv2d(8*self.dim, 1, 1, 1, 0))
        self.loc = nn.Sequential(nn.Upsample(scale_factor=2),
                                  Conv2dBlock(4*self.dim, 2*self.dim, 3, 1, 1, norm='none',activation='lrelu', bias=False),
                                  nn.Upsample(scale_factor=2),
                                  Conv2dBlock(2*self.dim, self.dim, 3, 1, 1, norm='none',activation='lrelu', bias=False),
                                  nn.Conv2d(self.dim, 1, 1, 1, 0))

    def forward(self, x):
        x = self.cnn_x(x)
        loc = self.loc(x)
        glob = self.glob(x)
        return [glob, loc]


class SegDiscriminator(nn.Module):
    def __init__(self):
        super(SegDiscriminator, self).__init__()
        n_classes = pspnet_specs['n_classes']
        # FCN classification layer

        self.feature = nn.Sequential(
            Conv2dBlock(n_classes, 64, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(64, 128, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(128, 256, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(256, 512, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            nn.Conv2d(512, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.feature(x)
        return x


class SABlock(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                        nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        return torch.mul(features, attention_mask)
