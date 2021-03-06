# coding=utf-8
"""
@File   : model_Ablation.py
@Time   : 2020/09/05
@Author : Junjie Wang
"""
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.heads import SegmentationHead
from segmentation_models_pytorch.unet.decoder import UnetDecoder
import torch.nn as nn
from torchviz import make_dot
from torch.autograd import Variable
from torchsummary import summary
import torch

class model(nn.Module):
    def __init__(self, encoder_depth=4, decoder_channels=None):
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [128, 64, 32, 16]
        net = smp.Unet(classes=1,
                       encoder_depth=encoder_depth,
                       decoder_channels=decoder_channels,
                       encoder_name='senet154',
                       activation='sigmoid')
        self.encoder = net.encoder
        # self.SegDecoder = net.decoder
        self.HVDecoder = UnetDecoder(encoder_channels=self.encoder.out_channels,
                                     decoder_channels=decoder_channels,
                                     n_blocks=encoder_depth)
        self.seDecoder = net.decoder
        # self.segmentation_head = net.segmentation_head
        self.horizontalVertical_head = SegmentationHead(in_channels=16,
                                                        out_channels=2,
                                                        kernel_size=3)
        self.semantics_head = SegmentationHead(in_channels=16,
                                               out_channels=3,
                                               kernel_size=3)

    def forward(self, x):
        features = self.encoder(x)
        # segmentation = self.segmentation_head(self.SegDecoder(*features))
        horizontalVertical = self.horizontalVertical_head(self.HVDecoder(*features))
        semantics = self.semantics_head(self.seDecoder(*features))

        # return (segmentation, horizontalVertical, semantics)
        # return (segmentation, semantics)
        # return (segmentation, semantics)
        return (horizontalVertical, semantics)
        # return (semantics)

if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    net = model().to(device)
    print(net)
    summary(net, (3, 32, 32))
    #
    # net = model()
    # x = Variable(torch.randn(1, 3, 384, 384))
    # y = net(x)
    # vis_graph = make_dot(y, params=dict(list(net.named_parameters()) + [('x', x)]))
    # vis_graph.view()
