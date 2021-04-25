import torch
from torch import nn
import torch.nn.functional as F


from .resnet import resnet50
from .seg_detector import SegDetector
from .loss_pan import PANLoss


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.backbone = resnet50(pretrained=config['model']['pretrained'])
        self.decoder = SegDetector(in_channels=[256, 512, 1024, 2048],
                                    k=50, adaptive=True)
        self.criterion = PANLoss(alpha=config['loss']['alpha'],
                                beta=config['loss']['beta'],
                                delta_agg=config['loss']['delta_agg'],
                                delta_dis=config['loss']['delta_dis'],
                                ohem_ratio=config['loss']['ohem_ratio'])

        if config['model']['resume'] != "":
            self.load_model(config['model']['resume'])
            self.epoch = config['model']['epoch']
            self.iter = config['model']['iter']
            print('load model: {}'.format(config['model']['resume']))
        else:
            self.epoch = 0
            self.iter = 0


    def to_device(self, device):
        self.backbone.to(device)
        self.decoder.to(device)
        

    def to_train(self):
        self.backbone.train()
        self.decoder.train()

    def to_eval(self):
        self.backbone.eval()
        self.decoder.eval()

    def get_parameters(self):
        return list(self.backbone.parameters()) + list(self.decoder.parameters())

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location='cuda')
        self.load_state_dict(checkpoint)

    def forward(self, x):

        features = self.backbone(x)
        decoder_out = self.decoder(features)
        pred = F.interpolate(decoder_out, size=(640, 640), mode='bilinear', align_corners=True)

        return pred
