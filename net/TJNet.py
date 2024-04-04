import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from net.Res2Net import res2net50_v1b_26w_4s


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride,
                      padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class FOM(nn.Module):
    def __init__(self, in_channel, out_channel, need_relu=True):
        super(FOM, self).__init__()
        self.need_relu = need_relu
        self.relu = nn.ReLU(True)
        self.conv0 = BasicConv2d(in_channel, out_channel, 3, padding=1)
        self.conv1 = BasicConv2d(out_channel, out_channel, 3, padding=1)
        self.conv2 = BasicConv2d(out_channel, out_channel, 3, padding=1)
        self.conv3 = BasicConv2d(out_channel, out_channel, 3, padding=1)
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = x0 + self.conv1(x0)
        x2 = x1 + self.conv2(x1)
        x3 = x2 + self.conv3(x2)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        if self.need_relu:
            x = self.relu(x_cat + self.conv_res(x))
        else:
            x = x_cat + self.conv_res(x)
        return x


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(256, 64)
        self.reduce4 = Conv1x1(2048, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out


class IGM(nn.Module):
    def __init__(self, channel):
        super(IGM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k,
                                padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.fom = FOM(1, 1)
        self.sigmoid_coarse = nn.Sigmoid()

    def forward(self, c, edge_att, coarse_att):
        if c.size() != edge_att.size():
            edge_att = F.interpolate(
                edge_att, c.size()[2:], mode='bilinear', align_corners=False)
        if c.size() != coarse_att.size():
            coarse_att = F.interpolate(
                coarse_att, c.size()[2:], mode='bilinear', align_corners=False)
        coarse_att1 = self.fom(coarse_att)
        coarse_att2 = self.sigmoid_coarse(coarse_att1)
        x = c * (edge_att+coarse_att2) + c
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)
                          ).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei

        return c + x


class FullImageAttention(nn.Module):
    def __init__(self, channel):
        super(FullImageAttention, self).__init__()
        self.conv1 = FOM(2 * channel, channel, True)
        self.conv2 = FOM(3 * channel, channel, False)
        self.conv3 = BasicConv2d(channel, channel, 3, padding=1)

    def forward(self, x, y):
        k = self.conv1(torch.cat((x, y), 1))
        z = x + y * k
        b = self.conv2(torch.cat((x, y, z), 1))
        ret = self.conv3(z + b)
        return ret


class UnNamedModule(nn.Module):
    def __init__(self, channel):
        super(UnNamedModule, self).__init__()
        self.fia1 = FullImageAttention(channel)
        self.fia2 = FullImageAttention(channel)
        self.fia3 = FullImageAttention(channel)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = BasicConv2d(channel, channel, 1)
        self.conv2 = nn.Conv2d(channel, 1, 1)

    def forward(self, x1, x2, x3, x4):
        x2a = self.fia1(x2, self.upsample(x1))
        x3a = self.fia2(x3, self.upsample(x2a))
        x4a = self.fia3(x4, self.upsample(x3a))
        x = self.conv1(x4a)
        ret = self.conv2(x)
        return ret


class Network(nn.Module):
    def __init__(self, channel=64):
        super(Network, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # if self.training:
        # self.initialize_weights()

        self.eam_edge = EAM()
        self.UNM = UnNamedModule(channel)

        # ---- Receptive Field Block like module ----
        self.fom1 = FOM(256, channel)
        self.fom2 = FOM(512, channel)
        self.fom3 = FOM(1024, channel)
        self.fom4 = FOM(2048, channel)

        self.igm1 = IGM(channel)
        self.igm2 = IGM(channel)
        self.igm3 = IGM(channel)
        self.igm4 = IGM(channel)

        self.fia1 = FullImageAttention(channel)
        self.fia2 = FullImageAttention(channel)
        self.fia3 = FullImageAttention(channel)

        self.predictor1 = nn.Conv2d(channel, 1, 1)
        self.predictor2 = nn.Conv2d(channel, 1, 1)
        self.predictor3 = nn.Conv2d(channel, 1, 1)

    def forward(self, x):
        x1, x2, x3, x4 = self.resnet(x)

        edge = self.eam_edge(x4, x1)
        edge_att = torch.sigmoid(edge)

        # FOM
        x1_fom = self.fom1(x1)
        x2_fom = self.fom2(x2)
        x3_fom = self.fom3(x3)
        x4_fom = self.fom4(x4)

        # UNM
        coarse_att = self.UNM(x4_fom, x3_fom, x2_fom, x1_fom)

        # IGM
        x1a = self.igm1(x1_fom, edge_att, coarse_att)
        x2a = self.igm2(x2_fom, edge_att, coarse_att)
        x3a = self.igm3(x3_fom, edge_att, coarse_att)
        x4a = self.igm4(x4_fom, edge_att, coarse_att)

        # FIA
        x4au = F.interpolate(x4a, size=x3a.size()[
                             2:], mode='bilinear', align_corners=False)
        x34 = self.fia1(x3a, x4au)
        x34u = F.interpolate(x34, size=x2a.size()[
                             2:], mode='bilinear', align_corners=False)
        x234 = self.fia2(x2a, x34u)
        x234u = F.interpolate(x234, size=x1a.size()[
                              2:], mode='bilinear', align_corners=False)
        x1234 = self.fia3(x1a, x234u)

        o3 = self.predictor3(x34)
        o3 = F.interpolate(o3, scale_factor=16,
                           mode='bilinear', align_corners=False)
        o2 = self.predictor2(x234)
        o2 = F.interpolate(o2, scale_factor=8,
                           mode='bilinear', align_corners=False)
        o1 = self.predictor1(x1234)
        o1 = F.interpolate(o1, scale_factor=4,
                           mode='bilinear', align_corners=False)
        oe = F.interpolate(edge_att, scale_factor=4,
                           mode='bilinear', align_corners=False)

        oc = F.interpolate(coarse_att, scale_factor=4,
                           mode='bilinear', align_corners=False)

        return o3, o2, o1, oe, oc
