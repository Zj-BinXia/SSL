import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN_WN, flow_warp, make_layer, Conv2D_WN, Conv2D_WN_upconv
from .spynet_arch import SpyNet


@ARCH_REGISTRY.register()
class BasicVSR(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=15, spynet_path=None):
        super().__init__()
        self.num_feat = num_feat
        self.pruned = False
        self.pruned_final = False
        self.isfull =False
        self.num_feat_backward = num_feat
        self.num_feat_forward = num_feat
        self.kept_wg_forward = None
        self.kept_wg_pre_forward=None
        self.kept_wg_backward = None
        self.kept_wg_pre_backward=None
        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # reconstruction
        self.fusion = Conv2D_WN(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = Conv2D_WN_upconv(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = Conv2D_WN_upconv(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = Conv2D_WN(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.ReLU( inplace=True)

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        if self.pruned:
            flows_forward, flows_backward = self.get_flow(x)
            b, n, _, h, w = x.size()

            # backward branch
            out_l = []
            feat_prop = x.new_zeros(b, self.num_feat, h, w)
            for i in range(n - 1, -1, -1):
                x_i = x[:, i, :, :, :]
                if i < n - 1:
                    flow = flows_backward[:, i, :, :, :]
                    feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                feat_prop = torch.cat([x_i, feat_prop], dim=1)
                feat_prop = self.backward_trunk(feat_prop,self.pruned, self.pruned_final,self.kept_wg_backward,self.kept_wg_pre_backward)
                out_l.insert(0, feat_prop)


            # forward branch
            feat_prop = x.new_zeros(b, self.num_feat, h, w)
            for i in range(0, n):
                x_i = x[:, i, :, :, :]
                if i > 0:
                    flow = flows_forward[:, i - 1, :, :, :]
                    feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                feat_prop = torch.cat([x_i, feat_prop], dim=1)
                feat_prop = self.forward_trunk(feat_prop, self.pruned,self.pruned_final,self.kept_wg_forward,self.kept_wg_pre_forward)


                out = torch.cat([out_l[i], feat_prop], dim=1)
                out = self.lrelu(self.fusion(out))
                out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
                out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
                out = self.lrelu(self.conv_hr(out))
                out = self.conv_last(out)
                base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
                out += base
                out_l[i] = out

            return torch.stack(out_l, dim=1)

        else:
            flows_forward, flows_backward = self.get_flow(x)
            b, n, _, h, w = x.size()

            # backward branch
            out_l = []

            feat_prop = x.new_zeros(b, self.num_feat, h, w)
            for i in range(n - 1, -1, -1):
                x_i = x[:, i, :, :, :]
                if i < n - 1:
                    flow = flows_backward[:, i, :, :, :]
                    feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                feat_prop = torch.cat([x_i, feat_prop], dim=1)
                feat_prop = self.backward_trunk(feat_prop, self.pruned)
                out_l.insert(0, feat_prop)

            # forward branch
            feat_prop = x.new_zeros(b, self.num_feat, h, w)
            for i in range(0, n):
                x_i = x[:, i, :, :, :]
                if i > 0:
                    flow = flows_forward[:, i - 1, :, :, :]
                    feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                feat_prop = torch.cat([x_i, feat_prop], dim=1)
                feat_prop = self.forward_trunk(feat_prop,self.pruned)

                out = torch.cat([out_l[i], feat_prop], dim=1)
                out = self.lrelu(self.fusion(out))
                out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
                out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
                out = self.lrelu(self.conv_hr(out))
                out = self.conv_last(out)
                base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
                out += base
                out_l[i] = out


            return torch.stack(out_l, dim=1)


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.ReLU(inplace=True),
            make_layer(ResidualBlockNoBN_WN, num_block, num_feat=num_out_ch))

    def forward(self, fea, pruned=False, pruned_final=False,kept_wg=None, kept_wg_pre = None ):
        fea = self.main[0](fea)
        fea = self.main[1](fea)
        for j in range(len(self.main[2])):
            if pruned:
                fea = self.main[2][j](fea,kept_wg[j],kept_wg_pre[j],pruned_final=pruned_final)
            else:
                fea = self.main[2][j](fea,pruned_final=pruned_final)
        return fea


