import torch
import torch.nn as nn
import copy
import time
import numpy as np
from math import ceil, sqrt
from collections import OrderedDict
from putils import strdict_to_dict
from fnmatch import fnmatch, fnmatchcase
from .layer import LayerStruct
from .utils import get_pr_model, get_constrained_layers, pick_pruned_model, get_kept_filter_channel, replace_module, get_next_bn
from .utils import get_masks
from basicsr.archs.arch_util import  Conv2D_WN

class MetaPruner:
    def __init__(self, model, args, opt, logger):
        self.model = model
        self.opt = opt
        self.args = args
        self.logger = logger
        self.num_feat_forward = None
        self.num_feat_backward = None
        # set up layers
        self.LEARNABLES = (nn.Conv2d, nn.Linear) # the layers we focus on for pruning
        layer_struct = LayerStruct(model, self.LEARNABLES,self.logger)
        self.layers = layer_struct.layers
        self._max_len_ix = layer_struct._max_len_ix
        self._max_len_name = layer_struct._max_len_name
        self.layer_print_prefix = layer_struct.print_prefix

        # set up pr for each layer
        self.raw_pr = get_pr_model(self.layers, args.stage_pr, skip=args.skip_layers, compare_mode=args.compare_mode)
        self.pr = copy.deepcopy(self.raw_pr)

        # pick pruned and kept weight groups
        self.constrained_layers = get_constrained_layers(self.layers, self.args.same_pruned_wg_layers)
        self.constrained_layers_backward = get_constrained_layers(self.layers, self.args.same_pruned_wg_layers_backward)
        self.logger.info(f'Constrained layers: {self.constrained_layers}')
        self.logger.info(f'Constrained layers_backward: {self.constrained_layers_backward}')

    def _get_kept_wg_L1(self, align_constrained=False):
        # ************************* core pruning function **************************
        self.pr, self.pruned_wg, self.kept_wg ,self.pruned_wg_pre, self.kept_wg_pre = pick_pruned_model(self.model, self.layers, self.raw_pr,
                                                        wg=self.args.wg, 
                                                        criterion=self.args.prune_criterion,
                                                        compare_mode=self.args.compare_mode,
                                                        sort_mode=self.args.pick_pruned,
                                                        constrained=self.constrained_layers,
                                                        constrained_backward = self.constrained_layers_backward,
                                                        align_constrained=align_constrained)
        # ***************************************************************************
        


    def _prune_and_build_new_model(self):
        if self.args.wg == 'weight':
            self.masks = get_masks(self.layers, self.pruned_wg)
            return
        if self.opt["dist"]:
            name_st = ["module.forward_trunk.main.0", "module.backward_trunk.main.0"]
            name_ed = ["module.forward_trunk.main.2.29.conv2", "module.backward_trunk.main.2.29.conv2"]
            name_forward = "module.forward_trunk.main.2.29.conv2"
            name_backward = "module.backward_trunk.main.2.29.conv2"
        else:
            name_st = ["forward_trunk.main.0" , "backward_trunk.main.0"]
            name_forward ="forward_trunk.main.2.29.conv2"
            name_backward = "backward_trunk.main.2.29.conv2"
        new_model = copy.deepcopy(self.model)
        for name, m in self.model.named_modules():
            if isinstance(m, self.LEARNABLES):
                kept_filter, kept_chl = get_kept_filter_channel(self.layers, name, m, pr=self.pr, kept_wg=self.kept_wg,kept_wg_pre=self.kept_wg_pre,opt=self.opt, wg=self.args.wg,name_st =name_st)
                
                # decide if renit the current layer
                reinit = False
                for rl in self.args.reinit_layers:
                    if fnmatch(name, rl):
                        reinit = True
                        break
                
                # get number of channels (can be manually assigned)
                num_chl = self.args.layer_chl[name] if name in self.args.layer_chl else len(kept_chl)

                # copy weight and bias
                bias = False if isinstance(m.bias, type(None)) else True
                if isinstance(m, nn.Conv2d):
                    if "upconv" in name:
                        scale = 2
                        num_fea=64
                        new_layer = nn.Conv2d(num_chl, len(kept_filter)*scale*scale, m.kernel_size,
                                              m.stride, m.padding, m.dilation, m.groups, bias).cuda()
                        if not reinit:
                            kept_weights = m.weight.data.view(-1,num_fea*scale*scale,3,3)
                            kept_weights = kept_weights[kept_filter]
                            kept_weights = kept_weights.view(-1,num_fea,3,3)
                            kept_weights = kept_weights[:, kept_chl, :, :]
                    else:

                        new_layer = nn.Conv2d(num_chl, len(kept_filter), m.kernel_size,
                                        m.stride, m.padding, m.dilation, m.groups, bias).cuda()
                        if not reinit:
                            kept_weights = m.weight.data[kept_filter][:, kept_chl, :, :]
                    if  name_forward in name:
                        self.num_feat_forward = len(kept_filter)
                    if  name_backward in name:
                        self.num_feat_backward = len(kept_filter)


                elif isinstance(m, nn.Linear):
                    kept_weights = m.weight.data[kept_filter][:, kept_chl]
                    new_layer = nn.Linear(in_features=len(kept_chl), out_features=len(kept_filter), bias=bias).cuda()
                
                if not reinit:
                    new_layer.weight.data.copy_(kept_weights) # load weights into the new module
                    if bias:
                        if "upconv" in name:
                            kept_bias = m.bias.data.view(num_fea,scale*scale)[kept_filter].view(-1)
                        else:
                            kept_bias = m.bias.data[kept_filter]
                        new_layer.bias.data.copy_(kept_bias)
                
                # load the new conv
                replace_module(new_model, name, new_layer)

                # get the corresponding bn (if any) for later use
                next_bn = get_next_bn(self.model, m)

            elif isinstance(m, nn.BatchNorm2d) and m == next_bn:
                new_bn = nn.BatchNorm2d(len(kept_filter), eps=m.eps, momentum=m.momentum, 
                        affine=m.affine, track_running_stats=m.track_running_stats).cuda()
                
                # copy bn weight and bias
                if self.args.copy_bn_w:
                    weight = m.weight.data[kept_filter]
                    new_bn.weight.data.copy_(weight)
                if self.args.copy_bn_b:
                    bias = m.bias.data[kept_filter]
                    new_bn.bias.data.copy_(bias)
                
                # copy bn running stats
                new_bn.running_mean.data.copy_(m.running_mean[kept_filter])
                new_bn.running_var.data.copy_(m.running_var[kept_filter])
                new_bn.num_batches_tracked.data.copy_(m.num_batches_tracked)
                
                # load the new bn
                replace_module(new_model, name, new_bn)

        self.model = new_model


        if self.opt["dist"]:
            self.model.module.num_feat_backward = self.num_feat_backward
            self.model.module.num_feat_forward = self.num_feat_forward
            self.model.module.pruned = True
        else:
            self.model.num_feat_backward = self.num_feat_backward
            self.model.num_feat_forward = self.num_feat_forward
            self.model.pruned = True
        # print the layer shape of pruned model
        LayerStruct(new_model, self.LEARNABLES,self.logger)
        return new_model

    def _prune_and_build_new_modelV2(self):
        if self.args.wg == 'weight':
            self.masks = get_masks(self.layers, self.pruned_wg)
            return
        if self.opt["dist"]:
            name_st = ["module.forward_trunk.main.0", "module.backward_trunk.main.0"]
            name_ed = ["module.forward_trunk.main.2.29.conv2", "module.backward_trunk.main.2.29.conv2"]
            name_forward = "module.forward_trunk.main.2.29.conv2"
            name_backward = "module.backward_trunk.main.2.29.conv2"
        else:
            name_st = ["forward_trunk.main.0", "backward_trunk.main.0"]
            name_forward = "forward_trunk.main.2.29.conv2"
            name_backward = "backward_trunk.main.2.29.conv2"
        new_model = copy.deepcopy(self.model)
        for name, m in self.model.named_modules():
            if isinstance(m, self.LEARNABLES):
                kept_filter, kept_chl = get_kept_filter_channel(self.layers, name, m, pr=self.pr, kept_wg=self.kept_wg,
                                                                kept_wg_pre=self.kept_wg_pre, opt=self.opt,
                                                                wg=self.args.wg, name_st=name_st)

                # decide if renit the current layer
                reinit = False
                for rl in self.args.reinit_layers:
                    if fnmatch(name, rl):
                        reinit = True
                        break

                # get number of channels (can be manually assigned)
                num_chl = self.args.layer_chl[name] if name in self.args.layer_chl else len(kept_chl)

                # copy weight and bias
                bias = False if isinstance(m.bias, type(None)) else True
                if isinstance(m, nn.Conv2d):
                    if "upconv" in name:
                        scale = 2
                        num_fea = 64
                        new_layer = nn.Conv2d(num_chl, len(kept_filter) * scale * scale, m.kernel_size,
                                              m.stride, m.padding, m.dilation, m.groups, bias).cuda()
                        if not reinit:
                            kept_weights = m.weight.data.view(-1, num_fea * scale * scale, 3, 3)
                            kept_weights = kept_weights[kept_filter]
                            kept_weights = kept_weights.view(-1, num_fea, 3, 3)
                            kept_weights = kept_weights[:, kept_chl, :, :]
                    elif "trunk.main" in name and "conv2" in name:
                        num_fea = 64
                        new_layer = Conv2D_WN(num_chl, num_fea, m.kernel_size,
                                              m.stride, m.padding, m.dilation, m.groups, bias).cuda()
                        new_layer.act_scale.data.copy_(m.act_scale.data)
                        new_layer.act_scale.requires_grad=False
                        if not reinit:
                            kept_weights = m.weight.data[:, kept_chl, :, :]
                    else:

                        new_layer = nn.Conv2d(num_chl, len(kept_filter), m.kernel_size,
                                              m.stride, m.padding, m.dilation, m.groups, bias).cuda()
                        if not reinit:
                            kept_weights = m.weight.data[kept_filter][:, kept_chl, :, :]
                    if name_forward in name:
                        self.num_feat_forward = len(kept_filter)
                    if name_backward in name:
                        self.num_feat_backward = len(kept_filter)


                elif isinstance(m, nn.Linear):
                    kept_weights = m.weight.data[kept_filter][:, kept_chl]
                    new_layer = nn.Linear(in_features=len(kept_chl), out_features=len(kept_filter), bias=bias).cuda()

                if not reinit:
                    new_layer.weight.data.copy_(kept_weights)  # load weights into the new module
                    if bias:
                        if "upconv" in name:
                            kept_bias = m.bias.data.view(num_fea, scale * scale)[kept_filter].view(-1)
                        elif "trunk.main" in name and "conv2" in name:
                            kept_bias = m.bias.data
                        else:
                            kept_bias = m.bias.data[kept_filter]
                        new_layer.bias.data.copy_(kept_bias)

                # load the new conv
                replace_module(new_model, name, new_layer)

                # get the corresponding bn (if any) for later use
                next_bn = get_next_bn(self.model, m)

            elif isinstance(m, nn.BatchNorm2d) and m == next_bn:
                new_bn = nn.BatchNorm2d(len(kept_filter), eps=m.eps, momentum=m.momentum,
                                        affine=m.affine, track_running_stats=m.track_running_stats).cuda()

                # copy bn weight and bias
                if self.args.copy_bn_w:
                    weight = m.weight.data[kept_filter]
                    new_bn.weight.data.copy_(weight)
                if self.args.copy_bn_b:
                    bias = m.bias.data[kept_filter]
                    new_bn.bias.data.copy_(bias)

                # copy bn running stats
                new_bn.running_mean.data.copy_(m.running_mean[kept_filter])
                new_bn.running_var.data.copy_(m.running_var[kept_filter])
                new_bn.num_batches_tracked.data.copy_(m.num_batches_tracked)

                # load the new bn
                replace_module(new_model, name, new_bn)

        self.model = new_model

        kept_wg_forward = []
        kept_wg_pre_forward = []
        kept_wg_backward = []
        kept_wg_pre_backward = []
        for i in range(30):
            if self.opt["dist"]:
                kept_wg_forward.append(list(range(64)))
                kept_wg_pre_forward.append(sorted(self.kept_wg_pre["module.forward_trunk.main.2.%d.conv1" % (i)]))
                kept_wg_backward.append(list(range(64)))
                kept_wg_pre_backward.append(sorted(self.kept_wg_pre["module.backward_trunk.main.2.%d.conv1" % (i)]))
            else:
                kept_wg_forward.append(list(range(64)))
                kept_wg_pre_forward.append(sorted(self.kept_wg_pre["forward_trunk.main.2.%d.conv1" % (i)]))
                kept_wg_backward.append(list(range(64)))
                kept_wg_pre_backward.append(sorted(self.kept_wg_pre["backward_trunk.main.2.%d.conv1" % (i)]))


        if self.opt["dist"]:
            self.model.module.num_feat_backward = self.num_feat_backward
            self.model.module.num_feat_forward = self.num_feat_forward
            self.model.module.pruned = True
            self.model.module.kept_wg_forward = kept_wg_forward
            self.model.module.kept_wg_pre_forward = kept_wg_pre_forward
            self.model.module.kept_wg_backward = kept_wg_backward
            self.model.module.kept_wg_pre_backward = kept_wg_pre_backward
        else:
            self.model.num_feat_backward = self.num_feat_backward
            self.model.num_feat_forward = self.num_feat_forward
            self.model.pruned = True
            self.model.kept_wg_forward = kept_wg_forward
            self.model.kept_wg_pre_forward = kept_wg_pre_forward
            self.model.kept_wg_backward = kept_wg_backward
            self.model.kept_wg_pre_backward = kept_wg_pre_backward
        # print the layer shape of pruned model
        LayerStruct(new_model, self.LEARNABLES, self.logger)
        return new_model

    def _prune_and_build_new_modelV3(self):
        if self.args.wg == 'weight':
            self.masks = get_masks(self.layers, self.pruned_wg)
            return
        if self.opt["dist"]:
            name_st = ["module.forward_trunk.main.0", "module.backward_trunk.main.0"]
            name_ed = ["module.forward_trunk.main.2.29.conv2", "module.backward_trunk.main.2.29.conv2"]
            name_forward = "module.forward_trunk.main.2.29.conv2"
            name_backward = "module.backward_trunk.main.2.29.conv2"
        else:
            name_st = ["forward_trunk.main.0", "backward_trunk.main.0"]
            name_forward = "forward_trunk.main.2.29.conv2"
            name_backward = "backward_trunk.main.2.29.conv2"

        for name, m in self.model.named_modules():
            if isinstance(m, self.LEARNABLES) and "trunk.main" in name and "conv2" in name:
                kept_filter, kept_chl = get_kept_filter_channel(self.layers, name, m, pr=self.pr, kept_wg=self.kept_wg,
                                                                kept_wg_pre=self.kept_wg_pre, opt=self.opt,
                                                                wg=self.args.wg, name_st=name_st)

                # decide if renit the current layer
                reinit = False
                for rl in self.args.reinit_layers:
                    if fnmatch(name, rl):
                        reinit = True
                        break

                # get number of channels (can be manually assigned)
                num_chl = self.args.layer_chl[name] if name in self.args.layer_chl else len(kept_chl)
                # copy weight and bias
                bias = False if isinstance(m.bias, type(None)) else True
                if isinstance(m, nn.Conv2d):
                    new_layer = nn.Conv2d(num_chl, len(kept_filter), m.kernel_size,
                                              m.stride, m.padding, m.dilation, m.groups, bias).cuda()
                    if not reinit:
                        kept_weights = m.weight.data[kept_filter]
                    if name_forward in name:
                        self.num_feat_forward = len(kept_filter)
                    if name_backward in name:
                        self.num_feat_backward = len(kept_filter)

                if not reinit:
                    new_layer.weight.data.copy_(kept_weights)  # load weights into the new module
                    if bias:
                        kept_bias = m.bias.data[kept_filter]
                        new_layer.bias.data.copy_(kept_bias)

                # load the new conv
                replace_module(self.model, name, new_layer)

        kept_wg_forward = []
        kept_wg_pre_forward = []
        kept_wg_backward = []
        kept_wg_pre_backward = []
        for i in range(30):
            if self.opt["dist"]:
                kept_wg_forward.append(sorted(self.kept_wg["module.forward_trunk.main.2.%d.conv2" % (i)]))
                kept_wg_pre_forward.append(sorted(self.kept_wg_pre["module.forward_trunk.main.2.%d.conv1" % (i)]))
                kept_wg_backward.append(sorted(self.kept_wg["module.backward_trunk.main.2.%d.conv2" % (i)]))
                kept_wg_pre_backward.append(sorted(self.kept_wg_pre["module.backward_trunk.main.2.%d.conv1" % (i)]))
            else:
                kept_wg_forward.append(sorted(self.kept_wg["forward_trunk.main.2.%d.conv2" % (i)]))
                kept_wg_pre_forward.append(sorted(self.kept_wg_pre["forward_trunk.main.2.%d.conv1" % (i)]))
                kept_wg_backward.append(sorted(self.kept_wg["backward_trunk.main.2.%d.conv2" % (i)]))
                kept_wg_pre_backward.append(sorted(self.kept_wg_pre["backward_trunk.main.2.%d.conv1" % (i)]))

        if self.opt["dist"]:
            self.model.module.num_feat_backward = self.num_feat_backward
            self.model.module.num_feat_forward = self.num_feat_forward
            self.model.module.pruned = True
            self.model.module.pruned_final = True
            self.model.module.kept_wg_forward = kept_wg_forward
            self.model.module.kept_wg_pre_forward = kept_wg_pre_forward
            self.model.module.kept_wg_backward = kept_wg_backward
            self.model.module.kept_wg_pre_backward = kept_wg_pre_backward
        else:
            self.model.num_feat_backward = self.num_feat_backward
            self.model.num_feat_forward = self.num_feat_forward
            self.model.pruned = True
            self.model.pruned_final = True
            self.model.kept_wg_forward = kept_wg_forward
            self.model.kept_wg_pre_forward = kept_wg_pre_forward
            self.model.kept_wg_backward = kept_wg_backward
            self.model.kept_wg_pre_backward = kept_wg_pre_backward


        # print the layer shape of pruned model
        LayerStruct(self.model, self.LEARNABLES, self.logger)
        return self.model