import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from decimal import Decimal
import os, copy, time, pickle, numpy as np, math
from .meta_pruner import MetaPruner
import utility
import matplotlib.pyplot as plt
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from tqdm import tqdm
from fnmatch import fnmatch, fnmatchcase
from .utils import get_score_layer, pick_pruned_layer
from collections import OrderedDict, Counter
from basicsr.losses import build_loss
from basicsr.models.video_base_model_pruned import VideoBaseModel_pruned
from basicsr.utils.dist_util import get_dist_info
from os import path as osp
from torch import distributed as dist
pjoin = os.path.join
tensor2list = lambda x: x.data.cpu().numpy().tolist()
tensor2array = lambda x: x.data.cpu().numpy()
totensor = lambda x: torch.Tensor(x)

class Pruner(MetaPruner, VideoBaseModel_pruned):
    def __init__(self, model, args,opt, logger,train_sampler=None,prefetcher=None,val_loaders=None):
        super(Pruner, self).__init__(model, args, opt, logger)
        VideoBaseModel_pruned.__init__(self,opt)
        # ************************** variables from RCAN **************************

        self.error_last = 1e8
        self.optimizers = []
        self.fix_flow_iter = self.args.fix_flow_iter
        self.setup_optimizers()
        self.cri_pix = build_loss(self.opt["train"]['pixel_opt'])
        self.train_sampler = train_sampler
        self.prefetcher = prefetcher
        self.val_loaders = val_loaders
        # Reg related variables
        self.reg = {}
        self.reg_pre = {}

        self.delta_reg = {}
        self._init_reg()
        self.iter_update_reg_finished = {}
        self.iter_finish_pick = {}
        self.iter_stabilize_reg = math.inf
        self.hist_mag_ratio = {}
        self.w_abs = {}
        self.act_scale = {}
        self.act_scale_pre = {}

        # init prune_state
        self.prune_state = 'update_reg'
        
        # init pruned_wg/kept_wg if they can be determined right at the begining
        if args.greg_mode in ['part'] and self.prune_state in ['update_reg']:
            self._get_kept_wg_L1(align_constrained=False) # this will update the 'self.kept_wg', 'self.pruned_wg', 'self.pr'

    def _init_reg(self):
        for name, m in self.model.named_modules():
            if name in self.layers:
                if self.args.wg == 'weight':
                    self.reg[name] = torch.zeros_like(m.weight.data).flatten().cuda()
                else:
                    shape = m.weight.data.shape
                    if "upconv" in name:
                        self.reg[name] = torch.zeros(shape[0]//4, shape[1]).cuda()
                    else:
                        self.reg[name] = torch.zeros(shape[0], shape[1]).cuda()
                    if hasattr(m, 'act_scale_pre'):
                        self.reg_pre[name] = torch.zeros( shape[1],shape[0]).cuda()

    def _greg_1(self, m, name):
        if self.pr[name] == 0:
            return True
        
        pruned = self.pruned_wg[name]
        if self.args.wg == "channel":
            self.reg[name][:, pruned] += self.args.reg_granularity_prune
        elif self.args.wg == "filter":
            if not self.reg[name].max() > self.args.reg_upper_limit:
                self.reg[name][pruned, :] += self.args.reg_granularity_prune
            if hasattr(m, 'act_scale_pre'):
                pruned_pre = self.pruned_wg_pre[name]
                if not self.reg_pre[name].max() > self.args.reg_upper_limit:
                    self.reg_pre[name][pruned_pre, :] += self.args.reg_granularity_prune
        elif self.args.wg == 'weight':
            self.reg[name][pruned] += self.args.reg_granularity_prune
        else:
            raise NotImplementedError

        # when all layers are pushed hard enough, stop
        if hasattr(m, 'act_scale_pre'):
            return self.reg[name].max() > self.args.reg_upper_limit and self.reg_pre[name].max() > self.args.reg_upper_limit
        else:
            return self.reg[name].max() > self.args.reg_upper_limit

    def _greg_penalize_all(self, m, name):
        if self.pr[name] == 0:
            return True
        
        if self.args.wg == "channel":
            self.reg[name] += self.args.reg_granularity_prune
        elif self.args.wg == "filter":
            if not self.reg[name].max() > self.args.reg_upper_limit:
                self.reg[name] += self.args.reg_granularity_prune
            if hasattr(m, 'act_scale_pre'):
                if not self.reg_pre[name].max() > self.args.reg_upper_limit:
                    self.reg_pre[name] += self.args.reg_granularity_prune
        elif self.args.wg == 'weight':
            self.reg[name] += self.args.reg_granularity_prune
        else:
            raise NotImplementedError

        # when all layers are pushed hard enough, stop
        if hasattr(m, 'act_scale_pre'):
            return self.reg[name].max() > self.args.reg_upper_limit and self.reg_pre[name].max() > self.args.reg_upper_limit
        else:
            return self.reg[name].max() > self.args.reg_upper_limit

    def _update_reg(self, skip=[]):
        for name, m in self.model.named_modules():
            if name in self.layers:                
                if name in self.iter_update_reg_finished.keys():
                    continue
                if name in skip:
                    continue

                # get the importance score (L1-norm in this case)
                out = get_score_layer(name,m, wg='filter', criterion='act_scale')
                self.w_abs[name], self.act_scale[name], self.act_scale_pre[name] = out['l1-norm'], out['act_scale'], out['act_scale_pre']
                
                # update reg functions, two things:
                # (1) update reg of this layer (2) determine if it is time to stop update reg
                if self.args.greg_mode in ['part']:
                    finish_update_reg = self._greg_1(m, name)
                elif self.args.greg_mode in ['all']:
                    finish_update_reg = self._greg_penalize_all(m, name)

                # check prune state
                if finish_update_reg:
                    # after 'update_reg' stage, keep the reg to stabilize weight magnitude
                    self.iter_update_reg_finished[name] = self.total_iter
                    self.logger.info(f"==> {self.layer_print_prefix[name]} -- Just finished 'update_reg'. Iter {self.total_iter}. pr {self.pr[name]}")

                    # check if all layers finish 'update_reg'
                    prune_state = "stabilize_reg"
                    for n, mm in self.model.named_modules():
                        if isinstance(mm, self.LEARNABLES):
                            if n not in self.iter_update_reg_finished:
                                prune_state = ''
                                break
                    if prune_state == "stabilize_reg":
                        self.prune_state = 'stabilize_reg'
                        self.iter_stabilize_reg = self.total_iter
                        self.logger.info("==> All layers just finished 'update_reg', go to 'stabilize_reg'. Iter = %d" % self.total_iter)

    def _apply_reg(self):
        for name, m in self.model.named_modules():
            if name in self.layers and self.pr[name] > 0:
                reg = self.reg[name] # [N, C]
                m.act_scale.grad += reg[:, 0].view(1,-1,1,1) * m.act_scale

                if hasattr(m, 'act_scale_pre'):
                    reg_pre = self.reg_pre[name]
                    m.act_scale_pre.grad += reg_pre[:, 0].view(1, -1, 1, 1) * m.act_scale
                # bias = False if isinstance(m.bias, type(None)) else True
                # if bias:
                #     m.bias.grad += reg[:, 0] * m.bias

    def _merge_wn_scale_to_weightsV3(self):
        '''Merge the learned weight normalization scale to the weights.
        '''
        for name, m in self.model.named_modules():
            if name in self.layers and hasattr(m, 'act_scale') and "trunk.main" in name and "conv2" in name:
                m.weight.data = m.weight.data * m.act_scale.data.view(-1,1,1,1)
                self.logger.info(f'Merged weight normalization scale to weights: {name}')
                bias = False if isinstance(m.bias, type(None)) else True
                if bias:
                    m.bias.data = m.bias.data * m.act_scale.data.view(-1)

                    self.logger.info(f'Merged activation scale to bias: {name}')

    def _merge_wn_scale_to_weightsV2(self):
        '''Merge the learned weight normalization scale to the weights.
        '''
        for name, m in self.model.named_modules():
            if name in self.layers and hasattr(m, 'act_scale'):
                if "upconv" in name:
                    cout, cin ,kh, kw = m.weight.data.shape
                    m.weight.data = m.weight.data.view(cout//4,-1,kh,kw) * m.act_scale.data.view(-1, 1, 1, 1)
                    m.weight.data = m.weight.data.view(cout, cin ,kh, kw)
                elif "trunk.main" in name and "conv2" in name:
                    m.act_scale.data[:,self.pruned_wg[name],:,:]=0
                    m.act_scale.requires_grad = False
                else:
                    m.weight.data = m.weight.data * m.act_scale.data.view(-1,1,1,1)
                    if hasattr(m, 'act_scale_pre'):
                        m.weight.data = m.weight.data * m.act_scale_pre.data.view(1,-1,1,1)
                self.logger.info(f'Merged weight normalization scale to weights: {name}')
                bias = False if isinstance(m.bias, type(None)) else True
                if bias:
                    if "upconv" in name:
                        cout = m.bias.data.size()[0]
                        m.bias.data = (m.bias.data.view(cout//4,-1) * m.act_scale.data.view(-1,1)).view(-1)
                    elif "trunk.main" in name and "conv2" in name:
                        pass
                    else:
                        m.bias.data = m.bias.data*m.act_scale.data.view(-1)

                    self.logger.info(f'Merged activation scale to bias: {name}')

    def _merge_wn_scale_to_weights(self):
        '''Merge the learned weight normalization scale to the weights.
        '''
        for name, m in self.model.named_modules():
            if name in self.layers and hasattr(m, 'act_scale'):
                if "upconv" in name:
                    cout, cin ,kh, kw = m.weight.data.shape
                    m.weight.data = m.weight.data.view(cout//4,-1,kh,kw) * m.act_scale.data.view(-1, 1, 1, 1)
                    m.weight.data = m.weight.data.view(cout, cin ,kh, kw)
                else:
                    m.weight.data = m.weight.data * m.act_scale.data.view(-1,1,1,1)
                    if hasattr(m, 'act_scale_pre'):
                        m.weight.data = m.weight.data * m.act_scale_pre.data.view(1,-1,1,1)
                self.logger.info(f'Merged weight normalization scale to weights: {name}')
                bias = False if isinstance(m.bias, type(None)) else True
                if bias:
                    if "upconv" in name:
                        cout = m.bias.data.size()[0]
                        print(m.bias.data.view(cout//4,-1).shape,m.act_scale.data.view(-1,1).shape)
                        m.bias.data = (m.bias.data.view(cout//4,-1) * m.act_scale.data.view(-1,1)).view(-1)
                    else:
                        m.bias.data = m.bias.data*m.act_scale.data.view(-1)

                    self.logger.info(f'Merged activation scale to bias: {name}')

    def _resume_prune_status(self, ckpt_path):
        raise NotImplementedError

    def cal_pr(self):
        for name, layer in self.layers.items():
            num_pruned = len(self.pruned_wg[name])
            self.pr[name] = num_pruned / len(layer.score)

    def _save_model(self, filename):
        savepath = os.path.join(self.opt['path']['models'],filename)
        ckpt = {
            'pruned_wg': self.pruned_wg,
            'kept_wg': self.kept_wg,
            'pruned_wg_pre': self.pruned_wg_pre,
            'kept_wg_pre': self.kept_wg_pre,
            'model': self.model,
            'state_dict': self.model.state_dict(),
        }
        torch.save(ckpt, savepath) 
        return savepath

    def prune(self,resume_state=None):
        self.total_iter = 0
        if self.args.resume_path:
            self._resume_prune_status(self.args.resume_path)
            self._get_kept_wg_L1() # get pruned and kept wg from the resumed model
            self.model = self.model.train()
            self.logger.info("Resume model successfully: '{}'. Iter = {}. prune_state = {}".format(
                        self.args.resume_path, self.total_iter, self.prune_state))

        if resume_state:
            self._merge_wn_scale_to_weightsV2()
            self._prune_and_build_new_modelV2()
            self.logger.info(f"==> Resume Final Pruned and built a new model. ")
            return copy.deepcopy(self.model)

        while True:
            finish_prune = self.train() # there will be a break condition to get out of the infinite loop
            # self._merge_wn_scale_to_weightsV2()
            # self._prune_and_build_new_modelV2()
            if finish_prune:
                path = self._save_model('model_prune_init.pt')
                self.logger.info(f"==> Final Pruned and built a new model. Ckpt saved: '{path}'. Testing...")
                return copy.deepcopy(self.model)

    def prune_final(self):
            #finish_prune = self.train() # there will be a break condition to get out of the infinite loop
            self._merge_wn_scale_to_weightsV3()
            self._prune_and_build_new_modelV3()

            path = self._save_model('model_prune_final.pt')
            self.logger.info(f"==> Final Pruned and built a new model. Ckpt saved: '{path}'. Testing...")
            return copy.deepcopy(self.model)

    def train(self):
        self.model.train()
        self.train_sampler.set_epoch(100000)
        self.prefetcher.reset()
        train_data = self.prefetcher.next()
        while train_data is not None:
            self.total_iter += 1

            self.lq = train_data['lq'].to(self.device)
            self.gt = train_data['gt'].to(self.device)

            finished = self.optimize_parameters(self.total_iter)
            if finished:
                return True

    def _print_reg_status(self):
        self.logger.info('************* Regularization Status *************')
        for name, m in self.model.named_modules():
            if name in self.layers and self.pr[name] > 0:
                logstr = [self.layer_print_prefix[name]]
                logstr += [f"reg_status: min {self.reg[name].min():.5f} ave {self.reg[name].mean():.5f} max {self.reg[name].max():.5f}"]
                if hasattr(m, 'act_scale_pre'):
                    logstr += [
                        f"reg_pre_status: min {self.reg_pre[name].min():.5f} ave {self.reg_pre[name].mean():.5f} max {self.reg_pre[name].max():.5f}"]
                out = get_score_layer(name,m, wg='filter', criterion='act_scale')
                w_abs, act_scale, act_scale_pre = out['l1-norm'], out['act_scale'], out['act_scale_pre']
                pruned, kept = pick_pruned_layer(score=act_scale, pr=self.pr[name], sort_mode='min')
                avg_mag_pruned, avg_mag_kept = np.mean(w_abs[pruned]), np.mean(w_abs[kept])
                avg_scale_pruned, avg_scale_kept = np.mean(act_scale[pruned]), np.mean(act_scale[kept])
                logstr += ["average w_mag: pruned %.6f kept %.6f" % (avg_mag_pruned, avg_mag_kept)]
                logstr += ["average act_scale: pruned %.6f kept %.6f" % (avg_scale_pruned, avg_scale_kept)]

                if hasattr(m, 'act_scale_pre'):
                    pruned, kept = pick_pruned_layer(score=act_scale_pre, pr=self.pr[name], sort_mode='min')
                    avg_scale_pruned, avg_scale_kept = np.mean(act_scale_pre[pruned]), np.mean(act_scale_pre[kept])
                    logstr += ["average act_scale_pre: pruned %.6f kept %.6f" % (avg_scale_pruned, avg_scale_kept)]
                logstr += [f'Iter {self.total_iter}']
                logstr += [f'cstn' if name in self.constrained_layers else 'free']
                logstr += [f'pr {self.pr[name]}']
                self.logger.info(' | '.join(logstr))
        self.logger.info('*************************************************')
        
    def val(self):
        is_train = self.model.training
        torch.set_grad_enabled(False)

        self.logger.info('Evaluation:')
        self.model.eval()

        start_time = time.time()
        if self.opt.get('val') is not None:
            for val_loader in self.val_loaders:
                self.validation(val_loader, self.total_iter, self.logger, self.opt['val']['save_img'])
        consumed_time = str(int(time.time() - start_time))
        self.logger.info(f'After finetuned Time consumed: {consumed_time},[prune_state: {self.prune_state} compare_mode: {self.args.compare_mode} greg_mode: {self.args.greg_mode}]')

        torch.set_grad_enabled(True)

        if is_train:
            self.model.train()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        self.logger.info(f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.model.parameters()
        else:  # separate flow params and normal params for different lr
            normal_params = []
            flow_params = []
            for name, param in self.model.named_parameters():
                if 'spynet' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if self.fix_flow_iter:
            if current_iter == 1:
                self.logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.model.named_parameters():
                    if 'spynet' in name or 'edvr' in name:
                        param.requires_grad_(False)
            # elif current_iter == self.fix_flow_iter:
            #     self.logger.warning('Train all the parameters.')
            #     self.model.requires_grad_(True)

        self.optimizer_g.zero_grad()
        self.output = self.model(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        l_pix = self.cri_pix(self.output, self.gt)
        l_total += l_pix
        loss_dict['l_pix'] = l_pix

        if self.total_iter % self.args.print_interval == 0:
            self.logger.info("")
            self.logger.info(
                f"Iter {self.total_iter} [prune_state: {self.prune_state} method: {self.args.method} compare_mode: {self.args.compare_mode} greg_mode: {self.args.greg_mode}] " + "-" * 40)

        l_total.backward()

        # @mst: update reg factors and apply them before optimizer updates
        if self.prune_state in ['update_reg'] and self.total_iter % self.args.update_reg_interval == 0:
            self._update_reg()

        # after reg is updated, print to check
        if self.total_iter % self.args.print_interval == 0:
            self._print_reg_status()

        if self.args.apply_reg:  # reg can also be not applied, as a baseline for comparison
            self._apply_reg()
        # --


        self.optimizer_g.step()

        if  self.total_iter % self.args.print_interval == 0:
            logstr = f'Iter {self.total_iter} loss_recon {l_pix.item():.4f}'
            self.logger.info(logstr)

        # @mst: exit of reg pruning loop
        if self.prune_state in [
            "stabilize_reg"] and self.total_iter - self.iter_stabilize_reg == self.args.stabilize_reg_interval:
            self.logger.info(
                f"==> 'stabilize_reg' is done. Iter {self.total_iter}.About to prune and build new model. Testing...")
            self.val()

            # if self.args.greg_mode in ['all']:
            self._get_kept_wg_L1()
            self.logger.info(f'==> Get pruned_wg/kept_wg.')

            self._merge_wn_scale_to_weightsV2()
            self._prune_and_build_new_modelV2()
            path = self._save_model('model_prune_optimization.pt')
            self.logger.info(f"==> Pruned and built a new model. Ckpt saved: '{path}'. Testing...")
            self.val()
            return True


    def dist_validation(self, dataloader, current_iter,logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None
        # initialize self.metric_results
        # It is a dict: {
        #    'folder1': tensor (num_frame x len(metrics)),
        #    'folder2': tensor (num_frame x len(metrics))
        # }
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {}
                num_frame_each_folder = Counter(dataset.data_info['folder'])
                for folder, num_frame in num_frame_each_folder.items():
                    self.metric_results[folder] = torch.zeros(
                        num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
            # initialize the best metric results
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        metric_data = dict()
        num_folders = len(dataset)
        num_pad = (world_size - (num_folders % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='folder')
        # Will evaluate (num_folders + num_pad) times, but only the first num_folders results will be recorded.
        # (To avoid wait-dead)
        for i in range(rank, num_folders + num_pad, world_size):
            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            folder = val_data['folder']

            # compute outputs
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)
            val_data['gt'].squeeze_(0)

            self.test()
            visuals = self.get_current_visuals()

            # tentative for out of GPU memory
            del self.lq
            del self.output
            if 'gt' in visuals:
                del self.gt
            torch.cuda.empty_cache()

            if self.center_frame_only:
                visuals['result'] = visuals['result'].unsqueeze(1)
                if 'gt' in visuals:
                    visuals['gt'] = visuals['gt'].unsqueeze(1)

            # evaluate
            if i < num_folders:
                for idx in range(visuals['result'].size(1)):
                    result = visuals['result'][0, idx, :, :, :]
                    result_img = tensor2img([result])  # uint8, bgr
                    metric_data['img'] = result_img
                    if 'gt' in visuals:
                        gt = visuals['gt'][0, idx, :, :, :]
                        gt_img = tensor2img([gt])  # uint8, bgr
                        metric_data['img2'] = gt_img
                    if save_img:
                        if self.opt['is_train']:
                            raise NotImplementedError('saving image is not supported during training.')
                        else:
                            if self.center_frame_only:  # vimeo-90k
                                clip_ = val_data['lq_path'].split('/')[-3]
                                seq_ = val_data['lq_path'].split('/')[-2]
                                name_ = f'{clip_}_{seq_}'
                                img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                    f"{name_}_{self.opt['name']}.png")
                            else:  # others
                                img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                    f"{idx:08d}_{self.opt['name']}.png")
                            # image name only for REDS dataset
                        imwrite(result_img, img_path)

                    # calculate metrics
                    if with_metrics:
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            result = calculate_metric(metric_data, opt_)
                            self.metric_results[folder][idx, metric_idx] += result

                # progress bar
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Folder: {folder}')

        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name,logger)

    def test(self):
        n = self.lq.size(1)
        self.model.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            self.lq = torch.cat([self.lq, self.lq.flip(1)], dim=1)

        with torch.no_grad():
            self.output = self.model(self.lq)

        if flip_seq:
            output_1 = self.output[:, :n, :, :, :]
            output_2 = self.output[:, n:, :, :, :].flip(1)
            self.output = 0.5 * (output_1 + output_2)

        if self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]

        self.model.train()



