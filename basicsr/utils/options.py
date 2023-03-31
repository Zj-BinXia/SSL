import argparse
import random
import torch
import yaml
from collections import OrderedDict
from os import path as osp
from putils import strlist_to_list, strdict_to_dict, check_path, parse_prune_ratio_vgg

from basicsr.utils import set_random_seed
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def _postprocess_yml_value(value):
    # None
    if value == '~' or value.lower() == 'none':
        return None
    # bool
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    # !!float number
    if value.startswith('!!float'):
        return float(value.replace('!!float', ''))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
        return float(value)
    # list
    if value.startswith('['):
        return eval(value)
    # str
    return value


def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')

    parser.add_argument('--method', type=str, default='SSL', choices=['SSL'],
                        help='method name')
    parser.add_argument('--wg', type=str, default='filter', choices=['filter', 'weight'],
                        help='weight group to prune')
    parser.add_argument('--stage_pr', type=float, default=0.5,
                        help='to appoint layer-wise pruning ratio')
    # parser.add_argument('--skip_layers', type=str, default="*spynet*,*conv_last,*fusion,*upconv1,*upconv2,*conv_hr,*forward_trunk.main.0,*backward_trunk.main.0",
    #                      help='layers to skip when pruning')
    parser.add_argument('--skip_layers', type=str, default="*spynet*,*conv_last,*forward_trunk.main.0,*backward_trunk.main.0",
                        help='layers to skip when pruning')
    parser.add_argument('--reinit_layers', type=str, default="",
                        help='layers to reinit (not inherit weights)')
    parser.add_argument('--same_pruned_wg_layers', type=str, default='',
                        help='layers to be set with the same pruned weight group in forward')
    parser.add_argument('--same_pruned_wg_layers_backward', type=str, default='',
                        help='layers to be set with the same pruned weight group in backward')

    parser.add_argument('--same_pruned_wg_criterion', type=str, default='reg', choices=['rand', 'reg'],
                        help='use which criterion to select pruned wg')
    parser.add_argument('--resume_path', type=str, default='',
                        help='path of the checkpoint to resume')

    # GReg method related (default setting is for ImageNet):
    parser.add_argument('--update_reg_interval', type=int, default=5)
    parser.add_argument('--stabilize_reg_interval', type=int, default=14000)
    parser.add_argument('--reg_upper_limit', type=float, default=0.1)
    parser.add_argument('--reg_granularity_prune', type=float, default=1e-4)
    parser.add_argument('--pick_pruned', type=str, default='min', choices=['min', 'max', 'rand'])
    parser.add_argument('--not_apply_reg', dest='apply_reg', action='store_false', default=True)
    parser.add_argument('--layer_chl', type=str, default='',
                        help='manually assign the number of channels for some layers. A not so beautiful scheme.')
    parser.add_argument('--greg_mode', type=str, default='part', choices=['part', 'all'])
    parser.add_argument('--compare_mode', type=str, default='global', choices=['local', 'global'])
    parser.add_argument('--prune_criterion', type=str, default='act_scale', choices=['l1-norm', 'act_scale'])
    parser.add_argument('--reinit', action='store_true',help='reinit weights of new model?')
    # WN+Reg
    # parser.add_argument('--wn', action='store_true', help='if use weight normalization')
    parser.add_argument('--lw_spr', type=float, default=1, help='lw for loss of sparsity pattern regularization')
    parser.add_argument('--iter_finish_spr', '--iter_ssa', dest='iter_ssa', type=int, default=3375,
                        help='batch size =8 270*100/8=3725 6750 is 2 epochs')
    parser.add_argument('--fix_flow_iter', type=int, default=3375,
                        help='flow net iterations')
    parser.add_argument('--print_interval', type=int, default=500)
    # parser.add_argument('--lr_prune', type=float, default=0.0002)


    args = parser.parse_args()

    if args.method in ['SSL']:
        assert args.stage_pr
        if args.compare_mode in ['global']:  # 'stage_pr' is a float
            args.stage_pr = float(args.stage_pr)
        elif args.compare_mode in ['local']:  # 'stage_pr' is a list
            args.stage_pr = parse_prune_ratio_vgg(args.stage_pr, num_layers=2000)
        args.skip_layers = strlist_to_list(args.skip_layers, str)
        args.reinit_layers = strlist_to_list(args.reinit_layers, str)
        args.same_pruned_wg_layers = strlist_to_list(args.same_pruned_wg_layers, str)
        args.layer_chl = strdict_to_dict(args.layer_chl, int)
        args.same_pruned_wg_layers_backward = strlist_to_list(args.same_pruned_wg_layers_backward, str)
    # parse yml to dict
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    # force to update yml options
    if args.force_yml is not None:
        for entry in args.force_yml:
            # now do not support creating new keys
            keys, value = entry.split('=')
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = 'opt'
            for key in keys.split(':'):
                eval_str += f'["{key}"]'
            eval_str += '=value'
            # using exec function
            exec(eval_str)

    opt['auto_resume'] = args.auto_resume
    opt['is_train'] = is_train

    # debug setting
    if args.debug and not opt['name'].startswith('debug'):
        opt['name'] = 'debug_' + opt['name']

    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()

    # datasets
    for phase, dataset in opt['datasets'].items():
        # for multiple datasets, e.g., val_1, val_2; test_1, test_2
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    if is_train:
        experiments_root = osp.join(root_path, 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')

        # change some options for debug mode
        if 'debug' in opt['name']:
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = osp.join(root_path, 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

    return opt, args


@master_only
def copy_opt_file(opt_file, experiments_root):
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile
    cmd = ' '.join(sys.argv)
    filename = osp.join(experiments_root, osp.basename(opt_file))
    copyfile(opt_file, filename)

    with open(filename, 'r+') as f:
        lines = f.readlines()
        lines.insert(0, f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n')
        f.seek(0)
        f.writelines(lines)
