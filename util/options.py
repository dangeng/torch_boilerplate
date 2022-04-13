import os
from pathlib import Path
import configargparse
import re

import pdb

def bool_converter(arg):
    trues = ['true', 'True', 't', 'T']
    falses = ['false', 'False', 'f', 'F']

    assert (arg in trues) or (arg in falses), 'Unrecognized bool argument!'
    if arg in trues:
        return True
    elif arg in falses:
        return False

def parse_args():
    parser = configargparse.ArgParser()
    parser.add('-c', '--config', required=True, is_config_file=True, help='config file path')

    # General
    parser.add('--device', type=int, nargs='+', required=True, help='device(s) to use')
    parser.add('--num_workers', type=int, default=32, help='device(s) to use')
    parser.add('--expr_name', type=str, required=True, help='experiment name')

    # Training
    parser.add('--resume', type=bool_converter, default=False, help='if true, resume training')
    parser.add('--epochs', type=int, default=30, help='number of epochs to train for')
    parser.add('--lr', type=float, default=1e-4, help='learning rate')
    parser.add('--save', type=bool_converter, default=True, help='if true, save and log')
    parser.add('--save_freq', type=int, default=5, help='frequency, in epochs, to save')

    # Data
    parser.add('--datadir', type=str, required=True, help='path to data')
    parser.add('--batch_size', type=int, default=64, help='training batch size')
    parser.add('--shuffle', action='store_true', help='shuffle the dataset (easier batches)')

    # Test
    parser.add('--test_epoch', type=int, default=-1, help='epoch to test (-1 means most recent named chkpt)')

    opt = parser.parse_args()

    ########################
    # Args Post-Processing #
    ########################

    # Make sure experiment name and config file path are the same
    # TODO: Why don't we just make this automatic? If we want it so much
    assert Path(opt.config).stem == opt.expr_name, "Experiment name doesn't match config file name!"

    # If test_epoch == None, then automatically set to latest
    if opt.test_epoch == -1:
        chkpts_dir = Path('results/chkpts') / opt.expr_name
        chkpts = os.listdir(chkpts_dir)
        p = re.compile('^[0-9]{4}\.pth$')
        chkpts = sorted([x for x in chkpts if p.match(x)])
        if len(chkpts) == 0:
            opt.test_epoch = 0
        else:
            opt.test_epoch = int(chkpts[-1].split('.')[0])

    return opt

