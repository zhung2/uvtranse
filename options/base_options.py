# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

import argparse
import os
import torch

class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, default='/data/zhung2/datasets/vrd/', help='path to dataset (including vrd and unrel)')
        self.parser.add_argument('--batchSize', type=int, default=1024, help='training batch size')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--model', type=str, default='union',
                                 help='chooses which model to use. vtranse, union, combine')
        self.parser.add_argument('--feat_net', type=str, default='vgg16', help='which net used to extract feat (vgg16 or res101)')
        self.parser.add_argument('--fc7_dim', type=int, default=4096, help='dimension of fc7 (4096 for vgg16, and 2048 for res101)')
        self.parser.add_argument('--keep_prob', type=float, default=0.5, help='Dropout probability')
        self.parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--name', type=str, default='union', help='name of the experiment. It decides where to store models')
        self.parser.add_argument('--loss', type=str, default='ce', help='Loss to use (ce: CrossEntropyLoss / bce: BinaryCrossEntropyLoss)')
        self.parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.initialized = True
        self.parser.add_argument('--use_gt', action='store_true', help='Use gt boxes and objs for evaluation predicate detection')
        self.parser.add_argument('--use_prior', action='store_true', help='Use prior statistics')
        self.parser.add_argument('--use_co_occur', action='store_true', help='Use co_occur in test time for reordering objects')
        self.parser.add_argument('--use_word_sim', action='store_true', help='Use word embedding similarity in training')
        self.parser.add_argument('--balance', type=float, default=0.0, help='Balancing term for kl')
        self.parser.add_argument('--use_lang', action='store_true', help='Use language feature')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt


