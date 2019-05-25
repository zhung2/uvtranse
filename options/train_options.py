# Modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--print_epoch_iter_freq', type=int, default=10240, help='frequency of showing training results on console (how many iteration)')
        self.parser.add_argument('--val_epoch_freq', type=int, default=1, help='frequency of doing validation on vrd_test (how many epoch)')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs (how many epoch)')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training. ** Note ** need to specify which_epoch')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, test, etc')
        self.parser.add_argument('--which_epoch', type=int, default=0, help='which epoch to load?')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter to train')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        self.parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=100, help='lr be multiplied by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay, i.e. l2 regularization\'s coeff')

        self.isTrain = True
