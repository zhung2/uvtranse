from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--demos_dir', type=str, default='./demos/', help='saves demos here.')
        self.parser.add_argument('--phase', type=str, default='test', help='train, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default=0, help='which epoch to load?')
        self.parser.add_argument('--n_rels', type=int, default=5, help='Number of top predicates for each possible relations when computing recall@50, and top recall@100')
        self.parser.add_argument('--no_evaluate', action='store_true', help='Do not evaluating top recall@50, and top recall@100')
        self.parser.add_argument('--n', type=int, default=0, help='Index of demo image')
        self.parser.add_argument('--use_q', action='store_true', help='Whether to use q distribution')

        self.isTrain = False
