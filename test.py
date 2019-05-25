import time
import os, sys
import pickle
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from options.test_options import TestOptions
from models.models import Model
from datasets.vrd import VrdDataset
import models


def test_net(model, opt, test_data_loader, dataset_name):

    pred_result = [[] for _ in range(len(test_data_loader))]
    for i_batch, data_dict in enumerate(test_data_loader):


        #if isinstance(data_dict['sub_loc'][0], int):
        if len(data_dict['sub_loc'][0].size()) == 0:
            # No detection
            continue


        sys.stdout.write(('Processing {:d} image  \r'.format(i_batch)))
        sys.stdout.flush()
        model.set_input(data_dict)
        rel_pred = model.get_test_result()
        # Size: len(sub_loc) * 70
        pred_result[i_batch] = rel_pred

    # Write to file

    result_dir = os.path.join(opt.results_dir, opt.name, dataset_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(os.path.join(result_dir, 'predicted_predicate.pkl'), 'wb') as f:
        pickle.dump(pred_result, f, pickle.HIGHEST_PROTOCOL)
    print('Finished {:d} testing images.'.format(len(test_data_loader)))


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1 (1 image at a time)
    opt.serial_batches = True
    opt.gpu_ids = [opt.gpu_ids[0]]
    if 'vrd' in opt.dataroot:
        test_dataset = VrdDataset(opt.dataroot, split='test', net=opt.feat_net, use_gt=opt.use_gt, use_lang=opt.use_lang)
    else:
        print('No this dataset')
        sys.exit(1)
    test_data_loader = DataLoader(test_dataset, batch_size=opt.batchSize,
                                  shuffle=False, num_workers=int(opt.nThreads))
    model = Model(opt)
    test_net(model, opt, test_data_loader, test_dataset.name)

    if not opt.no_evaluate:
        with open(os.path.join(opt.results_dir, opt.name, test_dataset.name, 'predicted_predicate.pkl'), 'rb') as f:
            rel_result = pickle.load(f)

        test_dataset.evaluate(rel_result, n_rels=opt.n_rels, obj_co_occur=model.relevance)
        #test_dataset.evaluate(rel_result, n_rels=opt.n_rels)
    
