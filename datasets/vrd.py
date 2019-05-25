from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys
import numpy as np
import scipy.io as sio
import pandas as pd
import PIL.Image as Image
import pickle
import copy

import torch
import torchvision
from torch.utils import data

from tqdm import tqdm
import pdb

#sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
#from utils.evaluation import top_recall_phrase
#from utils.evaluation import top_recall_relationship
#from utils.evaluation import setup
#import utils.evaluation as evaluation
from evaluation.evaluation import top_recall_phrase
from evaluation.evaluation import top_recall_relationship
from evaluation.evaluation import top_recall_predicate
from evaluation.evaluation import setup

def bbox_transform(ex_rois, gt_rois, im_w, im_h):
    """From Faster-RCNN, use in here to encode the location feature"""
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    obj_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    obj_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    obj_dw = np.log(gt_widths) - np.log(ex_widths)
    obj_dh = np.log(gt_heights) - np.log(ex_heights)
    #inter_over_union = get_inter_over_union(ex_rois, gt_rois)

    sub_dx = (ex_ctr_x - gt_ctr_x) / gt_widths
    sub_dy = (ex_ctr_y - gt_ctr_y) / gt_heights
    sub_dw = np.log(ex_widths) - np.log(gt_widths)
    sub_dh = np.log(ex_heights) - np.log(gt_heights)

    union_xmin = np.minimum(ex_rois[:, 0], gt_rois[:, 0])
    union_ymin = np.minimum(ex_rois[:, 1], gt_rois[:, 1])
    union_xmax = np.maximum(ex_rois[:, 2], gt_rois[:, 2])
    union_ymax = np.maximum(ex_rois[:, 3], gt_rois[:, 3])
    union_width = union_xmax - union_xmin + 1.0
    union_height = union_ymax - union_ymin + 1.0
    union_ctr_x = union_xmin + 0.5 * union_width
    union_ctr_y = union_ymin + 0.5 * union_height

    union_area_portion = union_width * union_height / im_w / im_h

    '''
    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh, inter_over_union, union_area_portion)).transpose()
    '''
    targets = np.vstack(
        (sub_dx, sub_dy, sub_dw, sub_dh, obj_dx, obj_dy, obj_dw, obj_dh, union_area_portion)).transpose()
    '''
    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh, inter_over_union,
         union_ctr_x, union_ctr_y, union_width, union_height, area_portion)).transpose()
    '''
    return targets

def new_bbox_transform(ex_rois, gt_rois, im_w, im_h):
    """From Faster-RCNN, use in here to encode the location feature"""
    '''
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (ex_ctr_x - gt_ctr_x) / gt_widths
    targets_dy = (ex_ctr_y - gt_ctr_y) / gt_heights
    targets_dw = np.log(ex_widths / gt_widths)
    targets_dh = np.log(ex_heights / gt_heights)
    target_aspect_ratio = np.log(ex_widths / ex_heights)
    #target_boxes_ratio = np.log((ex_widths*ex_heights) / (gt_widths*gt_heights))

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh, target_aspect_ratio)).transpose()
    '''
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_min_x = ex_rois[:, 0] / float(im_w)
    ex_min_y = ex_rois[:, 1] / float(im_h)
    ex_max_x = ex_rois[:, 2] / float(im_w)
    ex_max_y = ex_rois[:, 3] / float(im_h)
    #ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    #ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
    #ex_area_portion = (ex_widths * ex_heights) / (im_w * im_h)
    ex_area_portion = ex_widths * ex_heights / im_w / im_h

    '''
    targets = np.vstack(
        (ex_ctr_x, ex_ctr_y, ex_widths, ex_heights, ex_area_portion)).transpose()
    '''
    targets = np.vstack(
        (ex_min_x, ex_min_y, ex_max_x, ex_max_y, ex_area_portion)).transpose()
    return targets

def get_inter_over_union(ex_rois, gt_rois):
    ex_rois_w = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_rois_h = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    gt_rois_w = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_rois_h = gt_rois[:, 3] - gt_rois[:, 1] + 1.0

    inter_xmin = np.maximum(ex_rois[:, 0], gt_rois[:, 0])
    inter_ymin = np.maximum(ex_rois[:, 1], gt_rois[:, 1])
    inter_xmax = np.minimum(ex_rois[:, 2], gt_rois[:, 2])
    inter_ymax = np.minimum(ex_rois[:, 3], gt_rois[:, 3])
    inter_w = inter_xmax - inter_xmin + 1.0
    inter_h = inter_ymax - inter_ymin + 1.0

    inter_area = inter_w * inter_h
    inter_area[inter_area < 0] = 0
    union_area = (gt_rois_w * gt_rois_h) + (ex_rois_w * ex_rois_h) - inter_area
    targets_inter_union = inter_area / union_area
    #print(targets_inter_union)
    return targets_inter_union

def union_bbox_transform(union_box, sub_box, obj_box, h, w):
    '''
    union_xmin = union_box[:, 0] / float(w)
    union_xmax = union_box[:, 2] / float(w)
    union_ymin = union_box[:, 1] / float(h)
    union_ymax = union_box[:, 3] / float(h)
    #log_inter_over_union = get_inter_over_union(sub_box, obj_box)

    targets = np.vstack(
        (union_xmin, union_xmax, union_ymin, union_ymax)).transpose()
    '''
    union_w = union_box[:, 2] - union_box[:, 0] + 1.0
    union_h = union_box[:, 3] - union_box[:, 1] + 1.0
    targets = np.vstack(
        (union_box[:, 0], union_box[:, 1], union_w, union_h)).transpose()
    return targets

def load_word_emb(dataroot):
    predicate_emb = torch.load(os.path.join(dataroot, 'pred_emb.pth'))
    object_emb = torch.load(os.path.join(dataroot, 'obj_emb.pth'))
    #print('predicate embedding shape:', predicate_emb.size())
    #print('object embedding shape:', object_emb.size())
    return object_emb.numpy(), predicate_emb.numpy()

def get_obj_wordemb(sub_obj_id, object_emb):
    sub_wordemb = np.zeros((len(sub_obj_id), object_emb.shape[1]))
    obj_wordemb = np.zeros((len(sub_obj_id), object_emb.shape[1]))
    for i in range(len(sub_obj_id)):
        sub_wordemb[i, :] = object_emb[int(sub_obj_id[i, 0])-1]
        obj_wordemb[i, :] = object_emb[int(sub_obj_id[i, 1])-1]

    return sub_wordemb, obj_wordemb

class VrdDataset(data.Dataset):
    def __init__(self, root, split='test', dataset_name='vrd', net="vgg16", use_gt=False, use_lang=False):
        """
        Args:
            root (string): Directory to vrd or unrel.
            split (string): 'train', 'val', 'test'
            dataset_name (string): 'vrd'
        """
        self.root = root
        #data_file = os.path.join(root, split, split+'_dict.pkl')
        data_file = os.path.join(root, split, 'multi_'+ net + '_' + split+'_dict.pkl')
        #data_file = os.path.join(root, split, '1_33_multi_vgg16_best_test_dict.pkl')
        # gt test
        self.use_gt = use_gt
        if self.use_gt:
            assert split == 'test', "Can only use gt test when split == test for evaluation"
            data_file = os.path.join(root, split, 'multi_'+ net + '_' + split+'_gt_dict.pkl')

        with open(data_file, 'rb') as f:
            self.data_dict = pickle.load(f)
        self.split = split
        self.use_lang = use_lang

        self.name = dataset_name
        self.image_folder = os.path.join(self.root, 'images', 'sg_'+self.split+'_images')
        if self.split == 'val':
            self.image_folder = os.path.join(self.root, 'images', 'sg_test_images')
        # ex: self._image_index[0] is the corresponding filename without extension
        self._image_index = self._load_image_set_index()
        #self.ext = '.jpg'

        # object, relationship str2ind
        self._obj_classes = self.get_obj_classes()
        self._num_obj_classes = len(self._obj_classes)
        # 101 classes, including background
        self._obj_class_to_ind = dict(list(zip(self._obj_classes, list(range(self._num_obj_classes)))))

        self._rel_classes = self.get_rel_classes()
        self._num_rel_classes = len(self._rel_classes)
        # 71 relations, including no_relationship
        self._rel_class_to_ind = dict(list(zip(self._rel_classes, list(range(self._num_rel_classes)))))

        self.word_emb = load_word_emb(root)


        if self.split == 'train' or self.split == 'val':
            # Convert data_dict to data_list for batch processing
            self.data_list_cache_file = os.path.join(self.root, self.split, 'multi_'+net+'_data_list_cache.pkl')
            print('Preparing '+ self.split + ' data')
            if not os.path.exists(self.data_list_cache_file):
                self.data_list = self.convert_dict_to_list()
                with open(self.data_list_cache_file, 'wb') as f:
                    pickle.dump(self.data_list, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(self.data_list_cache_file, 'rb') as f:
                    self.data_list = pickle.load(f)
            print('Finish Preparing ' + self.split + ' data')

                    
    # Pytorch DataLoader stuff
    def __len__(self):
        if self.split == 'test':
            return len(self.data_dict)
        else: # train and val
            return len(self.data_list)

    def __getitem__(self, idx):
        # In testing, don't get annotation
        if self.split == 'test':
            if self.data_dict[idx]['feat']['sub_loc'].size == 0:
                # Return dummy variable for pytorch
                return {
                    'sub_fc7': 0,
                    'obj_fc7': 0,
                    'sub_loc': 0,
                    'obj_loc': 0,
                    'union_fc7': 0,
                    'union_loc': 0,
                    'obj_wordemb': 0,
                    'sub_wordemb': 0,
                }

            sub_boxes_enc = new_bbox_transform(self.data_dict[idx]['feat']['sub_loc'],
                                               self.data_dict[idx]['feat']['obj_loc'],
                                               self.data_dict[idx]['im_w'],
                                               self.data_dict[idx]['im_h'])
            obj_boxes_enc = new_bbox_transform(self.data_dict[idx]['feat']['obj_loc'],
                                               self.data_dict[idx]['feat']['sub_loc'],
                                               self.data_dict[idx]['im_w'],
                                               self.data_dict[idx]['im_h'])
            union_boxes_enc = bbox_transform(self.data_dict[idx]['feat']['sub_loc'],
                                             self.data_dict[idx]['feat']['obj_loc'],
                                             self.data_dict[idx]['im_w'],
                                             self.data_dict[idx]['im_h'])

            if self.use_lang:
                sub_wordemb, obj_wordemb = get_obj_wordemb(self.data_dict[idx]['sub_obj'], self.word_emb[0])

                return {
                    'sub_fc7': self.data_dict[idx]['feat']['sub_fc7'].astype(np.float32), 
                    'obj_fc7': self.data_dict[idx]['feat']['obj_fc7'].astype(np.float32),
                    'sub_loc': sub_boxes_enc.astype(np.float32),
                    'obj_loc': obj_boxes_enc.astype(np.float32),
                    #'sub_loc': self.data_dict[idx]['feat']['sub_loc'].astype(np.float32),
                    #'obj_loc': self.data_dict[idx]['feat']['obj_loc'].astype(np.float32),
                    #'union_loc': self.data_dict[idx]['feat']['union_loc'].astype(np.float32),
                    'union_loc': union_boxes_enc.astype(np.float32),
                    'union_fc7': self.data_dict[idx]['feat']['union_fc7'].astype(np.float32),
                    'sub_wordemb': sub_wordemb.astype(np.float32),
                    'obj_wordemb': obj_wordemb.astype(np.float32),
                    'anno': self.data_dict[idx]['anno']['gt_labels']
                }


            else:
                return {
                    'sub_fc7': self.data_dict[idx]['feat']['sub_fc7'].astype(np.float32), 
                    'obj_fc7': self.data_dict[idx]['feat']['obj_fc7'].astype(np.float32),
                    'sub_loc': sub_boxes_enc.astype(np.float32),
                    'obj_loc': obj_boxes_enc.astype(np.float32),
                    #'sub_loc': self.data_dict[idx]['feat']['sub_loc'].astype(np.float32),
                    #'obj_loc': self.data_dict[idx]['feat']['obj_loc'].astype(np.float32),
                    #'union_loc': self.data_dict[idx]['feat']['union_loc'].astype(np.float32),
                    'union_loc': union_boxes_enc.astype(np.float32),
                    'union_fc7': self.data_dict[idx]['feat']['union_fc7'].astype(np.float32),
                    #'anno': self.data_dict[idx]['anno']['gt_labels'] if self.split == 'test' else self.data_dict[idx]['anno'],
                    'anno': self.data_dict[idx]['anno']['gt_labels']
                }
        else: # train /val
            sample_list = self.data_list[idx]

            sub_box = sample_list[0]
            obj_box = sample_list[1]
            width = sample_list[7]
            height = sample_list[8]
            sub_box_enc = new_bbox_transform(sub_box[np.newaxis, :], obj_box[np.newaxis, :], width, height)[0]
            obj_box_enc = new_bbox_transform(obj_box[np.newaxis, :], sub_box[np.newaxis, :], width, height)[0]
            union_box_enc = bbox_transform(sub_box[np.newaxis, :], obj_box[np.newaxis, :], width, height)[0]

            if self.use_lang:
                return {
                    'sub_fc7': sample_list[3].astype(np.float32),
                    'obj_fc7': sample_list[4].astype(np.float32),
                    'sub_loc': sub_box_enc.astype(np.float32),
                    'obj_loc': obj_box_enc.astype(np.float32),
                    #'sub_loc': sample_list[0].astype(np.float32),
                    #'obj_loc': sample_list[1].astype(np.float32),
                    #'union_loc': sample_list[2].astype(np.float32),
                    'union_loc': union_box_enc.astype(np.float32),
                    'union_fc7': sample_list[5].astype(np.float32),
                    'sub_wordemb': sample_list[9].astype(np.float32),
                    'obj_wordemb': sample_list[10].astype(np.float32),
                    'anno': sample_list[6] .astype(np.int16)
                }
            else:
                return {
                    'sub_fc7': sample_list[3].astype(np.float32),
                    'obj_fc7': sample_list[4].astype(np.float32),
                    'sub_loc': sub_box_enc.astype(np.float32),
                    'obj_loc': obj_box_enc.astype(np.float32),
                    #'sub_loc': sample_list[0].astype(np.float32),
                    #'obj_loc': sample_list[1].astype(np.float32),
                    #'union_loc': sample_list[2].astype(np.float32),
                    'union_loc': union_box_enc.astype(np.float32),
                    'union_fc7': sample_list[5].astype(np.float32),
                    'anno': sample_list[6] .astype(np.int16)
                }

    def convert_dict_to_list(self):
        """Convert train/val data_dict to list for better use of batch training"""
        data_list = []
        for im_id, im_dict in self.data_dict.items():
            if im_dict['anno'].size != 0:
                sub_loc = im_dict['feat']['sub_loc']
                obj_loc = im_dict['feat']['obj_loc']
                union_loc = im_dict['feat']['union_loc']
                sub_fc7 = im_dict['feat']['sub_fc7']
                obj_fc7 = im_dict['feat']['obj_fc7']
                union_fc7 = im_dict['feat']['union_fc7']
                anno = im_dict['anno']
                im_w = im_dict['im_w']
                im_h = im_dict['im_h']
                for i in range(len(anno)):
                    data_list.append([sub_loc[i], obj_loc[i], union_loc[i],
                                      sub_fc7[i], obj_fc7[i], union_fc7[i],
                                      anno[i], im_w, im_h, self.word_emb[0][anno[i, 0]-1], 
                                      self.word_emb[0][anno[i, 1]-1]])
        return data_list

    # Objects/relations id part
    def get_obj_classes(self):
        """ Return all object classes that exist """
        object_list = sio.loadmat(os.path.join(self.root, 'objectListN.mat'))
        object_list = object_list['objectListN'].squeeze()
        classes = ('__background__', )
        for i in range(len(object_list)):
            classes += (str(object_list[i][0]), )

        return classes

    def get_rel_classes(self):
        """ Return all relation classes that exist """
        rel_list = sio.loadmat(os.path.join(self.root, 'predicate.mat'))
        rel_list = rel_list['predicate'].squeeze()
        classes = ()
        for i in range(len(rel_list)):
            classes += (str(rel_list[i][0]), )

        return classes

    def _load_image_set_index(self):
        """ Load the indexes listed in this dataset's. (ALl images in images/)"""
        if self.split == 'val' or self.split == 'test':
            image_set_file = os.path.join(self.root, 'ImageSets', 'test.txt')
        else:
            image_set_file = os.path.join(self.root, 'ImageSets', 'train.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
          image_index = [x.split()[0] for x in f.readlines()]
        return image_index
            
    # Evaluation stuff
    def evaluate(self, pred_result, n_rels=5, obj_co_occur=None):
        """Run Evaluation"""
        all_tuple_path = os.path.join(self.root, self.split, 'all_tuple.pkl')
        setup(pred_result, self.data_dict, all_tuple_path, n_rels=n_rels, obj_co_occur=obj_co_occur)

        print('#######  Top recall results  #######')
        recall100_r = top_recall_relationship(100, all_tuple_path)
        recall50_r = top_recall_relationship(50, all_tuple_path)
        print('Relationship Det. R@100: {:.02f}'.format(100 * recall100_r))
        print('Relationship Det. R@50: {:.02f}'.format(100 * recall50_r))
        
        recall100_p = top_recall_phrase(100, all_tuple_path)
        recall50_p = top_recall_phrase(50, all_tuple_path)
        print('Phrase Det. R@100: {:.02f}'.format(100 * recall100_p))
        print('Phrase Det. R@50: {:.02f}'.format(100 * recall50_p))

        if self.use_gt:
            recall100_pre = top_recall_predicate(100, all_tuple_path)
            recall50_pre = top_recall_predicate(50, all_tuple_path)
            print('Predicate Det. R@100: {:.02f}'.format(100 * recall100_pre))
            print('Predicate Det. R@50: {:.02f}'.format(100 * recall50_pre))


