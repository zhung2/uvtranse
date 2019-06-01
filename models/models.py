import os, sys
import torch
import numpy as np
import pickle
import math
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#import models.initialize as initialize
import initialize
import torch.nn as nn
import torch.nn.functional as F
#import models.WeightedBCEWithLogitsLoss as wb
import WeightedBCEWithLogitsLoss as wb
#import models.SoftCrossEntropyLoss as SCL
import SoftCrossEntropyLoss as SCL
#import models.L2Loss as L2Loss
import L2Loss as L2Loss

# Function

def create_model(opt):
    """Get model"""
    model = None
    #print(opt.model)
    if opt.model == 'union':
        from .union_model import UnionModel
        model = UnionModel(opt)
    elif opt.model == 'vtranse':
        from .vtranse_model import VtranseModel
        model = VtranseModel(opt)
    elif opt.model == 'combine':
        from .combined_model import ComebinedModel
        model = ComebinedModel(opt)
    else:
        raise ValueError('Model [{:s}] not recognized.'.format(opt.model))

    return model

def create_criterion(opt, weights, pos_w):
    """Get criterion"""
    if opt.isTrain:
        if opt.loss == 'ce':
            criterion = torch.nn.CrossEntropyLoss(weight=weights)
            print('Using CrossEntropyLoss')
        elif opt.loss == 'bce':
            #criterion = torch.nn.BCEWithLogitsLoss()
            #criterion = wb.WeightedBCEWithLogitsLoss(pos_weight=pos_w)
            #pos_w = (torch.ones(70) * 50).cuda()
            criterion = wb.WeightedBCEWithLogitsLoss(pos_weight=pos_w)
            print('Using BinaryCrossEntropyLoss')
        elif opt.loss == 'kl':
            criterion = (torch.nn.CrossEntropyLoss(), SCL.SoftCrossEntropyLoss())
            print('Using KLDivLoss')
        else:
            raise ValueError('Model [{:s}] not recognized.'.format(opt.model))

        return criterion
    else:
        return None

def create_final_layer(opt):
    """Get criterion"""
    if not opt.isTrain:
        if opt.loss == 'ce':
            final_layer = nn.Softmax(dim=1)
        elif opt.loss == 'bce':
            final_layer = nn.Sigmoid()
        elif opt.loss == 'kl':
            final_layer = nn.Softmax(dim=1)
        else:
            raise ValueError('Loss [{:s}] not recognized.'.format(opt.loss))

        return final_layer
    else:
        return None

def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        #print(type(p.data), p.size())
        #print(k)
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()
    norm = (clip_norm / max(totalnorm, clip_norm))
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)

# Class

class Model(object):
    """Model class that provide all networks with same interface"""
    def __init__(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.name = opt.name
        self.gpu_ids = opt.gpu_ids
        self.save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.feat_net)
        self.final_layer = create_final_layer(opt)
        self.model = create_model(opt)

        self.C = opt.balance 
        if self.C:
            self.constriant = L2Loss.L2Loss()

        # Load prior statistics
        self.setup_prior()
        # Load word embedding for BCE
        self.load_precompute_word_similarity()

        if len(self.gpu_ids):
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
        elif torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably run with --gpu_ids")

        if self.isTrain:
            initialize.init_weights(self.model, init_type=opt.init_type)

            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.lr, momentum=0.9,
                                              weight_decay=self.opt.weight_decay)
            self.scheduler = initialize.get_scheduler(self.optimizer, opt)
            if opt.continue_train:
                self.load_model(opt.which_epoch)

            # DataParallel only in train
            if len(self.opt.gpu_ids) > 1:
                self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)

            self.criterion = create_criterion(opt, self.class_weight, self.pos_w)

            self.model.train()
        else:
            # Testing
            self.load_model(opt.which_epoch)
            self.model.eval()


        print('model [{:s}] was created'.format(opt.model))
    
    def set_input(self, input):
        """Turn Tensor to Variable"""
        if len(self.gpu_ids):
            self.sub_loc = Variable(input['sub_loc'].cuda())
            self.obj_loc = Variable(input['obj_loc'].cuda())
            self.union_loc = Variable(input['union_loc'].cuda())
            self.sub_fc7 = Variable(input['sub_fc7'].cuda())
            self.obj_fc7 = Variable(input['obj_fc7'].cuda())
            self.union_fc7 = Variable(input['union_fc7'].cuda())
            #torch.set_printoptions(threshold=10000)
            #print(self.sub_fc7[0][0])
            if self.opt.use_lang:
                self.sub_wordemb = Variable(input['sub_wordemb'].cuda())
                self.obj_wordemb = Variable(input['obj_wordemb'].cuda())
                pred_word_emb_file = os.path.join('/data/zhung2/datasets/vrd/', 'pred_emb.pth')
                pred_word_emb = torch.load(pred_word_emb_file)
                self.pred_wordemb = Variable(pred_word_emb.cuda())

        else:
            self.sub_loc = Variable(input['sub_loc'])
            self.obj_loc = Variable(input['obj_loc'])
            self.union_loc = Variable(input['union_loc'])
            self.sub_fc7 = Variable(input['sub_fc7'])
            self.obj_fc7 = Variable(input['obj_fc7'])
            self.union_fc7 = Variable(input['union_fc7'])
            if self.opt.use_lang:
                self.sub_wordemb = Variable(input['sub_wordemb'])
                self.obj_wordemb = Variable(input['obj_wordemb'])
                pred_word_emb_file = os.path.join(self.opt.dataroot, 'pred_emb.pth')
                pred_word_emb = torch.load(pred_word_emb_file)
                self.pred_wordemb = Variable(pred_word_emb)

        self.anno = input['anno']

    def get_test_result(self):
        """Run network and return detections for an image"""
        if self.opt.use_lang:
            pred_predict, vis_pred, sub_vec, \
            obj_vec, union_vec = self.model(self.sub_loc[0], self.sub_fc7[0], self.obj_loc[0],
                                            self.obj_fc7[0], self.union_loc[0], self.union_fc7[0],
                                            self.sub_wordemb[0], self.obj_wordemb[0], self.pred_wordemb[0])
        else:
            pred_predict, sub_vec, \
            obj_vec, union_vec = self.model(self.sub_loc[0], self.sub_fc7[0], self.obj_loc[0],
                                            self.obj_fc7[0], self.union_loc[0], self.union_fc7[0])

        if self.final_layer:
            if self.opt.use_lang:
                pred_predict = (self.final_layer(pred_predict).data.cpu().numpy() + \
                                self.final_layer(vis_pred).data.cpu().numpy()) / 2.0
            else:
                pred_predict = self.final_layer(pred_predict).data.cpu().numpy()
        else:
            if self.opt.use_lang:
                pred_predict = (pred_predict.data.cpu().numpy() + \
                                vis_pred.data.cpu().numpy()) / 2.0
            else:
                pred_predict = pred_predict.data.cpu().numpy()


        if self.opt.loss == 'kl' and self.opt.use_q:
            rule_constraint = self.similarity_mat[pred_predict.argmax(axis=1)] 
            rule_constraint = np.exp(self.C) * rule_constraint
            pred_predict = pred_predict * 1.0 * rule_constraint.cpu().numpy()
            pred_predict = pred_predict / (np.linalg.norm(pred_predict, ord=2, axis=1, keepdims=True) + 1e-8)


        return pred_predict

    def get_author_test_result(self):
        """Run network and return scores for 70 predicates for sub/objs"""
        if self.opt.use_lang:
            pred_predict, vis_pred, \
            sub_vec, obj_vec, union_vec = self.model(self.sub_loc, self.sub_fc7, self.obj_loc,
                                                     self.obj_fc7, self.union_loc, self.union_fc7,
                                                     self.sub_wordemb, self.obj_wordemb, self.pred_wordemb)
        else:
            pred_predict, sub_vec, obj_vec, union_vec = self.model(self.sub_loc, self.sub_fc7, self.obj_loc,
                                                                   self.obj_fc7, self.union_loc, self.union_fc7)
        #pred_predict = self.model((self.sub_loc, self.sub_fc7, self.obj_loc,
                                   #self.obj_fc7, self.union_loc, self.union_fc7))
        pred_predict = pred_predict
        if self.final_layer:
            if self.opt.use_lang:
                pred_predict = (self.final_layer(pred_predict).data.cpu().numpy() + \
                                self.final_layer(vis_pred).data.cpu().numpy()) / 2.0
            else:
                pred_predict = self.final_layer(pred_predict).data.cpu().numpy()
        else:
            if self.opt.use_lang:
                pred_predict = (pred_predict.data.cpu().numpy() + \
                                vis_pred.data.cpu().numpy()) / 2.0
            else:
                pred_predict = pred_predict.data.cpu().numpy()
        return pred_predict

    def forward(self):
        """Set gt_label"""
        predicate_id = self.anno[:, 2]

        if self.criterion.__class__.__name__ == 'CrossEntropyLoss' or self.criterion.__class__.__name__== 'NLLLoss':
            # For softmax
            if len(self.gpu_ids):
                self.pred_label = Variable(predicate_id.long().cuda(), requires_grad=False)
            else:
                self.pred_label = Variable(predicate_id.long(), requires_grad=False)
        elif self.criterion.__class__.__name__ == 'BCEWithLogitsLoss' or self.criterion.__class__.__name__ == 'BCELoss' \
            or self.criterion.__class__.__name__ == 'WeightedBCEWithLogitsLoss':
            if self.similarity_mat is not None:
                pred_label_onehot = self.similarity_mat[predicate_id.long()]
            else:
                pred_label_onehot = torch.FloatTensor(predicate_id.shape[0], 70)
                pred_label_onehot.zero_()
                pred_label_onehot.scatter_(1, predicate_id.long().unsqueeze(1), 1)
            if len(self.gpu_ids):
                self.pred_label = Variable(pred_label_onehot.cuda(), requires_grad=False)
            else:
                self.pred_label = Variable(pred_label_onehot, requires_grad=False)
        elif isinstance(self.criterion, tuple) and self.criterion[0].__class__.__name__ == 'CrossEntropyLoss':
            assert self.similarity_mat is not None, 'KLDivLoss like loss need to use similarity_mat'
            rule_constraint = self.similarity_mat[predicate_id.long()] # For KL part
            rule_constraint = np.exp(self.C) * rule_constraint
            if len(self.gpu_ids):
                gt_label = Variable(predicate_id.long().cuda(), requires_grad=False) # For CELOSS part
                self.pred_label = (gt_label, Variable(rule_constraint.cuda(), requires_grad=False))
            else:
                gt_label = Variable(predicate_id.long(), requires_grad=False) # For CELOSS part
                self.pred_label = (gt_label, Variable(rule_constraint, requires_grad=False))
        else:
            raise NotImplementedError('Currently has no this loss')

    def save_model(self, epoch_label):
        save_filename = '{:s}_net.pth'.format(str(epoch_label))
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save({
            'model': self.model.module.state_dict() if len(self.opt.gpu_ids) > 1 else self.model.state_dict(),
            'epoch': epoch_label + 1,
            #'optimizer': self.optimizer.state_dict(),
            #'scheduler': self.scheduler.state_dict(),
        }, save_path)


        #if self.gpu_ids and torch.cuda.is_available():
            #self.model.cuda()

    def get_validate_loss(self):
        """Run Validation_loss per image for the network"""
        self.model.eval()
        self.forward()
        if self.opt.use_lang:
            pred_predict, vis_pred,  \
            sub_vec, obj_vec, union_vec = self.model(self.sub_loc, self.sub_fc7, self.obj_loc,
                                                     self.obj_fc7, self.union_loc, self.union_fc7,
                                                     self.sub_wordemb, self.obj_wordemb, self.pred_wordemb)

        else:
            pred_predict, sub_vec, obj_vec, union_vec = self.model(self.sub_loc, self.sub_fc7, self.obj_loc,
                                                                   self.obj_fc7, self.union_loc, self.union_fc7)

        if self.opt.loss == 'kl':
            q_y_given_x = F.softmax(pred_predict, dim=1) * 1.0 * self.pred_label[1]
            q_y_given_x = F.normalize(q_y_given_x, dim=1)
            q_y_given_x = Variable(q_y_given_x, requires_grad=False)
            loss = (1. - self.alpha) * self.criterion[0](pred_predict, self.pred_label[0]) + \
                    self.alpha * self.criterion[1](pred_predict, q_y_given_x)
        else:
            loss = self.criterion(pred_predict, self.pred_label)
            true_loss = loss.item()
            if self.C != 0:
                # Add constriant for vectors
                #target_one_norm = torch.ones(sub_vec.size(0)).cuda()
                target_one_norm = Variable(torch.ones(sub_vec.size(0)).cuda(), requires_grad=False)
                sub_vec_norm = sub_vec.norm(p=2, dim=1)
                obj_vec_norm = obj_vec.norm(p=2, dim=1)
                union_vec_norm = union_vec.norm(p=2, dim=1)
                #target_two_norm = target_one_norm
                target_two_norm = Variable(1*target_one_norm.data, requires_grad=False)
                loss += self.C * (self.constriant(sub_vec_norm, target_one_norm) + \
                                  self.constriant(obj_vec_norm, target_one_norm) + \
                                  self.constriant(union_vec_norm, target_two_norm))
            if self.opt.use_lang:
                lang_loss = self.criterion(vis_pred, self.pred_label)
                loss += lang_loss
                true_loss += lang_loss.item()

        self.model.train()
        return loss.item(), true_loss

    def load_model(self, epoch_label):
        save_filename = '{:s}_net.pth'.format(str(epoch_label))
        save_path = os.path.join(self.save_dir, save_filename)
        print("=> loading checkpoint '{}'".format(save_path))
        checkpoint = torch.load(save_path)
        self.opt.epoch_count = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        #self.optimizer.load_state_dict(checkpoint['optimizer'])
        #self.scheduler.load_state_dict(checkpoint['scheduler'])
        print('=> load model successfully!')
        #self.model.load_state_dict(torch.load(save_path))

    def backward_loss(self):
        if self.opt.use_lang:
            pred_predict, vis_pred, \
            sub_vec, obj_vec, union_vec = self.model(self.sub_loc, self.sub_fc7, self.obj_loc,
                                                     self.obj_fc7, self.union_loc, self.union_fc7,
                                                     self.sub_wordemb, self.obj_wordemb, self.pred_wordemb)
        else:
            pred_predict, sub_vec, obj_vec, union_vec = self.model(self.sub_loc, self.sub_fc7, self.obj_loc,
                                                                   self.obj_fc7, self.union_loc, self.union_fc7)

        if self.opt.loss == 'kl':
            q_y_given_x = F.softmax(pred_predict, dim=1) * 1.0 * self.pred_label[1]
            q_y_given_x = F.normalize(q_y_given_x, dim=1)
            q_y_given_x = Variable(q_y_given_x, requires_grad=False)
            loss = (1. - self.alpha) * self.criterion[0](pred_predict, self.pred_label[0]) + \
                    self.alpha * self.criterion[1](pred_predict, q_y_given_x)
        else:
            loss = self.criterion(pred_predict, self.pred_label)
            if self.C != 0:
                # Add constriant for vectors
                #target_one_norm = torch.ones(sub_vec.size(0)).cuda()
                target_one_norm = Variable(torch.ones(sub_vec.size(0)).cuda(), requires_grad=False)
                sub_vec_norm = sub_vec.norm(p=2, dim=1)
                obj_vec_norm = obj_vec.norm(p=2, dim=1)
                union_vec_norm = union_vec.norm(p=2, dim=1)
                #target_two_norm = 3 * target_one_norm
                #target_two_norm = target_one_norm
                target_two_norm = Variable(1 * target_one_norm.data, requires_grad=False)
                loss += self.C * (self.constriant(sub_vec_norm, target_one_norm) + \
                                  self.constriant(obj_vec_norm, target_one_norm) + \
                                  self.constriant(union_vec_norm, target_two_norm))
            if self.opt.use_lang:
                loss += self.criterion(vis_pred, self.pred_label)

        loss.backward()
        self.loss = loss.item()

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward_loss()
        #if self.opt.use_lang:
            #clip_gradient(self.model, 5.)
        self.optimizer.step()

    def get_loss(self):
        return self.loss

    def update_learning_rate(self, val_loss):
        """lr schedule"""
        if self.opt.lr_policy == 'plateau':
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        #print('learning rate = {:.7f}'.format(lr))
        return lr
    
    def update_alpha(self, cur_iter):
        """update weight of the distillation loss at iteration t"""
        #self.alpha = min(0.1, 1 - 0.95 ** ((cur_epoch-1) / (total_epoch-1)))
        #self.alpha = min(0.9, 1 - 0.9 ** (cur_epoch-1.))
        self.alpha = 1. - max(0.95 ** cur_iter, 0.1)
        return self.alpha


    def setup_prior(self):
        if self.opt.use_prior:
            prior_path = os.path.join(self.opt.dataroot, 'so_prior.pkl')
            with open(prior_path, 'rb') as f:
                # shape: (n_obj_classes+1, n_obj_classes+1, n_predicates)
                self.prior = pickle.load(f)

            total_rel = np.sum(self.prior)
            # Number of rels for each predicate
            pred_count = np.sum(self.prior, axis=(0,1))

            # For softmax, class weight
            if self.opt.loss == 'ce' or self.opt.loss == 'kl':
                pred_prior_prob = pred_count / float(total_rel)
                self.class_weight = 1.0 / pred_prior_prob
                self.class_weight = self.class_weight / np.amax(self.class_weight)
                #print(self.class_weight)
                self.class_weight = np.clip(self.class_weight, 0.1, 1)
                self.class_weight = torch.FloatTensor(self.class_weight)
                if len(self.gpu_ids):
                    self.class_weight = self.class_weight.cuda()
                self.pos_w = None
            # For Binary Cross Entropy, pos weight (increase recall)
            elif self.opt.loss == 'bce':
                # For BCE positive weight
                self.pos_w = (total_rel - pred_count) / pred_count 
                #self.pos_w = np.clip(self.pos_w, 1.0, 50.0)
                self.pos_w = np.clip(self.pos_w, 1.0, 10.0)
                #print(self.pos_w)
                self.pos_w = torch.FloatTensor(self.pos_w)
                if len(self.gpu_ids):
                    self.pos_w = self.pos_w.cuda()
                self.class_weight = None

            if self.opt.use_co_occur:
                co_occur_path = os.path.join(self.opt.dataroot, 'so_co_occur.pkl')
                with open(co_occur_path, 'rb') as f:
                    # shape: (n_obj_classes+1, n_obj_classes+1, n_predicates)
                    self.co_occur = pickle.load(f)

                # Rels count between object pairs, shape (n_obj_classes+1, n_obj_classes+1), 1: background
                obj_rels = np.sum(self.prior, axis=2)
                self.relevance = np.zeros(obj_rels.shape)
                # Avoid divide by zero
                self.relevance = np.divide(obj_rels, self.co_occur, 
                                           out=np.zeros_like(obj_rels), where=self.co_occur!=0)

                # Threshold to 0.01
                self.relevance[self.relevance < 0.01] = 0.01
                # Background should be 0
                self.relevance[0, :] = 0
                self.relevance[:, 0] = 0
            else:
                self.relevance = None
        else:
            self.class_weight = None
            self.pos_w = None
            self.relevance = None


    def load_precompute_word_similarity(self):
        if self.opt.use_word_sim:
            precomputed_sim = os.path.join(self.opt.dataroot, 'pred_sim.pth')
            if not os.path.isfile(precomputed_sim):
                self.similarity_mat = self.compute_word_similarity()
                torch.save(self.similarity_mat, precomputed_sim)
            else:
                self.similarity_mat = torch.load(precomputed_sim)

            #if len(self.gpu_ids):
                #self.similarity_mat = self.similarity_mat.cuda()
        else:
            self.similarity_mat = None

    
    def compute_word_similarity(self):
        pred_word_emb_file = os.path.join(self.opt.dataroot, 'pred_emb.pth')
        pred_word_emb = torch.load(pred_word_emb_file)

        # First dim: n_preds, Second dim: similarity
        similarity_mat = torch.FloatTensor(pred_word_emb.size(0), pred_word_emb.size(0))
        for i in range(pred_word_emb.size(0)):
            for j in range(pred_word_emb.size(0)):
                prod = torch.dot(pred_word_emb[i], pred_word_emb[j])
                # Normalize -> cosine similarity
                sim = float(prod) / (torch.norm(pred_word_emb[i]) * torch.norm(pred_word_emb[j]) + 1e-10)
                similarity_mat[i, j] = torch.exp(10*sim)
            similarity_mat[i] = similarity_mat[i] / (torch.norm(similarity_mat[i]) + 1e-10)

            #top5_ind, top5_score = torch.topk(similarity_mat[i], 5)
            #print('index', i, 'top5_ind', top5_ind, 'top5_score', top5_score)
        return similarity_mat



