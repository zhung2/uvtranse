import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnionModel(nn.Module):
    def __init__(self, opt):
        """Set up network module in UnionModel"""
        super(UnionModel, self).__init__()
        self.name = 'Union'
        self.fc7_dim = opt.fc7_dim
        self.keep_prob = opt.keep_prob

        # Extract ROI pooled feature
        # For subject
        self.sub_fc7_w = nn.Sequential(
            nn.Linear(self.fc7_dim, 512),
            nn.ReLU(True),
            #nn.Dropout(0.5),
        #)
            nn.Linear(512, 128), )
            #nn.Linear(512, 100), )
        # For object
        self.obj_fc7_w = nn.Sequential(
            nn.Linear(self.fc7_dim, 512),
            nn.ReLU(True),
            #nn.Dropout(0.5),
        #)
            nn.Linear(512, 128), )
            #nn.Linear(512, 100), )

        # For union
        self.union_fc7_w = nn.Sequential(
            nn.Linear(self.fc7_dim, 512),
            nn.ReLU(True),
            #nn.Dropout(0.5),
        #)
            nn.Linear(512, 128), )
            #nn.Linear(512, 100), )

        # Lang
        '''
        self.sub_word_w = nn.Sequential(
            nn.Linear(300, 128), )
        self.obj_word_w = nn.Sequential(
            nn.Linear(300, 128), )

        # Trnalation feature extract
        self.trans = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 128), )
        '''

        '''
        sub_context = [nn.Linear(self.fc7_dim, 256),
                       nn.ReLU(True)]
        self.sub_context = nn.Sequential(*sub_context)
        obj_context = [nn.Linear(self.fc7_dim, 256),
                       nn.ReLU(True)]
        self.obj_context = nn.Sequential(*obj_context)
        sub_obj_context = [nn.Linear(2*self.fc7_dim, 256),
                           nn.ReLU(True),
                           nn.Linear(256, 128)]
        self.sub_obj_context = nn.Sequential(*sub_obj_context)

        sub_union_context = [nn.Linear(2*self.fc7_dim, 256),
                             nn.ReLU(True),
                             nn.Linear(256, 128)]
        self.sub_union_context = nn.Sequential(*sub_union_context)
        obj_union_context = [nn.Linear(2*self.fc7_dim, 256),
                             nn.ReLU(True),
                             nn.Linear(256, 128)]
        self.obj_union_context = nn.Sequential(*obj_union_context)
        '''


        # Extract location feature (all sub, obj, union use the same)
        '''
        loc_w = [nn.Linear(2*4, 20),
                 nn.ReLU(True),
                 nn.Linear(20, 10),
                 nn.ReLU(True)]
        self.loc_w = nn.Sequential(*loc_w)

        union_loc_w = [nn.Linear(2*5+9, 32),
                       nn.ReLU(True),
                       #nn.BatchNorm1d(32),
                       #nn.Dropout(self.keep_prob),
                       nn.Linear(32, 16)]
                       #nn.ReLU(True)]
        self.union_loc_w = nn.Sequential(*union_loc_w)
        '''
        self.loc_w = nn.Sequential(
            nn.Linear(2*5+9, 32),
            nn.ReLU(True),
            nn.Linear(32, 16), )

        '''
        self.translation_emb = nn.Sequential(
            nn.Linear(128+16, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),)
        '''

        '''
        self.predicate_emb = nn.Sequential(
            nn.Linear(300, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),)
        '''


        # Weighted concatenate for sub, obj, loc, union
        self.cat_unionf = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.cat_unionl = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.cat_subf = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.cat_subl = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.cat_objf = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.cat_objl = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)

        # Finally 2-layer fully connected layer for predicting relation
        self.vis_extract = nn.Sequential(
            nn.Linear(128+16, 128),
            nn.ReLU(True),
        )
        self.vis_fc = nn.Sequential(
            #nn.Linear(100+16, 128),
            #nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 70), )

        self.gru = nn.GRU(input_size=300, hidden_size=100,
                          dropout=0.5, num_layers=2, bidirectional=True)

        self.final_fc = nn.Sequential(
            nn.Linear(600, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 70), )

    def forward(self, sub_loc, sub_fc7, obj_loc, obj_fc7,
               union_loc, union_fc7, sub_wordemb=None, obj_word_emb=None,
               pred_wordemb=None):
        """Run foward path
        Args:

        Returns:
            pred (ndarray): prediction without sigmoid (because of BCEWithLogitsLoss)
        """
        '''
        sub_fc7 = F.normalize(sub_fc7, dim=1)
        obj_fc7 = F.normalize(obj_fc7, dim=1)
        union_fc7 = F.normalize(union_fc7, dim=1)
        '''
        #sub_loc, sub_fc7, obj_loc, obj_fc7, union_loc, union_fc7 = input

        # Extract fc7 and location featue respectively
        sub_fc7_enc = self.sub_fc7_w(sub_fc7)
        #sub_loc_enc = self.loc_w(sub_loc)
        obj_fc7_enc = self.obj_fc7_w(obj_fc7)
        #obj_fc7_enc = self.sub_fc7_w(obj_fc7)
        #obj_loc_enc = self.loc_w(obj_loc)
        #sub_obj_fc7_enc = self.sub_fc7_w(torch.cat((sub_fc7, obj_fc7), -1))
        #sub_obj_loc_enc = self.loc_w(torch.cat((sub_loc, obj_loc), -1))

        #sub_lang_enc = self.sub_word_w(sub_wordemb)
        #obj_lang_enc = self.obj_word_w(obj_word_emb)

        #sub_fc7_enc = self.sub_to_rel(sub_fc7_enc)
        #obj_fc7_enc = self.obj_to_rel(obj_fc7_enc)
        #sub_fc7_enc = self.sub_to_rel_bn(sub_fc7_enc)
        #obj_fc7_enc = self.obj_to_rel_bn(obj_fc7_enc)

        union_fc7_enc = self.union_fc7_w(union_fc7)
        #union_fc7_enc = self.union_to_rel(union_fc7_enc)
        #union_fc7_enc = self.union_fc7_bn(union_fc7_enc)
        #union_loc_enc = self.union_low_w(union_loc)

        #sub_obj_context = self.sub_obj_context(torch.cat((sub_fc7, obj_fc7), -1))
        #sub_union_context = self.sub_union_context(torch.cat((sub_fc7, union_fc7), -1))
        #obj_union_context = self.obj_union_context(torch.cat((obj_fc7, union_fc7), -1))

        #sub = torch.cat((sub_fc7_enc, sub_lang_enc), -1)
        #obj = torch.cat((obj_fc7_enc, obj_lang_enc), -1)
        #union = torch.cat((union_fc7_enc, sub_lang_enc, obj_lang_enc), axis=-1)

        fc7_res = union_fc7_enc - sub_fc7_enc - obj_fc7_enc
        #fc7_res = union_fc7_enc - sub - obj
        #fc7_res = self.trans(fc7_res)
        #fc7_res = union_fc7_enc - sub_fc7_enc - obj_fc7_enc - \
                #sub_obj_context - sub_union_context - obj_union_context
        #fc7_res = union_fc7_enc - sub_obj_context - sub_union_context - obj_union_context
        #fc7_res = F.relu(fc7_res)
        #fc7_res = F.dropout(fc7_res, self.keep_prob)

        #fc7_res = self.union_fc7_bn(fc7_res)
        #fc7_res = union_fc7_enc - sub_obj_fc7_enc
        #loc_res = union_loc_enc - sub_loc_enc - obj_loc_enc
        #loc_res = union_loc_enc - sub_obj_loc_enc
        loc_res = self.loc_w(torch.cat((sub_loc, obj_loc, union_loc), -1))
        #loc_res = F.relu(loc_res)
        pred_vec = torch.cat((self.cat_unionf * fc7_res, self.cat_unionl * loc_res), -1)
        pred_vec = F.relu(pred_vec)

        vis_feat = self.vis_extract(pred_vec)
        vis_score = self.vis_fc(vis_feat)
        #pred_vec = F.dropout(pred_vec, self.keep_prob)

        vis_pred = F.pad(vis_feat, (86, 86)).unsqueeze_(0)
        sub_wordemb.unsqueeze_(0)
        obj_word_emb.unsqueeze_(0)
        triplet = torch.cat((sub_wordemb, vis_pred, obj_word_emb), 0)

        self.gru.flatten_parameters()
        output, hidden = self.gru(triplet, None)

        # Extract all output and send to final fc layer
        first_forward = output[0, :, :100]
        second_forward = output[1, :, :100]
        final_forward = output[2, :, :100]

        first_backward = output[2, :, 100:]
        second_backward = output[1, :, 100:]
        final_backward = output[0, :, 100:]
        gru_out_concat = torch.cat((first_forward, second_forward, final_forward,
                                    first_backward, second_backward, final_backward), -1)


        pred = self.final_fc(gru_out_concat)

        # Get Sub, Obj, Union feature using weighted concatenated
        #sub = torch.cat((self.cat_subf * sub_fc7_enc, self.cat_subl * sub_loc_enc), -1)
        #obj = torch.cat((self.cat_objf * obj_fc7_enc, self.cat_objl * obj_loc_enc), -1)
        #union = torch.cat((self.cat_unionf * union_fc7_enc, self.cat_unionl * union_loc_enc), -1)

        # Get subtraction
        #pred_vec = union - sub - obj
        #pred = self.final_fc(pred_vec)
        #translation = self.translation_emb(pred_vec)
        #predicate = self.predicate_emb(pred_wordemb)

        #pred = torch.matmul(translation, predicate.t())

        return pred, vis_score, sub_fc7_enc, obj_fc7_enc, union_fc7_enc

