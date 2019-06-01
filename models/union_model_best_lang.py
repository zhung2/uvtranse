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
            nn.Linear(512, 128), )
        # For object
        self.obj_fc7_w = nn.Sequential(
            nn.Linear(self.fc7_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 128), )

        # For union
        self.union_fc7_w = nn.Sequential(
            nn.Linear(self.fc7_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 128), )

        # Lang

        # Extract location feature (all sub, obj, union use the same)
        self.loc_w = nn.Sequential(
            nn.Linear(2*5+9, 32),
            nn.ReLU(True),
            nn.Linear(32, 16), )

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
        #sub_loc, sub_fc7, obj_loc, obj_fc7, union_loc, union_fc7 = input

        # Extract fc7 and location featue respectively
        sub_fc7_enc = self.sub_fc7_w(sub_fc7)
        obj_fc7_enc = self.obj_fc7_w(obj_fc7)

        union_fc7_enc = self.union_fc7_w(union_fc7)

        fc7_res = union_fc7_enc - sub_fc7_enc - obj_fc7_enc
        loc_res = self.loc_w(torch.cat((sub_loc, obj_loc, union_loc), -1))
        pred_vec = torch.cat((self.cat_unionf * fc7_res, self.cat_unionl * loc_res), -1)
        pred_vec = F.relu(pred_vec)

        vis_feat = self.vis_extract(pred_vec)
        vis_score = self.vis_fc(vis_feat)

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


        return pred, vis_score, sub_fc7_enc, obj_fc7_enc, union_fc7_enc

