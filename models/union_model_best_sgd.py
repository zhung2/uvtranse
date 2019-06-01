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
            nn.Linear(512, 256), )
        # For object
        self.obj_fc7_w = nn.Sequential(
            nn.Linear(self.fc7_dim, 512), 
            nn.ReLU(True), 
            nn.Linear(512, 256), )


        # For union
        self.union_fc7_w = nn.Sequential(
            nn.Linear(self.fc7_dim, 512), 
            nn.ReLU(True), 
            nn.Linear(512, 256), )

        # Lang


        # Extract location feature (all sub, obj, union use the same)
        self.loc_w = nn.Sequential(
            nn.Linear(2*5+9, 32),
            nn.ReLU(True),
            nn.Linear(32, 16), )
            #nn.ReLU(True), )



        # Weighted concatenate for sub, obj, loc, union
        self.cat_unionf = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.cat_unionl = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)

        # Finally 2-layer fully connected layer for predicting relation
        self.final_fc = nn.Sequential(
            nn.Linear(256+16, 256),
            nn.ReLU(True),
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
        #loc_res = F.relu(loc_res)
        pred_vec = torch.cat((self.cat_unionf * fc7_res, self.cat_unionl * loc_res), -1)
        pred_vec = F.relu(pred_vec)
        #pred_vec = F.dropout(pred_vec, self.keep_prob)

        # Get subtraction
        pred = self.final_fc(pred_vec)

        return pred, sub_fc7_enc, obj_fc7_enc, union_fc7_enc

