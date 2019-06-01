# Modified from Lu's Matlab evaluation code

import os, sys
import time
import pickle
import numpy as np

#n_rels = 5 # Slect top n_rels score for each triplet only

def get_sub_obj_overlap(gt_sub_bbox, gt_obj_bbox, sub_bbox, obj_bbox):
    inter_s = [max(gt_sub_bbox[0], sub_bbox[0]), max(gt_sub_bbox[1], sub_bbox[1]),
               min(gt_sub_bbox[2], sub_bbox[2]), min(gt_sub_bbox[3], sub_bbox[3])]
    inter_s_w = inter_s[2] - inter_s[0] + 1
    inter_s_h = inter_s[3] - inter_s[1] + 1

    inter_o = [max(gt_obj_bbox[0], obj_bbox[0]), max(gt_obj_bbox[1], obj_bbox[1]),
               min(gt_obj_bbox[2], obj_bbox[2]), min(gt_obj_bbox[3], obj_bbox[3])]
    inter_o_w = inter_o[2] - inter_o[0] + 1
    inter_o_h = inter_o[3] - inter_o[1] + 1

    if inter_s_w > 0 and inter_s_h > 0 \
       and inter_o_w > 0 and inter_o_h > 0:
        union_s = (sub_bbox[2] - sub_bbox[0] + 1) * (sub_bbox[3] - sub_bbox[1] + 1) \
                    + (gt_sub_bbox[2] - gt_sub_bbox[0] + 1) * (gt_sub_bbox[3] - gt_sub_bbox[1] + 1) \
                    - inter_s_w * inter_s_h
        union_o = (obj_bbox[2] - obj_bbox[0] + 1) * (obj_bbox[3] - obj_bbox[1] + 1) \
                    + (gt_obj_bbox[2] - gt_obj_bbox[0] + 1) * (gt_obj_bbox[3] - gt_obj_bbox[1] + 1) \
                    - inter_o_w * inter_o_h
        overlap_s = inter_s_w * inter_s_h / union_s
        overlap_o = inter_o_w * inter_o_h / union_o
        return min(overlap_o, overlap_s)
    else:
        return 0


def get_overlap(gt_bbox, bbox):
    inter = [max(gt_bbox[0], bbox[0]), max(gt_bbox[1], bbox[1]),
             min(gt_bbox[2], bbox[2]), min(gt_bbox[3], bbox[3])]

    inter_w = inter[2] - inter[0] + 1
    inter_h = inter[3] - inter[1] + 1

    if inter_w > 0 and inter_h > 0:
        # computer overlap as inter / union
        union = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1) \
                + (gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1) \
                - inter_w * inter_h

        overlap = inter_w * inter_h / union
        return overlap
    else:
        return 0 


def setup(pred_result, data_dict, save_path, all_pred_id=np.arange(70), 
          valid_rel=None, n_rels=5, obj_co_occur=None):
    # Setup
    relation_triplet = [[] for _ in range(len(pred_result))]
    relation_score = [[] for _ in range(len(pred_result))]
    sub_boxes = [[] for _ in range(len(pred_result))]
    obj_boxes = [[] for _ in range(len(pred_result))]
    sub_scores = [[] for _ in range(len(pred_result))]
    obj_scores = [[] for _ in range(len(pred_result))]

    gt_sub_boxes = [[] for _ in range(len(pred_result))]
    gt_obj_boxes = [[] for _ in range(len(pred_result))]
    gt_labels = [[] for _ in range(len(pred_result))]

    # Get the value for Setup
    for i, preds in enumerate(pred_result):
        # Setup gt_box (make boxes 2 dimension)
        gt_sub_boxes[i] = data_dict[i]['anno']['gt_sub_boxes']
        if isinstance(gt_sub_boxes[i], np.ndarray) and gt_sub_boxes[i].ndim == 1:
            gt_sub_boxes[i] = gt_sub_boxes[i][np.newaxis, :]
        gt_obj_boxes[i] = data_dict[i]['anno']['gt_obj_boxes']
        if isinstance(gt_obj_boxes[i], np.ndarray) and gt_obj_boxes[i].ndim == 1:
            gt_obj_boxes[i] = gt_obj_boxes[i][np.newaxis, :]

        # Setup gt_labels
        gt_labels[i] = data_dict[i]['anno']['gt_labels']

        if preds == []:
            # No detections, then let all our stuff be []
            continue
        # Extract valid predicate, and sort along each row (triplet)
        valid_preds = preds[:, all_pred_id]
        # Normalize it
        #valid_preds = valid_preds / np.linalg.norm(valid_preds, axis=1, keepdims=True)

        # Add subject / object detection scores for evaluation
        # If don't use this, the accuracy will be 5~7% lower
        sub_score = data_dict[i]['sub_obj_score'][:, 0]
        sub_score = np.repeat(sub_score[:, np.newaxis], len(all_pred_id), axis=1)
        obj_score = data_dict[i]['sub_obj_score'][:, 1]
        obj_score = np.repeat(obj_score[:, np.newaxis], len(all_pred_id), axis=1)


        # Use co_occur to reorde sub_obj score
        if obj_co_occur is not None:
            sub_id = data_dict[i]['sub_obj'][:, 0]
            obj_id = data_dict[i]['sub_obj'][:, 1]
            # Get out the sub, obj prob for being annotated as relationship
            relevance_prob = obj_co_occur[sub_id, obj_id]
            relevance_prob = np.repeat(relevance_prob[:, np.newaxis], len(all_pred_id), axis=1)

            valid_preds = valid_preds + sub_score + obj_score + relevance_prob
        else:
            valid_preds = valid_preds + 1.0 *(sub_score + obj_score)

        # In numpy, sort will put the smallest element in index 0
        # Select top n_rels for each row, then flatten to 1d array
        sorted_ind = np.argsort(-valid_preds, axis=1)
        sorted_ind = sorted_ind[:, :n_rels].flatten()
        # Same for score, but score is negative for soting. 
        neg_valid_pred_sorted = np.sort(-valid_preds, axis=1)
        neg_valid_pred_sorted = neg_valid_pred_sorted[:, :n_rels].flatten()

        # Sort again according to score from differnet triplets
        neg_valid_pred_sorted_ind = np.argsort(neg_valid_pred_sorted)
        sorted_ind = sorted_ind[neg_valid_pred_sorted_ind]
        neg_valid_pred_sorted = neg_valid_pred_sorted[neg_valid_pred_sorted_ind]

        # Get pred_id
        pred_id = all_pred_id[sorted_ind]

        sub_id = data_dict[i]['sub_obj'][:, 0]
        #assert len(preds) == len(sub_id), '{:d} != {:d}'.format(len(preds), len(sub_id))
        # Make sub_id, obj_id the same size as rel_flat
        sub_id = np.repeat(sub_id, n_rels, axis=0)
        #assert sub_id.ndim == 1
        sub_id = sub_id[neg_valid_pred_sorted_ind]

        obj_id = data_dict[i]['sub_obj'][:, 1]
        obj_id = np.repeat(obj_id, n_rels, axis=0)
        obj_id = obj_id[neg_valid_pred_sorted_ind]

        # Order [sub, obj, predicate]
        relation_triplet[i] = np.concatenate((
            sub_id[:, np.newaxis], obj_id[:, np.newaxis], pred_id[:, np.newaxis]),
            axis=1)
        relation_score[i] = neg_valid_pred_sorted


        # If provide valid relation, prune other relationships not in here
        if valid_rel:
            valid_row = []
            for j in range(len(relation_triplet[i])):
                if list(relation_triplet[i][j]) in valid_rel:
                    valid_row.append(j)
            if valid_row:
                relation_triplet[i] = relation_triplet[i][valid_row, :]
                relation_score[i] = relation_score[i][valid_row]
            else:
                relation_triplet[i] = []
                relation_score[i] = [] 


        # Setup detected box (similar to sub_id)
        sub_loc = data_dict[i]['feat']['sub_loc']
        # Make box twot dimension
        if sub_loc.ndim == 1:
            sub_loc = sub_loc[np.newaxis, :]
        sub_loc = np.repeat(sub_loc, n_rels, axis=0)
        sub_boxes[i] = sub_loc[neg_valid_pred_sorted_ind, :]
        if valid_rel:
            if valid_row:
                sub_boxes[i] = sub_boxes[i][valid_row, :]
            else:
                sub_boxes[i] = []
        obj_loc = data_dict[i]['feat']['obj_loc']
        if obj_loc.ndim == 1:
            obj_loc = obj_loc[np.newaxis, :]
        obj_loc = np.repeat(obj_loc, n_rels, axis=0)
        obj_boxes[i] = obj_loc[neg_valid_pred_sorted_ind, :]
        if valid_rel:
            if valid_row:
                obj_boxes[i] = obj_boxes[i][valid_row, :]
            else:
                obj_boxes[i] = []


    all_data = (relation_triplet, relation_score, sub_boxes, obj_boxes,
                gt_sub_boxes, gt_obj_boxes, gt_labels)

    #print(relation_triplet[0][:10, :])
    with open(save_path, 'wb') as f:
        pickle.dump(all_data, f, pickle.HIGHEST_PROTOCOL)

    # Save matlab file for evaluation
    if True:
        import scipy.io as sio
        import os
        import copy
        #save_dir = save_path.split('/')[:-1]
        #mat_file = '/'.join(save_path.split('/')[:-1]) + '/' + 'relationship_det_result.mat'
        mat_file = '/'.join(save_path.split('/')[:-1]) + '/' + 'multi_relationship_det_result.mat'
        print(mat_file)
        new_relation_triplet = copy.deepcopy(relation_triplet)
        for i in range(len(relation_triplet)):
            if relation_triplet[i] == []:
                continue
            # sub (we also start from 1)
            new_relation_triplet[i][:, 0] = relation_triplet[i][:, 0]
            # pred (we start from 0, matlab from 1)
            new_relation_triplet[i][:, 1] = relation_triplet[i][:, 2] + 1.0
            # obj
            new_relation_triplet[i][:, 2] = relation_triplet[i][:, 1]
            relation_score[i] = -relation_score[i]
            # matlab start from 1
            sub_boxes[i] += 1
            obj_boxes[i] += 1
        #print(new_relation_triplet[0][:10, :])
        sio.savemat(mat_file, {
            'rlp_labels_ours': new_relation_triplet,
            'rlp_confs_ours': relation_score,
            'sub_bboxes_ours': sub_boxes,
            'obj_bboxes_ours': obj_boxes
        })

def top_recall_phrase(nre, all_tuple_path):
    # Load data
    with open(all_tuple_path, 'rb') as f:
        all_tuple = pickle.load(f)

    relation_triplet = all_tuple[0]
    relation_score = all_tuple[1]
    sub_boxes = all_tuple[2]
    obj_boxes = all_tuple[3]
    gt_sub_boxes = all_tuple[4]
    gt_obj_boxes = all_tuple[5]
    gt_labels = all_tuple[6]
    
    # select top nre
    num_pos_tuple = 0
    for i in range(len(relation_triplet)):
        if len(relation_triplet[i]) > nre:
            relation_triplet[i] = relation_triplet[i][:nre]
            relation_score[i] = relation_score[i][:nre]
            obj_boxes[i] = obj_boxes[i][:nre]
            sub_boxes[i] = sub_boxes[i][:nre]
        #else:
            # Do nothing
        num_pos_tuple += len(gt_labels[i])


    gt_thr = 0.5;
    tp_list = [[] for _ in range(len(relation_triplet))]
    fp_list = [[] for _ in range(len(relation_triplet))]
    for i in range(len(relation_triplet)):
        # ours
        our_labels = relation_triplet[i]
        if our_labels == []:
            continue

        # Get out union box
        our_sub_boxes = sub_boxes[i]
        our_obj_boxes = obj_boxes[i]
        our_box_entity = np.concatenate((
            np.minimum(our_sub_boxes[:, :2], our_obj_boxes[:, :2]),
            np.maximum(our_sub_boxes[:, 2:], our_obj_boxes[:, 2:])
        ), axis=1)

        # ground truth
        gt_tuple = gt_labels[i]
        num_gt_tuple = len(gt_tuple)
        gt_detected = np.zeros(num_gt_tuple)
        gt_box_entity = np.concatenate((
            np.minimum(gt_sub_boxes[i][:, :2], gt_obj_boxes[i][:, :2]),
            np.maximum(gt_sub_boxes[i][:, 2:], gt_obj_boxes[i][:, 2:])
        ), axis=1)

        num_our_tuple = len(our_labels)
        tp = [0 for _ in range(num_our_tuple)]
        fp = [0 for _ in range(num_our_tuple)]

        for j in range(num_our_tuple):
            bbox = our_box_entity[j, :]
            ovmax = float('-inf')
            kmax = -1

            for k in range(num_gt_tuple):
                if gt_detected[k] > 0:
                    continue
                # May have several predicate means the same thing
                # in this context
                for p in range(len(gt_tuple[k][2])):
                    gt_sub = gt_tuple[k][0]
                    gt_obj = gt_tuple[k][1]
                    gt_pred = gt_tuple[k][2][p]
                    if gt_sub != our_labels[j][0] or gt_obj != our_labels[j][1] \
                       or gt_pred != our_labels[j][2]:
                        continue
                    # else: maybe correct, check for box

                    gt_bbox = gt_box_entity[k, :]
                    overlap = get_overlap(gt_bbox, bbox)

                    # makes sure that this object is detected accordin to its individual threshold
                    if overlap >= gt_thr and overlap > ovmax:
                        ovmax = overlap
                        kmax = k
                        # Don't care about other predicate (must be wrong)
                        break
            if kmax > -1:
                tp[j] = 1
                gt_detected[kmax] = 1
            else:
                fp[j] = 1

        tp_list[i] = tp[:]
        fp_list[i] = fp[:]
    
    tp_all = []
    fp_all = []
    confs = []
    for i in range(len(tp_list)):
        tp_all = tp_all + tp_list[i]
        fp_all = fp_all + fp_list[i]
        if isinstance(relation_score[i], np.ndarray):
            confs = confs + relation_score[i].tolist()
        else:
            confs = confs + relation_score[i]

    ind = np.argsort(confs)
    tp_all = np.array(tp_all)[ind]
    fp_all = np.array(fp_all)[ind]

    tp = np.cumsum(tp_all)
    fp = np.cumsum(fp_all)
    recall = tp / float(num_pos_tuple)
    top_recall = recall[-1]
    return top_recall


def top_recall_relationship(nre, all_tuple_path):
    with open(all_tuple_path, 'rb') as f:
        all_tuple = pickle.load(f)
    relation_triplet = all_tuple[0]
    relation_score = all_tuple[1]
    sub_boxes = all_tuple[2]
    obj_boxes = all_tuple[3]
    gt_sub_boxes = all_tuple[4]
    gt_obj_boxes = all_tuple[5]
    gt_labels = all_tuple[6]
    # select top nre
    num_pos_tuple = 0
    for i in range(len(relation_triplet)):
        if len(relation_triplet[i]) > nre:
            relation_triplet[i] = relation_triplet[i][:nre]
            relation_score[i] = relation_score[i][:nre]
            obj_boxes[i] = obj_boxes[i][:nre]
            sub_boxes[i] = sub_boxes[i][:nre]
        #else:
            # Do nothing
        num_pos_tuple += len(gt_labels[i])

    gt_thr = 0.5;
    tp_list = [[] for _ in range(len(relation_triplet))]
    fp_list = [[] for _ in range(len(relation_triplet))]
    for i in range(len(relation_triplet)):
        # ground truth
        gt_tuple = gt_labels[i]
        num_gt_tuple = len(gt_tuple)
        gt_detected = np.zeros(num_gt_tuple)
        gt_sub_box = gt_sub_boxes[i]
        gt_obj_box = gt_obj_boxes[i]

        # ours
        our_labels = relation_triplet[i]
        if our_labels == []:
            continue

        our_sub_box = sub_boxes[i]
        our_obj_box = obj_boxes[i]

        num_our_tuple = len(our_labels)
        tp = [0 for _ in range(num_our_tuple)]
        fp = [0 for _ in range(num_our_tuple)]

        for j in range(num_our_tuple):
            sub_bbox = our_sub_box[j, :]
            obj_bbox = our_obj_box[j, :]
            ovmax = float('-inf')
            kmax = -1

            for k in range(num_gt_tuple):
                if gt_detected[k] > 0:
                    continue
                # May have several predicate means the same thing
                # in this context
                for p in range(len(gt_tuple[k][2])):
                    gt_sub = gt_tuple[k][0]
                    gt_obj = gt_tuple[k][1]
                    gt_pred = gt_tuple[k][2][p]
                    if gt_sub != our_labels[j][0] or gt_obj != our_labels[j][1] \
                       or gt_pred != our_labels[j][2]:
                        continue
                    # else: maybe correct, check for box

                    gt_sub_bbox = gt_sub_box[k, :]
                    gt_obj_bbox = gt_obj_box[k, :]
                    overlap = get_sub_obj_overlap(gt_sub_bbox, gt_obj_bbox, sub_bbox, obj_bbox)

                    # makes sure that this object is detected accordin to its individual threshold
                    if overlap >= gt_thr and overlap > ovmax:
                        ovmax = overlap
                        kmax = k
                        # Don't care about other predicate (must be wrong)
                        break
            if kmax > -1:
                tp[j] = 1
                gt_detected[kmax] = 1
            else:
                fp[j] = 1

        tp_list[i] = tp[:]
        fp_list[i] = fp[:]
    
    tp_all = []
    fp_all = []
    confs = []
    for i in range(len(tp_list)):
        tp_all = tp_all + tp_list[i]
        fp_all = fp_all + fp_list[i]
        if isinstance(relation_score[i], np.ndarray):
            confs = confs + relation_score[i].tolist()
        else:
            confs = confs + relation_score[i]

    ind = np.argsort(confs)
    tp_all = np.array(tp_all)[ind]
    fp_all = np.array(fp_all)[ind]

    tp = np.cumsum(tp_all)
    fp = np.cumsum(fp_all)
    recall = tp / float(num_pos_tuple)
    top_recall = recall[-1]
    return top_recall

