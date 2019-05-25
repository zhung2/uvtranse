addpath('evaluation');
%load('relationship_det_result.mat', 'rlp_labels_ours', ...
%    'rlp_confs_ours', 'sub_bboxes_ours', 'obj_bboxes_ours');
%{
load('union_epoch_150_vgg_relationship_det_result.mat', 'rlp_labels_ours', ...
    'rlp_confs_ours', 'sub_bboxes_ours', 'obj_bboxes_ours');
%}
load('test_rel_preds_1_2_3_4_5_6_7_8_9_10_11.mat', 'rel_labels', ...
    'rel_confs', 'sub_bboxes', 'obj_bboxes');
rlp_confs_ours = rel_confs;
rlp_labels_ours = rel_labels;
sub_bboxes_ours = sub_bboxes;
obj_bboxes_ours = obj_bboxes;
%

fprintf('#######  Top recall results  ####### \n');
recall100P = top_recall_Phrase(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
recall50P = top_recall_Phrase(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours); 
fprintf('Phrase Det. R@100: %0.2f \n', 100*recall100P);
fprintf('Phrase Det. R@50: %0.2f \n', 100*recall50P);

recall100R = top_recall_Relationship(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
recall50R = top_recall_Relationship(50, rlp_confs_ours, rlp_labels_ours, ...
sub_bboxes_ours, obj_bboxes_ours);
fprintf('Relationship Det. R@100: %0.2f \n', 100*recall100R);
fprintf('Relationship Det. R@50: %0.2f \n', 100*recall50R);

fprintf('\n');
fprintf('#######  Zero-shot results  ####### \n');
zeroShot100P = zeroShot_top_recall_Phrase(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
zeroShot50P = zeroShot_top_recall_Phrase(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
fprintf('zero-shot Phrase Det. R@100: %0.2f \n', 100*zeroShot100P);
fprintf('zero-shot Phrase Det. R@50: %0.2f \n', 100*zeroShot50P);

zeroShot100R = zeroShot_top_recall_Relationship(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
zeroShot50R = zeroShot_top_recall_Relationship(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
fprintf('zero-shot Relationship Det. R@100: %0.2f \n', 100*zeroShot100R);
fprintf('zero-shot Relationship Det. R@50: %0.2f \n', 100*zeroShot50R);