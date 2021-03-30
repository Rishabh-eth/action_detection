"""
This file shall create the file which steers the creation of the frame folders later
Given the initial annotation file, the following steps are executed
1) exclude unlabeled samples
2) reduce to in scope classes since producing the frames for all classes takes too long
3) exclude later frame, if two frames are annotated per video
4) reduce duplicated records due to multiple bounding boxes or several actions per bounding box
"""
import os
import pandas as pd

orig_annot = 'sdf/srv/beegfs02/scratch/da_action/data/kinetics700/ava-kinetics/ava_kinetics_v1_0/kinetics_val_v1.0.csv'
target_steer = 'sdf/srv/beegfs02/scratch/da_action/data/kinetics700/csv_steer/10_cl_val_frames.csv'


classes = [80,79,74,11,17,14,59,1,8,49]

kinetics_all = pd.read_csv(orig_annot, header=None, sep='\n')
kinetics_all = kinetics_all[0].str.split(',', expand=True)
print('initial length:', len(kinetics_all[0]))

# 1) Drop None rows
kinetics_all = kinetics_all.dropna(axis=0, how="any")
print('after none:', len(kinetics_all[0]))

kinetics_all[6] = kinetics_all[6].astype(int)

# 2) Exclude not considered classes
kinetics_all = kinetics_all.loc[kinetics_all[6].isin(classes)]
print('after classes:', len(kinetics_all[0]))

# 3) Exclude duplicate frames per video
# 4) reduce duplicate records due to multiple bounding boxes or several actions per bounding box
# Create helper that does not contain duplicates considering first two columns
kinetics_helper = kinetics_all.drop_duplicates(subset=[0,1], keep='first')
kinetics_helper = kinetics_helper.drop_duplicates(subset=[0], keep='first')

# Save the file needed to produce the kinetics frames
kinetics_frames = kinetics_helper[[0,1]]
print('number of frames to save', len(kinetics_frames[0]))
kinetics_frames.to_csv(target_steer, header=None, index=False, float_format='%.3f')


# Produce the corresponding ground truth annotation file containing the annotations for all the saved frames
"""
frames = []
for index, row in kinetics_frames.iterrows():
    frames.append(kinetics_all.loc[(kinetics_all[0] == row[0]) & (kinetics_all[1] == row[1])])

kinetics_gt = pd.concat(frames)
print('gt_list', len(kinetics_gt[0]))

ava_train_pred.to_csv(os.path.join(annot_dir, 'ava_train_predicted_boxes.csv'), header=None,
             #         index=False, float_format='%.3f')

"""
