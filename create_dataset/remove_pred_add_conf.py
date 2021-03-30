"""
Script to remove column 7 (somehow predicted class for a certain frame) from the file ava_val_v2.2.csv
This is needed, when one one wants to do inference on the ground truth bounding boxes
Additionally, a bounding box confidence of 1 is added in column 7 since the box is precisely known.
- saves new file with the name ava_val_predicted_boxes_gt.csv at same location
"""
import pandas as pd
import os

folder = '/srv/beegfs02/scratch/da_action/data/ava/annotations_10_500_100'
file = 'ava_train_v2.2.csv'


# load ava_train_predicted_boxes.csv even if it is probably not used in actual project
ava_train_predicted_boxes = pd.read_csv(os.path.join(folder, file), header=None)

ava_train_predicted_boxes[6] = ''
ava_train_predicted_boxes[7] = 1

# save to a file
ava_train_predicted_boxes.to_csv(os.path.join(folder, 'ava_train_predicted_boxes_gt.csv'), header=None,
                      index=False, float_format='%.3f')
