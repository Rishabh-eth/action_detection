"""
Script to remove column 7 (somehow predicted class for a certain frame) from the file ava_train_predicted_boxes.csv
There should not be any evidence for a class in the box predictions file
- saves new file with appendix "all" at same location
"""
import pandas as pd
import os

folder = '/srv/beegfs02/scratch/da_action/data/ava/annotations_5_200_40'
file = 'ava_train_predicted_boxes.csv'


# load ava_train_predicted_boxes.csv even if it is probably not used in actual project
ava_train_predicted_boxes = pd.read_csv(os.path.join(folder, file), header=None)

ava_train_predicted_boxes[6] = ''


# save to a file
ava_train_predicted_boxes.to_csv(os.path.join(folder, 'ava_train_predicted_boxes_removed.csv'), header=None,
                      index=False, float_format='%.3f')