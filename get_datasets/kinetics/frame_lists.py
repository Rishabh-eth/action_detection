"""
Given the groundtruth annotation files, this file can be used to create the frame_lists
"""

#https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=dq9GY37ml1kr

import torch
import numpy as np
import os, json, cv2, random
import pandas as pd
import csv



kinetics = '/srv/beegfs02/scratch/da_action/data/kinetics700/frames'

source = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/test.csv'
#target_file = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_val_predicted_boxes.csv'
#target_file = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_train_predicted_boxes.csv'
target_file = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/bla.csv'



# Start of the actual script

# load a list of the video names where we need to create predictions
source = pd.read_csv(source, header=None, sep=',')
source = source.drop([1,2,3,4,5,6], axis=1)
source = source.drop_duplicates()
print('after dropping duplicates')



# open the target file
f = open(target_file, 'w')
with f:
    writer = csv.writer(f, delimiter=' ', doublequote=False, escapechar='\\', quotechar='%')
    writer.writerow(['original_vido_id', 'video_id', 'frame_id', 'path', 'labels'])
    video_id = 0

    for index, row in source.iterrows():
        # write predictions to a .csv file
        # data to be written row-wise in csv fil
        list = []
        for i in range(180):
            list.append([row[0], video_id, i, row[0] + '/' + row[0] + '_' + str(i + 1).zfill(6) + '.jpg', '\"\"'])
        for row in list:
            writer.writerow(row)

        video_id += 1






















