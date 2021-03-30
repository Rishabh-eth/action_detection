"""
Script to check whether the reduced duplicate files of both gt and predicted time instances are the same
"""

#https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=dq9GY37ml1kr

import torch, torchvision
import numpy as np
import os, json, cv2, random
import pandas as pd
import csv



kinetics = '/srv/beegfs02/scratch/da_action/data/kinetics700/frames'
#source = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_val_v2.1.csv'
#source = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_train_v2.1.csv'
source = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations_10_500_100/kinetics_val_predicted_boxes.csv'
#target_file = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_val_predicted_boxes.csv'
#target_file = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_train_predicted_boxes.csv'
target_file = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_train_predicted_boxes.csv'
#failed_file = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_val_failed_path.csv'
#failed_file = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_train_failed_path.csv'
#failed_file = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_val_failed_path_srun.csv'

# Start of the actual script

# load a list of the video names where we need to create predictions
source = pd.read_csv(source, header=None, sep=',')
print(source)
source = source.drop([2,3,4,5,6,7], axis=1)
source = source.drop_duplicates()
print('number of keyframes in source: ', len(source[0]))
"""
counter = 0

for index, row in source.iterrows():
    if (row[1] % 1) > 0.1:
        counter += 1

print('counter:', counter)




target = pd.read_csv(target_file, header=None, sep=',')
print(target)
target = target.drop([2,3,4,5,6,7], axis=1)
target = target.drop_duplicates()
print('number of keyframes in target: ', len(target[0]))
"""



















