"""
Script to iterate through first two columns of csv and confirm that the wanted frames actually exist
Should be done after the creation of a ground truth file in order to confirm that all the frames are present
"""
#https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=dq9GY37ml1kr

import torch
import numpy as np
import os, json, cv2, random
import pandas as pd
import csv



kinetics = '/srv/beegfs02/scratch/da_action/data/kinetics700/frames'

source = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_val_v2.1.csv'

failed_file = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/bla.csv'

# Function declaration
# Start of the actual script

# load a list of the video names where we need to create predictions
source = pd.read_csv(source, header=None, sep=',')
source = source.drop([2,3,4,5,6], axis=1)
source = source.drop_duplicates()
print('after dropping duplicates')

failed_paths = []

for index, row in source.iterrows():

    # calculate the frame id:
    add = int(30 * (row[1] % 1))
    frame_id = int(90 + add)
    video_name = row[0]
    frame_finder = video_name + '/' + video_name + '_' + str(frame_id).zfill(6) + '.jpg'
    path = os.path.join(kinetics, frame_finder)

    if not os.path.isfile(path):
        failed_paths.append(path)
        print('failed: ', frame_finder)


f = open(failed_file, 'w')
with f:
    writer = csv.writer(f)
    for row in failed_paths:
        writer.writerow(row)




















