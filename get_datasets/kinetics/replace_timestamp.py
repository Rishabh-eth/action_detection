"""
In order to be closer at the AVA case the timestamp in the annotation files are changed to an integer value
0903 -> since this will result in frame *0090.jpg being loaded as keyframe which is always the case
"""
import os
import pandas as pd

frames = '/srv/beegfs02/scratch/da_action/data/kinetics700/frames'
input_csv = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations_prep/kinetics_val_predicted_boxes.csv'

target_annotations = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_val_predicted_boxes.csv'

# read the .csv file with the wanted frames into a dataframe
kinetics_all = pd.read_csv(input_csv, header=None, sep=',')

print(kinetics_all)
#kinetics_all[1] = kinetics_all[1].replace([*],'0904', regex=True)
kinetics_all[1] = '0903'
"""
# iterate through all rows of dataframe and replace the second column
for index, row in kinetics_all.iterrows():
    row[1] = '0903'
"""

print(kinetics_all)


kinetics_all.to_csv(target_annotations, header=None, index=False, float_format='%.3f')


