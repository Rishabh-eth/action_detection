"""
This file creates the groundtruth files given the extracted frames in a folder and the wanted video name and time
instance of the keyframes in interests. The script also prints statistics about the found frames.
"""
import os
import pandas as pd

frames = '/srv/beegfs02/scratch/da_action/data/kinetics700/frames'
input_csv = '/srv/beegfs02/scratch/da_action/data/kinetics700/csv_steer/10_cl_val_frames.csv'

annot_orig = '/srv/beegfs02/scratch/da_action/data/kinetics700/ava-kinetics/ava_kinetics_v1_0/kinetics_val_v1.0.csv'
target_annotations = 'kjkj/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_val_v2.1.csv'

# read the .csv file with the wanted frames into a dataframe
kinetics_all = pd.read_csv(input_csv, header=None, sep=',')

exists = []
false_counter = 0

# iterate through all rows of dataframe and check whether a certain filename exists
for index, row in kinetics_all.iterrows():
    path_bool = os.path.isdir(os.path.join(frames, row[0]))
    if path_bool:
        exists.append(True)
    else:
        exists.append(False)
        false_counter += 1

kinetics_all.insert(2, 2, exists)
print('length of actually wanted keyframes: ', len(kinetics_all[1]))


# drop rows where path did not exist
kinetics_all = kinetics_all.drop(kinetics_all[kinetics_all[2] == False].index)
# drop column 2
kinetics_all.drop(2, inplace=True, axis=1)
print('length of keyframes after reduction due to extraction error: ', len(kinetics_all[1]))


# gather other information to create the annotation files
annot_orig = pd.read_csv(annot_orig, header=None, sep=',')
print('length of original annotations: ', len(annot_orig[0]))
annot_orig = annot_orig.dropna(axis=0, how="any")
print('length of original annotations without nan: ', len(annot_orig[0]))

list = []

for index, row in kinetics_all.iterrows():
    list.append(annot_orig.loc[(annot_orig[0] == row[0]) & (annot_orig[1] == row[1])])

new_annot = pd.concat(list)


"""
# do the timestamp conversion 01WKY1mn5FI
helper = new_annot[1]

for index, value in helper.iteritems():
    add = int(30 * (value % 1))
    helper[index] = int(90 + add)

new_annot[1] = helper.astype(int)
"""
new_annot[6] = new_annot[6].astype(int)

print('row length of final annotation file: ', len(new_annot[1]))


new_annot.to_csv(target_annotations, header=None, index=False, float_format='%.3f')


