"""This file iterates through a given 'frames' folder and checks whether a certain amount of images is present and
prints folder names that contain less images than the threshold"""
import os, os.path, csv

# Frame directory
frames = '/srv/beegfs02/scratch/da_action/data/kinetics700/frames'

f = open(os.path.join(frames, 'check_frame_presence.csv'), 'w')
with f:
    writer = csv.writer(f)
    # iterate through all folders in the frame directory
    for root, subdirectories, files in os.walk(frames):
        for subdirectory in subdirectories:
            path = os.path.join(root, subdirectory)
            length = len([name for name in os.listdir(path)])
            if length < 150:
                print(path, ': ', length)
                writer.writerow([path, length])








