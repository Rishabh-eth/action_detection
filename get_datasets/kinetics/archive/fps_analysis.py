import cv2
import os

directory = "/srv/beegfs02/scratch/da_action/data/kinetics700/videos/validate"
directory = "/srv/beegfs02/scratch/da_action/data/ava/videos"

for filename in os.listdir(directory):
    cap=cv2.VideoCapture(os.path.join(directory, filename))


    fps = cap.get(cv2.CAP_PROP_FPS)

    print(fps)