"""
This file shall be used to test the alternative dataloader that produces samples for the auxiliary loss
"""

#python rot_test.py --cfg /home/sieberl/SA2020/pyslowfast/experiments/da_10_5_2_v3/SLOWFAST_32x2_R101_50_50.yaml

import sys
sys.path.insert(0, "/home/sieberl/SA2020/pyslowfast/slowfast")

from slowfast.datasets import loader
import torch
import numpy as np
import random
from slowfast.visualization.video_visualizer import VideoVisualizer
from slowfast.utils.parser import load_config, parse_args

args = parse_args()
cfg = load_config(args)


# Hyperparameters
NR = 8 # how many batches shall be put down the drain from the lodader

# Set random seed from configs.  https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
np.random.seed(cfg.RNG_SEED)
torch.manual_seed(cfg.RNG_SEED)
torch.cuda.manual_seed_all(cfg.RNG_SEED)
torch.cuda.manual_seed(cfg.RNG_SEED)
random.seed(cfg.RNG_SEED)
#torch.set_deterministic(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

da_train_loader = loader.construct_loader_da(cfg, "da_train")
#da_train_loader = loader.construct_loader(cfg, "train")
da_train_iterator = iter(da_train_loader)

number = NR
for i in range(number):
    (inputs, labels, _, meta) = next(da_train_iterator)

print('rotation labels for the current batch: ', labels)
print('metadata for the current batch: ', meta)

num_classes = 4
class_name_json = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations_10_5_2/class_names.json'
video_vis = VideoVisualizer(
            num_classes=num_classes,
            class_names_path=class_name_json,
            top_k=0,
            thres=0.7,
            lower_thres=0.3,
            common_class_names=None,
            colormap="Reds",
            mode="top-k",
        )


img = inputs[1][3,:,16,:,:]

preds = labels[10:,:]
bboxes = meta['boxes'][10:,1:]



for idx in range(img.shape[0]):
    img[idx] = img[idx] * cfg.DATA.STD[idx]
    img[idx] = img[idx] + cfg.DATA.MEAN[idx]

img = img*255
img = img.type(torch.IntTensor)
# bring it to integer values




img = img.permute(1,2,0)
output = video_vis.draw_one_frame(frame=img, preds=preds, bboxes=bboxes, alpha=0.5, text_alpha=0.7,
    ground_truth=False, )

from PIL import Image as im

data = im.fromarray(output)
data.save('/home/sieberl/SA2020/rand3.jpg')

