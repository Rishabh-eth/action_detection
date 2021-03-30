#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import sys
#sys.path.append("/home/sieberl/SA2020/pyslowfast/slowfast")
#sys.path.append("/home/sieberl/SA2020/pyslowfast/detectron2_repo")
#print(sys.path)

# Luca Sieber
paths = sys.path
for p in paths:
    print (p)
print ('========')
sys.path.remove('/usr/itetnas04/data-scratch-01/risingh/data/slowfast')
sys.path.append("/usr/itetnas04/data-scratch-01/risingh/data/videorder/slowfast/")
paths = sys.path
for p in paths:
    print (p)

"""Wrapper to train and test a video classification model."""

from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from demo_net import demo
from test_net import test
from train_net import train
from train_da import train_da
from train_aux import train_aux
from test_aux import test_aux
from visualization import visualize

import torch #Luca

def main():
    """
    Main function to spawn the train and test process.
    """

    args = parse_args()
    cfg = load_config(args)

    # Perform training on the auxiliary loss
    if cfg.DA.AUX_TEST:
        launch_job(cfg=cfg, init_method=args.init_method, func=test_aux)
        cfg.DA.ENABLE = False
        cfg.DA.AUX = False

    if cfg.DA.AUX:
        launch_job(cfg=cfg, init_method=args.init_method, func=train_aux)
        cfg.DA.ENABLE = False

    # Perform domain adaptive training
    if cfg.DA.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train_da)

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

    # Perform model visualization.
    if cfg.TENSORBOARD.ENABLE and (
        cfg.TENSORBOARD.MODEL_VIS.ENABLE
        or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    ):
        launch_job(cfg=cfg, init_method=args.init_method, func=visualize)

    # Run demo.
    if cfg.DEMO.ENABLE:
        demo(cfg)



if __name__ == "__main__":
    main()
