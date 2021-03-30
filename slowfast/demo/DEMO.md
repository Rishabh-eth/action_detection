In the demo mode one can do inference for any video.

Bounding boxes appear and vanish smoothly, however do not follow the person. In order to linearly interpolate the bounding boxes between the single steps one needs to activate the following lines in the code:
1) slowfast/slowfast/visualization/async_predictor.py - line: 290
2) slowfast/slowfast/visualization/video_visualizer.py - line: 623
3) slowfast/slowfast/visualization/utils.py - line: 357, 379

