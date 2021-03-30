import tensorflow as tf
import PIL
import numpy as np
from datetime import datetime




# create an empty image
#img = tf.zeros([1, 3, 3, 3])

img = tf.keras.preprocessing.image.load_img(
    '/srv/beegfs02/scratch/da_action/data/ava/frames/-5KQ66BBWC4/-5KQ66BBWC4_004001.jpg', grayscale=False,
    color_mode='rgb', target_size=None,
    interpolation='nearest')
"""
img = tf.keras.preprocessing.image.img_to_array(
    img, data_format=None, dtype=None
)
"""

img = np.expand_dims(img, axis=0)

print(img.shape)

# draw a box around the image

box = np.array([0.1, 0.1, 0.9, 0.9])

boxes = box.reshape([1, 1, 4])

colors = np.array([[124.5, 252.0, 0.0]])

img = tf.image.draw_bounding_boxes(images=img, boxes=boxes, colors=colors, name=None)


img = tf.keras.preprocessing.image.array_to_img(
    img[0,:,:,:], data_format=None, scale=True, dtype=None
)
img = np.expand_dims(img, axis=0)

logdir = "/srv/beegfs02/scratch/da_action/data/output/tb_visualization/" + datetime.now().strftime("%Y%m%d-%H%M%S")

file_writer = tf.summary.create_file_writer(logdir)
with file_writer.as_default():
    tf.summary.image("Example", img, step=0)








