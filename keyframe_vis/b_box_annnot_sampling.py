# add new folder to path since other modules are used
import sys
sys.path.insert(0, "/home/sieberl/SA2020/pyslowfast/slowfast")



from functions import visualize_samples

SAMPLES = 20
TENSORBOARD = "/srv/beegfs02/scratch/da_action/data/output/ki_10_500_100_v1/"

detections_file = '/home/sieberl/SA2020/pyslowfast/experiments/ki_10_500_100_v1' \
                  '/15_detections_latest.csv'
gt_file = '/home/sieberl/SA2020/pyslowfast/experiments/ki_10_500_100_v1/15_groundtruth_latest' \
          '.csv'

class_name_json = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations_10_500_100/class_names.json'
NUM_CLASSES = 10

image_folder = '/srv/beegfs02/scratch/da_action/data/kinetics700/frames'
iteration = 16



visualize_samples(samples=SAMPLES,
                  tensorboard=TENSORBOARD,
                  detections_file=detections_file,
                  gt_file=gt_file,
                  class_name_json=class_name_json,
                  num_classes=NUM_CLASSES,
                  image_folder=image_folder,
                  iteration=iteration)






