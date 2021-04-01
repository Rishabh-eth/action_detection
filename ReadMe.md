# Domain Adaptation in Action Detection using Video Clip Order Prediction

This work has been built upon the Action Detection method proposed in SlowFast Networks for Video Recognition (https://github.com/facebookresearch/SlowFast). 
The Action Detection pipeline is described in the documentation provided by the offical implementation by FAIR in [README.md](slowfast/README.md). This README describes the domain aptation and self-supervision based features added on top of the existing FAIR pipeline. The discussion includes instructions to train/validate a model as well as create a demo on a given video clip input.

### Setting up the Environment

This project used Conda environment. The used environment.yml files can be found in the folder [environment](environment). You are encouraged to use the most recent version. Please refer to [README.md](slowfast/README.md) for the additional steps necessary to setup the pipeline e.g. configure Detectron etc.

### Key files
Initialize the following files and their paths depending on the workstation where they are used at:
 - "Main Code" [slowfast](slowfast): This folder contains the code for Action Detection along with the Domain Adaptation component.
 - "Train files" (batch job) [train_slowfast.sh](slowfast/train_slowfast.sh): This file is used to schedule a training job to SLURM, it first activates the conda environment and runs the file [run_net.py](tools/run_net.py) based on the training configuration files provided (see "Configuration files")
 - "Test files" [test_slowfast.sh](slowfast/test_slowfast.sh): Same as "Train files", the configuration files are changed to "Test" mode accordingly.
 - "Configuration files" [experiments](experiments): The directory of used configuration files for carrying out experiments, use one of these in "Train files".
 - "Output files" `/srv/beegfs-benderdata/scratch/da_action/data/output_rishi/ `: In "Configuration files", the path of Output files is mentioned. This location is used to save checkpoints, confusion matrices, tensorboard logs etc.
 - "Dataset" `/srv/beegfs02/scratch/da_action/ava/ and /srv/beegfs02/scratch/da_action/data/kinetics700/`: There are different data splits under the paths mentioned. These splits vary in terms of number of smaples used in the training and validation stages, however the most commonly used split throughout the project is 10_500_100, corresponding to 10 classes, 500 samples/class for training and 100 samples/class for validation. The corresponding directories "frame_lists_10_500_100" and "annotations_10_500_100" can be found for both the AVA and Kinetics path mentioned in the beginning. For furhter details, please refer to "Configuration files"
 - "Log files" [train_logs](slowfast/train_logs), [test_logs](slowfast/test_logs): Directory to store the log files produced by scheduled files on SLURM. These are useful for monitoring the training updates and viewing class-wise performance during testing.  
 
 
### How to run an experiment
1. Activate conda environemnt
2. Make sure you are in the directory [slowfast](slowfast)
3. Schedule job on SLURM:
 - For training: use the command  `sbatch --output=train_logs/%j.out --gres=gpu:1 --mem=30G train_slowfast.sh`. If this doesn't work, try with `sbatch --output=train_logs/%j.out --gres=gpu:1 --mem=30G --constraint='geforce_gtx_titan_x' train_slowfast.sh`. The former command worked for most of the experiments. Remember to provide the right configuration file in  `train_slowfast.sh`
 - For testing: use the command  `sbatch --output=test_logs/%j.out --gres=gpu:1 --mem=30G test_slowfast.sh' (make approprite adjustments in the other command if necessary as mentioned above). As again, remember to provide the configuration file that has 'test' mode activated.
 - For a debug job: use the command `srun --time 300 --partition=gpu.debug --gres=gpu:1 --nodelist='biwirender09' --pty bash -i` to get a debug node and then use the command `./train_slowfast.sh` to run the experiment. No log directory is needed to be provided here as the debug mode does not generate any log file, the corresponging outputs are printed on the terminal. 


## Domain Adaptation training
The hyperparameters used in action detection expeiments are provided by the PySlowFast implementation in [defaults.py](slowfast/slowfast/config/defaults.py). Most of the hyperparameters used in configuration files are taken from this file however, a few more are added to enable domain adaptation experiments as discussed below:

### Parameters to enable Domain/Adaptation and/or Self-supervised training 

 - _C.DA.ENABLE = False -> combined training on the DA data 
 - _C.DA.AUX = False -> auxiliary task training on the DA data
 - _C.DA.AUX_TEST = False -> auxiliary task validation on the DA data (next section)
 
Always only one of these tasks is carried out, i.e. setting DA.AUX to true disables DA.ENABLE, setting DA.AUX_TEST to true disables both others.
Thus, to train only the auxilliary task aka the self-supervision task say for eg. to perform transfer learning experiments, keep _C.DA.AUX = True and rest False.
And to perform Multi-task training, keep _C.DA.ENABLE = True and the rest False.

### Dataloader for Auxilliary task
The domain adaptation dataset is loaded from the same kind of annotation files as the ava or kinetics dataset. The tag 'DA' is used in raw auxiliary task trainings for transfer learning or in the combined training for multi-task setup.
 - _C.DA.CLASSES = 0 -> required in the Da dataloader ([ava_dataset.py](slowfast//slowfast/datasets/ava_dataset.py) , class Da) to know the number of classes to be predicted for video clip order predcition. It has mostly been set to 6.
 - _C.DA.DATASET = "da" -> defines which dataloader has to be used for the following data
 - _C.DA.FRAME_DIR = "path to frame directory" eg. "/srv/beegfs-benderdata/scratch/da_action/data/ava/frames/"
 - _C.DA.LABEL_MAP_FILE = "path to action list" eg. "ava_action_list_v2.2.pbtxt"
 - _C.DA.FRAME_LIST_DIR = "path to frame list directory" eg. "/srv/beegfs-benderdata/scratch/da_action/data/ava/frame_lists_10_500_100/"
 - _C.DA.ANNOTATION_DIR = "path to annotations directory" eg. "/srv/beegfs-benderdata/scratch/da_action/data/ava/annotations_10_500_100/"
 - _C.DA.TRAIN_LISTS = ["train.csv"]
 - _C.DA.TEST_LISTS = ["val.csv"]
 - _C.DA.EXCLUSION_FILE = "kinetics_val_excluded_timestamps_v2.1.csv" or "ava_val_excluded_timestamps_v2.2.csv"
 - _C.DA.DETECTION_SCORE_THRESH = 0.8
 - _C.DA.TRAIN_GT_BOX_LISTS = "kinetics_train_v2.1.csv" or "ava_train_v2.2.csv"
 - _C.DA.TEST_GT_BOX_LISTS = []
 - _C.DA.TRAIN_PREDICT_BOX_LISTS = ["kinetics_train_predicted_boxes.csv"]
 - _C.DA.TEST_PREDICT_BOX_LISTS = ["kinetics_val_predicted_boxes.csv"] or ["ava_val_predicted_boxes.csv"]
 - _C.DA.GROUNDTRUTH_FILE = "kinetics_val_v2.1.csv" or "ava_val_v2.2.csv "
 - _C.DA.FULL_TEST_ON_VAL = True

### Optimisation
 - _C.DA.LR_FACTOR = 10.0 -> the factor by which the auxiliary task learning rate shall be smaller than the action detection learning rate
 - _C.DA.F_CONFUSION = -1 -> after every DA.F_CONFUSION-th batch a confusion matrix is logged to tensorboard , usually it is kept 100 0r 500

## Auxiliary task dataloader (use this to create the data shuffler for the video clip order prediction part)
The auxiliary task data-loader (in [ava_dataset.py](slowfast/slowfast/datasets/ava_dataset.py)) inherits from the standard Ava data-loader declared in the same file.
`def __getitem__(self, idx)` is modified to create the self-supervised frames and the corresponding labels. 

The data loader for clip order prediction begins at line 720. 
Line 724-726 creates three clips out of the given input clip to the usual pipeline.
Line 741 generates a random class label
Line 748-777 shuffle the three clips according the label generated among one of the 6 order classes.
The rest of the pipeline is similar to the dataloader for the action detection case. 
Readers are expected to integrate this dataloader with their pipeline (some objective other than action detection in this work) if they intend to use the self-supervision part.


## Joint training for Multi-task training 
The combined training procedure is implemented in the file [train_da.py](slowfast/tools/train_da.py). Two data-loaders are used in parallel. The source length of the source domain data-loader determines the number of iterations. If the target domain data-loader (da) contains more batches, the last batches are cut. If it contains less batches, the data-loader is duplicated and once the first one is empty, batches from the second one are taken.
The optimisation chain makes use of two optimizers and iteratively does backpropagation on a batch of source domain data and a batch of target domain data.

## Model manipulation
The standard process in the files `train_net`, `train_aux` and `train_da` is to randomly initialise the model and then overwrite the weights with the given checkpoint model.

### Freezing layers
 - _C.MODEL.FREEZE_TO = 152 -> This parameter freezes all model weights until layer 152. This will set the requires_grad for low level layers to false. Consequently, these layers won’t be affected by the later training and always keep the weights from the loaded model. The particular index corresponds to freezing Conv 1 and Res 2 of SlowFast. You can determine the number by printing out layers or viewing the model summary in any training logs of a model. Some commonly used values are 605, 392 and 152 for the experiments in this project. 

### Random initialisation of weights
 - _C.MODEL.LAST_PRE_TRAIN = "s3_fuse.bn.num_batches_tracked" -> This parameter is used to only overwrite the randomly initialised weights until a certain layer, this is used for experiments when a few layers were kept randomly initialised while the rest were still intialsed with FB pretrained weights. Determine the name of the layer by reading printed layer names. Note that the parameter makes sense to be used only with the correct value for the parameter _C.MODEL.FREEZE_TO. Some commonly used combinations in this work include:

   - FREEZE_TO: 605 , LAST_PRE_TRAIN: "s4_fuse.bn.num_batches_tracked"
   - FREEZE_TO: 392 , LAST_PRE_TRAIN: "s4.pathway0_res22.branch2.c_bn.num_batches_tracked"
   - FREEZE_TO: 152 , LAST_PRE_TRAIN: "s3_fuse.bn.num_batches_tracked"

Use one of these combinations for carrying out experiments with random initialisation of few layers.

## Training a Transfer Learning model

1. Train the auxilliary task:  
- Set _C.DA.AUX = True. 
- Provide the necessary file pahts in under the tag 'DA'. 
- Provide TRAIN.CHECKPOINT_FILE_PATH as initialisation for learning weights during the pretext task training. 
- Keep TRAIN.ENABLE= False (this will prevent the training of the main task as desired). 
- Provide appropriate values to the rest of the tags, especially being careful with SOLVER, MODEL and OUTPUT_DIR. 
- See an examle of [auxilliary config file](experiments/aux_10_500_100_v7).

2. Train the main task: 
- Keep DA.ENABLE= False, DA.AUX= False, TRAIN.ENABLE= True. 
- Set TRAIN.CHECKPOINT_FILE_PATH to the checkpoint from the auxilliary traininng you wish to use as weight initialisation. 
- Provide the required data paths under the tag 'AVA'. 
- Set the rest of the tags and especially the OUTPUT_DIR.

3. Test the trained model from step 2 on any dataset using a test config file. To convert a file to test mode, keep tags 'DA' and 'TRAIN' to False (to prevent any training). Provide the test data paths in the tag 'AVA'. Set the tag TEST.ENABLE=True, provide the path to model checkpoint which needs to be tested in TEST.CHECKPOINT_FILE_PATH. Note that we don't need the auxilliary paths while testing and hence the tag DA is redundant. Thus the test config files used in pyslowfast to test the action detection pipeline (without self-supervision) can also be used. See more on this in the section on CD/WD experiments below.

## Training a Multi-task model

1. Both main and aux tasks are to be trained now.
- So DA.ENABLE= True and TRAIN.ENABLE= True. 
- Keep DA.AUX= False as it will only train the aux part alone which we don't want. 
- Provide the path to dataset used for training the self-supervision part in the corresponding tags under 'DA'. 
- Provide the path to dataset used for training the main task under the tag 'AVA'. 
- Set DA.F_CONFUSION and DA.LR_FACTOR. Give the weight initialisation in TRAIN.CHECKPOINT_FILE_PATH. 
- Set TRAIN.CHECKPOINT_EPOCH_RESET = True (necessary for updates of weights in multi-task setting). 
- Set the rest of the tags accordingly. 
- See  [joint config file](experiments/joint_10_500_100_v11) as an example.

2. Test the model from step 1: Follow the same procedure as mentioned for Transfer Learning model. 


## Cross-Domain/ Within domain experiments

Once the models are trained and checkpoints stored, you can test a model trained on one dataset on another dataset making proper adjustments to the configuration files. Some examples are provided in [example_scripts](slowfast/example_scripts). Note that the files starting with the prefix 'test' are to be used while testing. The files starting with 'train' are examples for training a SlowFast model for action detection without domain adaptation. For domain adaptation training, you should refer to [experiments](experiments)

### Example scripts for Domain Adaption experiments

Some example scripts are provided in [experiments](experiments). There is no strict naming convention however, a few consistent ones are discussed below. The prefix of a filename relates to the task the config file is used for:

 - `ex`: AVA action classification training
 - `ki`: Kinetics action classification training
 - `aux`: Auxiliiary task training (clip order prediction)
 - `joint`: Multi-task training

 The rest of the name of the file post prefix tells about the experiment setting. For eg. aux_10_500_100_v1 means 10 classes are predicted in action detection (first suffix), 500 samples per class were used for training (second suffix), 100 samples per class were used for validation (third suffix) and version number (fourth suffix). However, the reader is encouraged to follow their own nomenclature. 

## Downloading Dataset
### AVA
Following files are used to download the AVA dataset. Adjust file paths  and dataset version numbers (2.1, 2.2) to download the propoer annotations and videos:
1. [01_ava_download](dataset_download/ava/01_ava_download) -> Download all the AVA videos
2. [02_ava_cut](dataset_download/ava/02_ava_cut) -> Cut the AVA videos to 15 mins
3. [03_ava_frames](dataset_download/ava/03_ava_frames) -> Extract the frames
4. [04_ava_annotations](dataset_download/ava/04_ava_annotations) -> Download annotation files
5. [05_ava_frame_lists](dataset_download/ava/05_frame_lists) -> Download the frame lists
6. [06_ava_person_boxes](dataset_download/ava/06_person_boxes) -> Download person boxes provided by FAIR

For more details, refer to (https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/DATASET.md) based on which these scripts are written.

### Kinetics700
The repository [KineticX-Downloader](dataset_download/KineticX-Downloader) is used to download Kinetics700. The repository is included in this project and has been cloned from [chi0tzp](https://github.com/chi0tzp/KineticX-Downloader).
Then the original ava_kinetics_v1_0 annotations from the official [AVA](http://research.google.com/ava/) homepage are to be downloaded into a folder with the name kinetics (this is STEP 0).
Kinetics frame extraction needs a lot of time. Therefore it is only done for certain classes. The following steps need to be carried out for both the training and validation sets (SPLIT):

1. [01_csv_steer_keyframe.py](dataset_download/kinetics/01_csv_steer_keyframe.py) -> do reduction steps on `kinetics_SPLIT_v1.0.csv`: 1) remove unlabelled key-frames, 2) only consider in scope classes, 3) exclude later key-frame given two annotated key-frames -> obtained .csv file is used in next step
2. [02_csv_parser_keyframe.py](dataset_download/kinetics/02_csv_parser_keyframe.py) -> extract frames for 6s around the key-frame, does ignore samples that are too close to the beginning/end of the video or if video does not exist
3. [03_create_gt_files.py](dataset_download/kinetics/03_create_gt_files.py) -> iterates through `kinetics_SPLIT_v1.0.csv` and checks whether the path for the wanted key-frames exist -> then creates the annotation files on the basis of the initial annotation file (`kinetics_SPLIT_v1.0.csv)
4. [04_path_existence.py](dataset_download/kinetics/04_path_existence.py) -> check whether the key-frame paths really exist (produce it for both splits to obtain `kinetics_SPLIT_failed_path.csv`)
5. [05_frame_lists.py](dataset_download/kinetics/05_frame_lists.py) -> produce the frame lists that map from the video name and the frame_id to the frame paths
6. [06_predict_bboxes.py](dataset_download/kinetics/06_predict_bboxes.py) -> predict the bounding boxes for the kinetics key-frames (use the [06_bboxes_eval.py](dataset_download/kinetics/06_bboxes_eval.py) file to see whether predictions for all key-frames have been made)
7. [07_replace_timestamp.py](dataset_download/kinetics/07_replace_timestamp.py) -> time stamp replacement is necessary to be able to use AVA data-loaders afterwards. This replacement is to be done for `kinetics_SPLIT_v2.1.csv` and `kinetics_SPLIT_predicted_boxes.csv` (maybe do a backup of the input files prior to starting this step)

Below structure depicts the target folder structure. The [STEP] after each line shows in which step the respective file/folder is created.

```
kinetics700
|_ ava-kinetics_v1_0
|  |_ ...
|  |_ kinetics_train_v1.0.csv [0]
|  |_ kinetics_val_v1.0.csv [0]
|  |_ ...
|_ csv_steer
|  |_ 10_cl_train_frames.csv [1]
|  |_ 10_cl_val_frames.csv [1]
|  |_ 10_cl_train_frames_considered.csv [2]
|  |_ 10_cl_train_frames_failed.csv [2]
|  |_ 10_cl_val_frames_considered.csv [2]
|  |_ 10_cl_val_frames_failed.csv [2]
|_ frames [2]
|  |_ [video name 0]
|  |  |_ [video name 0]_000001.jpg
|  |  |_ [video name 0]_000002.jpg
|  |  |_ ...
|  |_ [video name 1]
|     |_ [video name 1]_000001.jpg
|     |_ [video name 1]_000002.jpg
|     |_ ...
|_ frame_lists
|  |_ train.csv [5]
|  |_ val.csv [5]
|_ annotations
   |_ ava_action_list_v2.2.pbtxt [copy from ava annotations]
   |_ ava_included_timestamps_v2.2.txt [copy from ava annotations]
   |_ kinetics_train_excluded_timestamps_v2.1.csv [empty csv]
   |_ kinetics_train_v2.1.csv [3] [7]
   |_ kinetics_train_failed_path.csv [4]
   |_ kinetics_train_predicted_boxes.csv [6] [7]
   |_ kinetics_val_excluded_timestamps_v2.1.csv  [empty csv]
   |_ kinetics_val_v2.1.csv [3] [7]
   |_ kinetics_val_failed_path.csv [4]
   |_ kinetics_val_predicted_boxes.csv [6] [7]
```

## Dataset Reduction
The following dataset reduction scripts work if for the respective dataset the original annotations are given. It is ideal for experiments where the spatial localization is not part of the problem, i.e. inference happens on the ground-truth bounding boxes.
For doing spatial localization experiments as well it is recommended to enhance the reduction procedure to have a background class (such that always all boxes are part of the ground-truth annotation file).
Each scripts allows to specify the wanted classes as well as a maximum number of training/validation annotations. It will then sample this number of annotations from the original files and produce the annotation and frame_list folder following the name convention. 

### 80 class setting
Use these files to create a class and annotation reduction where labels follow the 80 class ids.
 - [create_files_80num_ava.py](dataset_reduction/create_files_80num_ava.py): reduce ava annotations
 - [create_files_80num_kinetics.py](dataset_reduction/create_files_80num_kinetics.py): reduce kinetics annotations

### XX class setting
Use these files to create a class and annotation reduction where labels are mapped to the reduced number of classes (consecutive ids).
 - [create_files_ava.py](dataset_reduction/create_files_ava.py): reduce ava annotations
 - [create_files_kinetics.py](dataset_reduction/create_files_kinetics.py): reduce kinetics annotations
 
### Ground-truth bounding boxes for inference
In order to do inference on ground-truth bounding boxes, the true labels need to be removed from the files called `DATASET_SPLIT_vX.X.csv`. The following script removes the label and adds a confidence value of 1 to the line entry.
 - [remove_pred_add_conf.py](dataset_reduction/remove_pred_add_conf.py): the files shall be added to the folder structure as follows for both datasets (AVA and Kinetics)
```
AVA
.
.
.
|_ annotations
   |_ ...
   |_ ava_train_predicted_boxes_gt.csv
   |_ ava_val_predicted_boxes_gt.csv
   |_ ...
```
The term "predicted" is used since these are boxes used for inference (even though they are actually ground-truth)

## Linear Interpolated Demo
Not that the initial Facebook Demo is not suitable for interpolated boxes since it works with multiple processes.
The here documented solution uses the folder system for communication between processes as a hotfix.

The following changes in the pipeline must be done to get linearly interpolated bounding boxes:

Befor running the demo do the following steps:
 - [async_predictor.py](slowfast/slowfast/visualization/async_predictor.py) -> set the hotfix parameter in line 290 to True
 - [utils.py](slowfast/slowfast/visualization/utils.py) -> set the hotfix parameter in line 356 to True
 - [utils.py](slowfast/slowfast/visualization/utils.py) -> set the hotfix parameter in line 378 to True
 - [video_visualizer.py](slowfast/slowfast/visualization/video_visualizer.py) -> set the hotfix parameter in line 623 to True

After running a demo:
 - [hotfix](slowfast/slowfast/hotfix) -> delete all files in this folder, this needs to be done everytime, the folder is used to handle information flow between workers

The following conditions must be fulfilled for the interpolated demo to work:
 - the actors must be in the screen since the beginning
 - movement in the beginning or the end should not change abruptly, since otherwise the continuation of the bounding box needs to be estimated
 - video length should not be longer than approximately 20s

Find an example demo file in:
 - [SLOWFAST_32x2_R101_50_50.yaml](visualizations_presentation/SLOWFAST_32x2_R101_50_50.yaml)
