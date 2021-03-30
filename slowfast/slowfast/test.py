import pickle
from fvcore.common.file_io import PathManager
import torch

path_to_checkpoint = '/srv/beegfs02/scratch/da_action/data/models_pretrained/SLOWFAST_32x2_R101_50_50.pkl'

with PathManager.open(path_to_checkpoint, "rb") as f:
	checkpoint = torch.load(f, map_location="cpu")


model_state_dict_3d = (model.module.state_dict() if data_parallel else model.state_dict())
checkpoint["model_state"] = normal_to_sub_bn(checkpoint["model_state"], model_state_dict_3d)
"""
if inflation:
	# Try to inflate the model.
	inflated_model_dict = inflate_weight(checkpoint["model_state"], model_state_dict_3d)
	ms.load_state_dict(inflated_model_dict, strict=False)
else:
	if clear_name_pattern:
		for item in clear_name_pattern:
			model_state_dict_new = OrderedDict()
			for k in checkpoint["model_state"]:
				if item in k:
					k_re = k.replace(item, "")
					model_state_dict_new[k_re] = checkpoint["model_state"][k]
					logger.info("renaming: {} -> {}".format(k, k_re))
				else:
					model_state_dict_new[k] = checkpoint["model_state"][k]
			checkpoint["model_state"] = model_state_dict_new
	pre_train_dict = checkpoint["model_state"]
	model_dict = ms.state_dict()
	# Match pre-trained weights that have same shape as current model.
	pre_train_dict_match = {k: v for k, v in pre_train_dict.items() if
		k in model_dict and v.size() == model_dict[k].size()}
	# Weights that do not have match from the pre-trained model.
	not_load_layers = [k for k in model_dict.keys() if k not in pre_train_dict_match.keys()]
	# Log weights that are not loaded with the pre-trained weights.
	if not_load_layers:
		for k in not_load_layers:
			logger.info("Network weights {} not loaded.".format(k))
	# Load pre-trained weights.
	ms.load_state_dict(pre_train_dict_match, strict=False)
	epoch = -1

"""