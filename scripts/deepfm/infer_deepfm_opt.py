import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from src.dataset.criteo import CriteoDataset, CriteoIterDataset
from src.dataset.criteo.criteo_torchfm import CriteoDataset as CriteoFMData
from src.models.deepfm import DeepFM
from src.trainer.deepfm import validate_epoch

# Load state dict and data
config_path = "configs/deepfm/opt_embed_debug.yaml"
checkpoint_path = "checkpoints/deepfm_checkpoint_opt.pth"
with open(config_path) as fin:
    config = yaml.safe_load(fin)

checkpoint = torch.load(checkpoint_path, map_location="cpu")

model = DeepFM(checkpoint["field_dims"], **config["model"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# load mask_d
print("load mask d")
initial_path = "checkpoints/deepfm/opt/initial.pth"
initial_checkpoint = torch.load(initial_path, map_location="cpu")
mask = initial_checkpoint["mask"]["mask_d"]
model.embedding.get_weight(mask)
model.to("cuda")
model.embedding._cur_weight = model.embedding._cur_weight.to("cuda")
# print(model.embedding.get_sparsity())
# exit()


# Load data
def get_dataset_cls(loader_config) -> str:
    num_workers = loader_config.get("num_workers", 0)
    shuffle = loader_config.get("shuffle", False)

    if "train_test_info" in loader_config["dataset"]:
        return "torchfm"
    if num_workers == 0 and not shuffle:
        return "iter"
    else:
        return "normal"


NAME_TO_DATASET_CLS = {
    "iter": CriteoIterDataset,
    "normal": CriteoDataset,
    "torchfm": CriteoFMData,
}


def load_and_infer(config, name="val_dataloader"):
    val_dataloader_config = config[name]
    val_dataset_config = val_dataloader_config["dataset"]
    val_dataset_cls = get_dataset_cls(val_dataloader_config)
    logger.info(f"Val dataset type: {val_dataset_cls}")
    val_dataset_cls = NAME_TO_DATASET_CLS[val_dataset_cls]

    val_dataset = val_dataset_cls(**val_dataset_config)
    val_dataloader = DataLoader(
        val_dataset,
        val_dataloader_config["batch_size"],
        shuffle=False,
        num_workers=val_dataloader_config["num_workers"],
    )

    print(f"Run Validate {name}")
    print(validate_epoch(val_dataloader, model, "cuda"))


load_and_infer(config)
load_and_infer(config, "test_dataloader")
