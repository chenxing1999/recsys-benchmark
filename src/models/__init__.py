import os
from typing import Dict, Type, Union

import torch
from loguru import logger

from src.models.lightgcn import LightGCN, SingleLightGCN

from .base import IGraphBaseCore
from .dcn import DCN_Mix, DCNv2
from .deepfm import DeepFM
from .hccf import HCCFModelCore


def get_graph_model(
    num_users: int,
    num_items: int,
    model_config: Dict,
) -> IGraphBaseCore:
    """Wrapper to get model from config"""

    # pop name for ** trick
    name = model_config.pop("name")

    name_to_cls: Dict[str, Type[IGraphBaseCore]] = {
        "lightgcn": LightGCN,
        "single-lightgcn": SingleLightGCN,
        "hccf": HCCFModelCore,
    }
    assert name in name_to_cls
    cls = name_to_cls[name]
    model = cls(num_users, num_items, **model_config)

    model_config["name"] = name

    return model


def load_graph_model(checkpoint_path: str, strict=True) -> IGraphBaseCore:
    """Load checkpoint to correct model"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    num_users = checkpoint["num_users"]
    num_items = checkpoint["num_items"]
    model_config = checkpoint["model_config"]

    model = get_graph_model(num_users, num_items, model_config)
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    return model


def save_cf_emb_checkpoint(
    model: Union[LightGCN, SingleLightGCN],
    checkpoint_dir: str,
    name: str = "target",
):
    """Wrapper to save checkpoint embedding to a folder
    in the belowing format:
        {checkpoint_dir}/{field_name}/{name}.pth
    """

    for field_name, emb in model.get_embs():
        field_dir = os.path.join(checkpoint_dir, field_name)
        os.makedirs(field_dir, exist_ok=True)

        path = os.path.join(field_dir, f"{name}.pth")
        torch.save(emb.state_dict(), path)


def get_ctr_model(field_dims, model_config: dict):
    name = "deepfm"
    if "name" in model_config:
        name = model_config.pop("name")

    if name == "deepfm":
        return DeepFM(field_dims, **model_config)
    elif name == "dcn_mix":
        compile_model = True
        if "compile_model" in model_config:
            compile_model = model_config.pop("compile_model")
        model = DCN_Mix(field_dims, **model_config)
        model_config["compile_model"] = compile_model
        if compile_model:
            return torch.compile(model)
        return model
    elif name == "dcn":
        model = DCNv2(field_dims, **model_config)
        return model
    else:
        raise NotImplementedError()


def load_ctr_model(model_config, checkpoint, strict=True, *, empty_embedding=False):
    name = "deepfm"
    if "name" in model_config:
        name = model_config.pop("name")
    if name == "deepfm":
        return DeepFM.load(checkpoint, strict, empty_embedding=empty_embedding)
    elif name == "dcn_mix":
        model = DCN_Mix.load(checkpoint, strict, empty_embedding=empty_embedding)
        return model
    else:
        raise NotImplementedError()


def save_ctr_checkpoint(
    model: Union[DeepFM, DCN_Mix],
    checkpoint_dir: str,
    name: str = "target",
):
    """Wrapper to save checkpoint embedding to a folder
    in the belowing format:
        {checkpoint_dir}/deepfm/{name}.pth
    """

    emb = model.embedding
    if hasattr(model, "_orig_mod"):
        logger.debug("Save _orig_mod")
        model = model._orig_mod

    if isinstance(model, DeepFM):
        field_name = "deepfm"
    elif isinstance(model, DCN_Mix):
        field_name = "dcn"
    else:
        raise NotImplementedError(f"Not supported for {model.__class__=}")

    field_dir = os.path.join(checkpoint_dir, field_name)
    os.makedirs(field_dir, exist_ok=True)

    path = os.path.join(field_dir, f"{name}.pth")
    torch.save(emb.state_dict(), path)
