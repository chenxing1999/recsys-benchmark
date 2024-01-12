import torch

from src.models.deepfm import DeepFM

inp_checkpoint = "checkpoints/deepfm/opt-embed-80.pth"
out_checkpoint = "checkpoints/deepfm/opt-80-vanilla.pth"
out_checkpoint = "checkpoints/deepfm/opt-80-csr.pth"

# 2 mode: Opt and OptRetrain
# in both two mode, require one version for sparse_W, one not
is_retrain = True
is_sparse = True

checkpoint = torch.load(inp_checkpoint, map_location="cpu")
model = DeepFM.load(checkpoint, strict=False)

if is_retrain:
    model.embedding._mask = checkpoint["state_dict"]["embedding._mask"]
    with torch.no_grad():
        weight = model.embedding.get_weight()

    new_state = {
        name: p
        for name, p in checkpoint["state_dict"].items()
        if "embedding." not in name
    }

    if is_sparse:
        sparse_w = weight.to_sparse_csr()

        new_state["embedding.sparse_w"] = sparse_w

        checkpoint["state_dict"] = new_state
        checkpoint["model_config"]["embedding_config"] = {"name": "pruned-sparse-csr"}
        torch.save(checkpoint, out_checkpoint)

        # return
        exit()

    new_state["embedding._emb_module.weight"] = weight
    checkpoint["state_dict"] = new_state
    checkpoint["model_config"]["embedding_config"] = {"name": "vanilla"}

    torch.save(checkpoint, out_checkpoint)


# test
torch.load(out_checkpoint)
