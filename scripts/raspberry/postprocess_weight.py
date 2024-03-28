import torch

from src.models.deepfm import DeepFM

inp_checkpoint = "checkpoints/deepfm/pep-80.pth"
out_checkpoint = "checkpoints/deepfm/pep-80-csr.pth"


checkpoint = torch.load(inp_checkpoint, map_location="cpu")
model = DeepFM.load(checkpoint)

with torch.no_grad():
    weight = model.embedding.get_weight()

sparse_w = weight.to_sparse_csr()

new_state = {
    name: p for name, p in checkpoint["state_dict"].items() if "embedding." not in name
}
new_state["embedding.sparse_w"] = sparse_w


checkpoint["state_dict"] = new_state
checkpoint["model_config"]["embedding_config"] = {"name": "pruned-sparse-csr"}
# checkpoint["embedding.sparse_w"] = sparse_w
torch.save(checkpoint, out_checkpoint)


# test
torch.load(out_checkpoint)
