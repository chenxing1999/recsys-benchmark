"""Helper script to convert LightGCN's FBTT to Torch TTRec for CPU inference"""

import copy
import sys

from src.models import load_graph_model
from src.models.embeddings.tensortrain_embeddings import TTRecTorch

checkpoint_path = sys.argv[1]
out_checkpoint_path = sys.argv[2]


model = load_graph_model(checkpoint_path)

new_model = copy.deepcopy(model)

for name in ["user_emb_table", "item_emb_table"]:
    emb = model.get_module(name)
    new_emb = TTRecTorch.init_from_fbtt(emb)
    setattr(new_model, name, new_emb)
