import torch
from einops import einsum


def bpr_loss(user_embs, pos_embs, neg_embs):
    """Bayesian Personalized Ranking loss re-implementation

    Args:
        user_embs: N x D -
        pos_embs:  N x D
        neg_embs:  N x D  - neg_embs[i] is the negative correspondence of pos_embs[i]

    Returns:
        single loss item
    """
    y_hat_pos = einsum(user_embs, pos_embs, "i j, i j -> i")
    y_hat_neg = einsum(user_embs, neg_embs, "i j, i j -> i")

    loss = -torch.log(torch.sigmoid(y_hat_pos - y_hat_neg)).mean()

    return loss
