import torch
from einops import einsum
from torch.nn import functional as F


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

    loss = -F.logsigmoid(y_hat_pos - y_hat_neg).mean()

    return loss


def info_nce(
    view1: torch.Tensor,
    view2: torch.Tensor,
    temperature: float = 1,
    b_cos: bool = True,
) -> torch.Tensor:
    """
    Args:
        view1: (torch.Tensor - N x D)
        view2: (torch.Tensor - N x D)
        temperature: float
        b_cos (bool)

    Note:
    https://github.com/Coder-Yu/SELFRec/blob/d3be574e27e1e7b931b63949dbcc73b8e5ac53b9/util/loss_torch.py#L35C1-L43C31
    This implementation is compared with the above implementation
    """
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()


def bpr_loss_multi(user_embs, pos_embs, neg_embs):
    """Bayesian Personalized Ranking loss re-implementation for multiple negative items

    Args:
        user_embs: N x D -
        pos_embs:  N x D
        neg_embs:  N x K x D  -
            neg_embs[i] is the negative correspondence of pos_embs[i]

    Returns:
        single loss item
    """
    y_hat_pos = einsum(user_embs, pos_embs, "i j, i j -> i").unsqueeze(-1)
    y_hat_neg = einsum(user_embs, neg_embs, "i j, i k j -> i k")

    # loss = -F.logsigmoid(y_hat_pos - y_hat_neg).mean()
    loss = -F.logsigmoid(y_hat_pos - y_hat_neg).sum() / len(user_embs)

    return loss
