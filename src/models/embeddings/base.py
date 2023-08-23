from torch import nn


class IEmbedding(nn.Module):
    """
    Note: Not all weight is initialized from the same distribution
    as I found out that the Xavier uniform despite converge slower
    but usually get higher accuracy.
    """

    def get_weight(self):
        ...


class VanillaEmbedding(nn.Embedding, IEmbedding):
    """Wrapper for default PyTorch Embedding"""

    def __init__(self, *args, initializer="xavier", **kwargs):
        super().__init__(*args, **kwargs)

        if initializer == "xavier":
            nn.init.xavier_uniform_(self.weight)
        else:
            nn.init.normal_(self.weight, std=0.1)

    def get_weight(self):
        return self.weight
