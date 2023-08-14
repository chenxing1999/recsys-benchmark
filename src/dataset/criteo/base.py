from abc import ABC


class ICriteoDatset(ABC):
    """Small Interface for Criteo Dataset"""

    def pop_info(self):
        ...

    def describe(self):
        ...
