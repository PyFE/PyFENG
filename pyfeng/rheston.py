from .opt_abc import OptABC
from .params import RoughHestonParams


class RoughHestonABC(RoughHestonParams, OptABC):
    """
    Rough Heston model Abstract class
    """
