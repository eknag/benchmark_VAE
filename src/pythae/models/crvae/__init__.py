"""Implementation of a Vanilla Autoencoder model.

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    :nosignatures:

"""

from .crvae_model import CRVAE
from .crvae_config import CRVAEConfig

__all__ = ["CRVAE", "CRVAEConfig"]
