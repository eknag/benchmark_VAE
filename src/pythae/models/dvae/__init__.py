"""This module is the implementation of the BetaVAE proposed in
(https://openreview.net/pdf?id=Sy2fzU9gl).
This model adds a new parameter to the VAE loss function balancing the weight of the 
reconstruction term and KL term.


Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    :nosignatures:
"""

from .dvae_config import DVAEConfig
from .dvae_model import DVAE

__all__ = ["DVAE", "DVAEConfig"]
