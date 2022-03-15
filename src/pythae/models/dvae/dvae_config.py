from pydantic.dataclasses import dataclass

from ...models import VAEConfig


@dataclass
class DVAEConfig(VAEConfig):
    r"""
    :math:`\beta`-VAE model config config class

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        sigma (float): The curruption factor, ie the standard deviation of the Gaussian noise. Default: 0.025
    """

    sigma: float = 0.025
