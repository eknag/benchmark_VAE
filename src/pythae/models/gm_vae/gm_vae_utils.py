import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..base.base_utils import ModelOutput
from ...models.nn.base_architectures import BaseEncoder

class MixtureGenerator(nn.Module):

    def __init__(self, model_config: dict):

        #BaseEncoder.__init__(self, model_config)

        nn.Module.__init__(self)

       
        self.upper_layer = nn.Sequential(
                        nn.Linear(np.prod(model_config.w_prior_latent_dim), 512),
                        nn.ReLU()
                    )

        self.means_layers = nn.ModuleList()
        self.log_covariances_layers = nn.ModuleList()

        for i in range(model_config.number_components):
            self.means_layers.append(
                nn.Linear(512, model_config.latent_dim)
            )
            self.log_covariances_layers.append(
                nn.Linear(512, model_config.latent_dim)
            )

        self.model_config = model_config

    def forward(self, w: torch.tensor):

        gmm_means = torch.zeros(
            (
                w.shape[0],
                self.model_config.latent_dim,
                self.model_config.number_components
            )
        ).to(w.device)

        gmm_log_covariances = torch.zeros(
            (
                w.shape[0],
                self.model_config.latent_dim,
                self.model_config.number_components
            )
        ).to(w.device)

        output = ModelOutput()

        hidden_out = self.upper_layer(w)

        for i in range(self.model_config.number_components):
            
            gmm_means[:, :, i] = self.means_layers[i](hidden_out)
            gmm_log_covariances[:, :, i] = self.log_covariances_layers[i](hidden_out)

        output['gmm_means'] = gmm_means
        output['gmm_log_covariances'] = gmm_log_covariances

        return output