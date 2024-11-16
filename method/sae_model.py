# Reference: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/model.py
from typing import Callable, Any
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


def LN(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std

class MultiLayerAutoEncoder(nn.Module):
    """Sparse autoencoder

    Implements:
        latents = activation(encoder(x - pre_bias) + latent_bias)
        recons = decoder(latents) + pre_bias
    """

    def __init__(
        self, 
        n_latents: int, 
        n_inputs: int,
        layer_num: int,
        activation: Callable = nn.ReLU(), 
        tied: bool = False,
        normalize: bool = False,
        project_dim: int = None,
        dataset_level_norm: bool = False
    ) -> None:
        """
        :param n_latents: dimension of the autoencoder latent
        :param n_inputs: dimensionality of the original data (e.g residual stream, number of MLP hidden units)
        :param activation: activation function
        :param tied: whether to tie the encoder and decoder weights
        """
        super().__init__()

        self.project_dim = project_dim
        if self.project_dim is not None:
            self.projection: nn.Module = nn.Linear(n_latents, project_dim)
        
        self.encoder_head: nn.Module = nn.Linear(n_inputs, n_latents, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.encoder_tail = nn.ModuleList([nn.Linear(n_latents, n_latents, bias=False) for _ in range(layer_num-1)])
        
        #nn.init.kaiming_uniform_(self.encoder.weight)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        self.tail_bias = nn.ParameterList([nn.Parameter(torch.zeros(n_latents)) for _ in range(layer_num-1)])
        
        self.activation = activation
        if tied:
            self.decoder_head: nn.Linear | TiedTranspose = TiedTranspose(self.encoder_head)
            self.decoder_tail = nn.ModuleList([TiedTranspose(self.encoder_tail[i]) for i in range(layer_num-2, -1, -1)])
        else:
            self.decoder_head = nn.Linear(n_latents, n_inputs, bias=False)
            self.decoder_tail = nn.ModuleList([nn.Linear(n_latents, n_latents, bias=False) for _ in range(layer_num-1)])
        self.normalize = normalize
        self.dataset_level_norm = dataset_level_norm
        self.register_buffer('data_mean', torch.zeros(n_inputs, dtype=torch.float))
        self.register_buffer('data_mean_counter', torch.tensor(0, dtype=torch.long))

        self.stats_last_nonzero: torch.Tensor
        self.latents_activation_frequency: torch.Tensor
        self.latents_mean_square: torch.Tensor
        self.register_buffer("stats_last_nonzero", torch.zeros(n_latents, dtype=torch.long))
        self.register_buffer(
            "latents_activation_frequency", torch.ones(n_latents, dtype=torch.float)
        )
        self.register_buffer("latents_mean_square", torch.zeros(n_latents, dtype=torch.float))

    def encode_pre_act(self, x: torch.Tensor, latent_slice: slice = slice(None)) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, n_inputs])
        :param latent_slice: slice of latents to compute
            Example: latent_slice = slice(0, 10) to compute only the first 10 latents.
        :return: autoencoder latents before activation (shape: [batch, n_latents])
        """
        x = x - self.pre_bias
        latents_pre_act = F.linear(
            x, self.encoder.weight[latent_slice], self.latent_bias[latent_slice]
        )
        return latents_pre_act

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.normalize:
            return x, dict()
        x, mu, std = LN(x)
        return x, dict(mu=mu, std=std)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return: autoencoder latents (shape: [batch, n_latents])
        """
        x = x.float()
        x, info = self.preprocess(x)
        return self.activation(self.encode_pre_act(x)), info

    def decode(self, latents: torch.Tensor, info: dict[str, Any] = None) -> torch.Tensor:
        """
        :param latents: autoencoder latents (shape: [batch, n_latents])
        :return: reconstructed data (shape: [batch, n_inputs])
        """
        ret = self.decoder(latents) + self.pre_bias
        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return:  autoencoder latents pre activation (shape: [batch, n_latents])
                  autoencoder latents (shape: [batch, n_latents])
                  reconstructed data (shape: [batch, n_inputs])
        """
        x = x.float()
        if self.dataset_level_norm is True:
            x = x - (self.data_mean)
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        recons = self.decode(latents, info)

        # set all indices of self.stats_last_nonzero where (latents != 0) to 0
        self.stats_last_nonzero *= (latents == 0).all(dim=0).long()
        self.stats_last_nonzero += 1

        return latents_pre_act, latents, recons
    
    def projection_forward(self, latents: torch.Tensor) -> torch.Tensor:
        #latents, _ = self.encode(x)
        return torch.nn.functional.sigmoid(self.projection(latents))
    
    def update_data_mean(self, x: torch.Tensor):
        assert len(x.shape) == 2
        x = x.to(self.device)
        
        batch_size = x.shape[0]
        batch_mean = x.mean(dim=0)

        delta = batch_mean - self.data_mean
        total_n = self.data_mean_counter + batch_size

        # Update mean
        self.data_mean += delta * batch_size / total_n

        # Update count
        self.data_mean_counter = total_n

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> "Autoencoder":
        n_latents, d_model = state_dict["encoder.weight"].shape
        tied = state_dict.pop("tied", False)
        project_dim = state_dict.pop("project_dim", None)
        # Retrieve activation
        activation_class_name = state_dict.pop("activation", "ReLU")
        activation_class = ACTIVATIONS_CLASSES.get(activation_class_name, nn.ReLU)
        #normalize = activation_class_name == "TopK"  # NOTE: hacky way to determine if normalization is enabled
        normalize = state_dict.pop("normalize", False)
        activation_state_dict = state_dict.pop("activation_state_dict", {})
        for key in list(state_dict.keys()):
            matches = re.findall(r'\bactivation\.\w+', key)
            if len(matches) > 0:
                state_dict.pop(key)
        if hasattr(activation_class, "from_state_dict"):
            activation = activation_class.from_state_dict(
                activation_state_dict, strict=strict
            )
        else:
            activation = activation_class()
            if hasattr(activation, "load_state_dict"):
                activation.load_state_dict(activation_state_dict, strict=strict)

        autoencoder = cls(n_latents, 
                          d_model, 
                          activation=activation, 
                          normalize=normalize, 
                          tied=tied, 
                          project_dim=project_dim)
        # Load remaining state dict
        autoencoder.load_state_dict(state_dict, strict=strict)
        return autoencoder

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd[prefix + "activation"] = self.activation.__class__.__name__
        if hasattr(self.activation, "state_dict"):
            sd[prefix + "activation_state_dict"] = self.activation.state_dict()
        sd["normalize"] = self.normalize
        sd["tied"] = isinstance(self.decoder, TiedTranspose)
        sd["project_dim"] = self.project_dim
        return sd

    @property
    def device(self):
        return next(self.parameters()).device
    
class Autoencoder(nn.Module):
    """Sparse autoencoder

    Implements:
        latents = activation(encoder(x - pre_bias) + latent_bias)
        recons = decoder(latents) + pre_bias
    """

    def __init__(
        self, 
        n_latents: int, 
        n_inputs: int,
        activation: Callable = nn.ReLU(), 
        tied: bool = False,
        normalize: bool = False,
        project_dim: int = None,
        dataset_level_norm: bool = False
    ) -> None:
        """
        :param n_latents: dimension of the autoencoder latent
        :param n_inputs: dimensionality of the original data (e.g residual stream, number of MLP hidden units)
        :param activation: activation function
        :param tied: whether to tie the encoder and decoder weights
        """
        super().__init__()

        self.pre_bias = nn.Parameter(torch.zeros(n_inputs))
        self.project_dim = project_dim
        if self.project_dim is not None:
            self.projection: nn.Module = nn.Linear(n_latents, project_dim)
        
        self.encoder: nn.Module = nn.Linear(n_inputs, n_latents, bias=False)
        #nn.init.kaiming_uniform_(self.encoder.weight)
        self.latent_bias = nn.Parameter(torch.zeros(n_latents))
        
        self.activation = activation
        if tied:
            self.decoder: nn.Linear | TiedTranspose = TiedTranspose(self.encoder)
        else:
            self.decoder = nn.Linear(n_latents, n_inputs, bias=False)
        self.normalize = normalize
        self.dataset_level_norm = dataset_level_norm
        self.register_buffer('data_mean', torch.zeros(n_inputs, dtype=torch.float))
        self.register_buffer('data_mean_counter', torch.tensor(0, dtype=torch.long))

        self.stats_last_nonzero: torch.Tensor
        self.latents_activation_frequency: torch.Tensor
        self.latents_mean_square: torch.Tensor
        self.register_buffer("stats_last_nonzero", torch.zeros(n_latents, dtype=torch.long))
        self.register_buffer(
            "latents_activation_frequency", torch.ones(n_latents, dtype=torch.float)
        )
        self.register_buffer("latents_mean_square", torch.zeros(n_latents, dtype=torch.float))

    def encode_pre_act(self, x: torch.Tensor, latent_slice: slice = slice(None)) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, n_inputs])
        :param latent_slice: slice of latents to compute
            Example: latent_slice = slice(0, 10) to compute only the first 10 latents.
        :return: autoencoder latents before activation (shape: [batch, n_latents])
        """
        x = x - self.pre_bias
        latents_pre_act = F.linear(
            x, self.encoder.weight[latent_slice], self.latent_bias[latent_slice]
        )
        return latents_pre_act

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.normalize:
            return x, dict()
        x, mu, std = LN(x)
        return x, dict(mu=mu, std=std)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return: autoencoder latents (shape: [batch, n_latents])
        """
        x = x.float()
        x, info = self.preprocess(x)
        return self.activation(self.encode_pre_act(x)), info

    def decode(self, latents: torch.Tensor, info: dict[str, Any] = None) -> torch.Tensor:
        """
        :param latents: autoencoder latents (shape: [batch, n_latents])
        :return: reconstructed data (shape: [batch, n_inputs])
        """
        ret = self.decoder(latents) + self.pre_bias
        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return:  autoencoder latents pre activation (shape: [batch, n_latents])
                  autoencoder latents (shape: [batch, n_latents])
                  reconstructed data (shape: [batch, n_inputs])
        """
        x = x.float()
        if self.dataset_level_norm is True:
            x = x - (self.data_mean)
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        recons = self.decode(latents, info)

        # set all indices of self.stats_last_nonzero where (latents != 0) to 0
        self.stats_last_nonzero *= (latents == 0).all(dim=0).long()
        self.stats_last_nonzero += 1

        return latents_pre_act, latents, recons
    
    def projection_forward(self, latents: torch.Tensor) -> torch.Tensor:
        #latents, _ = self.encode(x)
        return torch.nn.functional.sigmoid(self.projection(latents))
    
    def update_data_mean(self, x: torch.Tensor):
        assert len(x.shape) == 2
        x = x.to(self.device)
        
        batch_size = x.shape[0]
        batch_mean = x.mean(dim=0)

        delta = batch_mean - self.data_mean
        total_n = self.data_mean_counter + batch_size

        # Update mean
        self.data_mean += delta * batch_size / total_n

        # Update count
        self.data_mean_counter = total_n

    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> "Autoencoder":
        n_latents, d_model = state_dict["encoder.weight"].shape
        tied = state_dict.pop("tied", False)
        project_dim = state_dict.pop("project_dim", None)
        # Retrieve activation
        activation_class_name = state_dict.pop("activation", "ReLU")
        activation_class = ACTIVATIONS_CLASSES.get(activation_class_name, nn.ReLU)
        #normalize = activation_class_name == "TopK"  # NOTE: hacky way to determine if normalization is enabled
        normalize = state_dict.pop("normalize", False)
        activation_state_dict = state_dict.pop("activation_state_dict", {})
        for key in list(state_dict.keys()):
            matches = re.findall(r'\bactivation\.\w+', key)
            if len(matches) > 0:
                state_dict.pop(key)
        if hasattr(activation_class, "from_state_dict"):
            activation = activation_class.from_state_dict(
                activation_state_dict, strict=strict
            )
        else:
            activation = activation_class()
            if hasattr(activation, "load_state_dict"):
                activation.load_state_dict(activation_state_dict, strict=strict)

        autoencoder = cls(n_latents, 
                          d_model, 
                          activation=activation, 
                          normalize=normalize, 
                          tied=tied, 
                          project_dim=project_dim)
        # Load remaining state dict
        autoencoder.load_state_dict(state_dict, strict=strict)
        return autoencoder

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        sd[prefix + "activation"] = self.activation.__class__.__name__
        if hasattr(self.activation, "state_dict"):
            sd[prefix + "activation_state_dict"] = self.activation.state_dict()
        sd["normalize"] = self.normalize
        sd["tied"] = isinstance(self.decoder, TiedTranspose)
        sd["project_dim"] = self.project_dim
        return sd

    @property
    def device(self):
        return next(self.parameters()).device

class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias


class TopK(nn.Module):
    def __init__(self, k: int, postact_fn: Callable = nn.ReLU()) -> None:
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        # make all other values 0
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict.update({prefix + "k": self.k, prefix + "postact_fn": self.postact_fn.__class__.__name__})
        return state_dict

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor], strict: bool = True) -> "TopK":
        k = state_dict["k"]
        postact_fn = ACTIVATIONS_CLASSES[state_dict["postact_fn"]]()
        return cls(k=k, postact_fn=postact_fn)

ACTIVATIONS_CLASSES = {
    "ReLU": nn.ReLU,
    "Identity": nn.Identity,
    "TopK": TopK,
}


#================ Below are the loss functions ================

import torch


def autoencoder_loss(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
    latent_activations: torch.Tensor,
    l1_weight: float,
) -> torch.Tensor:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param l1_weight: weight of L1 loss
    :return: loss (shape: [1])
    """
    return (
        normalized_mean_squared_error(reconstruction, original_input)
        + normalized_L1_loss(latent_activations, original_input) * l1_weight
    )


def normalized_mean_squared_error(
    reconstruction: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param reconstruction: output of Autoencoder.decode (shape: [batch, n_inputs])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized mean squared error (shape: [1])
    """
    return (
        ((reconstruction - original_input) ** 2).mean(dim=1) / (original_input**2).mean(dim=1)
    ).mean()


def contrastive_loss(original: torch.Tensor, 
                     mutated: torch.Tensor,
                     model, 
                     norm=True, 
                     margin=2.0):
    num = original.shape[0]
    if norm == True:
        original = original / torch.norm(original, p=2)
        mutated = mutated / torch.norm(mutated, p=2)
    loss = max(0, margin - torch.dist(original, mutated)) / num
    return loss

def contrastive_loss_softmax(
    original: torch.Tensor,
    mutated: torch.Tensor,
    ori_index: torch.Tensor,
    mut_index: torch.Tensor,
):
    s_pos = 0
    s_neg = 0

def CCS_loss(
    original: torch.Tensor,
    mutated: torch.Tensor,
    model: Autoencoder,
):
    original_proj = model.projection_forward(original)
    mutated_proj = model.projection_forward(mutated)
    l_consis = (mutated_proj - (1 - original_proj))**2
    l_conf =   torch.min(original_proj, mutated_proj)**2
    return (l_consis.mean() + l_conf.mean()) / 2


def contrastive_pred_loss(
    original: torch.Tensor,
    mutated: torch.Tensor,
    model: Autoencoder,
):
    original_proj = model.projection_forward(original)
    mutated_proj = model.projection_forward(mutated)
    x1 = torch.nn.functional.sigmoid(original_proj)
    y1 = torch.zeros_like(x1)
    x2 = torch.nn.functional.sigmoid(mutated_proj)
    y2 = torch.zeros_like(x2)
    loss_1 = torch.nn.functional.binary_cross_entropy(x1, y1)
    loss_2 = torch.nn.functional.binary_cross_entropy(x2, y2)
    return loss_1 + loss_2

def normalized_L1_loss(
    latent_activations: torch.Tensor,
    original_input: torch.Tensor,
) -> torch.Tensor:
    """
    :param latent_activations: output of Autoencoder.encode (shape: [batch, n_latents])
    :param original_input: input of Autoencoder.encode (shape: [batch, n_inputs])
    :return: normalized L1 loss (shape: [1])
    """
    return (latent_activations.abs().sum(dim=1) / original_input.norm(dim=1)).mean()

class NaiveAutoEncoder(nn.Module):
    def __init__(self, 
                 n_latents: int,
                 n_inputs: int,
                 layer_num: int,
                 activation: Callable = nn.ReLU(inplace=True),
                 tied: bool = None,
                 normalize: bool = False,
                 project_dim: int = None,
                 dataset_level_norm: bool = False):
        super().__init__()
        self.register_buffer('data_mean', torch.zeros(n_inputs, dtype=torch.float))
        self.register_buffer('data_mean_counter', torch.tensor(0, dtype=torch.long))
        self.normalize = normalize
        self.dataset_level_norm = dataset_level_norm
        assert layer_num >= 1
        self.encoder = NaiveEncoder(n_latents, n_inputs, layer_num, activation, tied, normalize, project_dim, dataset_level_norm)
        self.decoder = NaiveDecoder(n_latents, n_inputs, layer_num, activation, tied, normalize, project_dim, dataset_level_norm)
        
    def forward(self, x):
        x = x.float()
        if self.dataset_level_norm is True:
            x = x - (self.data_mean)
        x, info = self.preprocess(x)
        latents = self.encoder(x)
        recons = self.decode(latents, info)
        return None, latents, recons

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.normalize:
            return x, dict()
        x, mu, std = LN(x)
        return x, dict(mu=mu, std=std)
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return: autoencoder latents (shape: [batch, n_latents])
        """
        x = x.float()
        x, info = self.preprocess(x)
        return self.encoder(x), info

    def decode(self, latents: torch.Tensor, info: dict[str, Any] = None) -> torch.Tensor:
        """
        :param latents: autoencoder latents (shape: [batch, n_latents])
        :return: reconstructed data (shape: [batch, n_inputs])
        """
        ret = self.decoder(latents)
        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret

    def update_data_mean(self, x: torch.Tensor):
        assert len(x.shape) == 2
        x = x.to(self.device)
        
        batch_size = x.shape[0]
        batch_mean = x.mean(dim=0)

        delta = batch_mean - self.data_mean
        total_n = self.data_mean_counter + batch_size

        # Update mean
        self.data_mean += delta * batch_size / total_n

        # Update count
        self.data_mean_counter = total_n

    @property
    def device(self):
        return next(self.parameters()).device
    
class NaiveEncoder(nn.Module):
    def __init__(self, 
                 n_latents: int,
                 n_inputs: int,
                 layer_num: int,
                 activation: Callable = nn.ReLU(inplace=True),
                 tied: bool = None,
                 noramalize: bool = False,
                 project_dim: int = None,
                 dataset_level_norm: bool = False):
        super().__init__()

        blocks = [nn.Linear(n_inputs, n_latents)]
        
        for _ in range(layer_num-1):
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.Linear(n_latents, n_latents))
        #blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

    @property
    def device(self):
        return next(self.parameters()).device

class NaiveDecoder(nn.Module):
    def __init__(self, 
                 n_latents: int,
                 n_inputs: int,
                 layer_num: int,
                 activation: Callable = nn.ReLU(inplace=True),
                 tied: bool = None,
                 noramalize: bool = False,
                 project_dim: int = None,
                 dataset_level_norm: bool = False):
        super().__init__()

        blocks = [nn.Linear(n_latents, n_latents)]
        
        for _ in range(layer_num-2):
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.Linear(n_latents, n_latents))
        
        if layer_num > 1:
            blocks += [nn.ReLU(inplace=True), nn.Linear(n_latents, n_inputs)]
        #blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

    @property
    def device(self):
        return next(self.parameters()).device