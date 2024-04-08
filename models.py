import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

torch.manual_seed(0)



class BayesianLayer(nn.Module):
    def __init__(self):
        super(BayesianLayer, self).__init__()

    def resample_eps(self):
        raise NotImplementedError

    def zero_eps(self):
        raise NotImplementedError


class BayesianLinearLayer(BayesianLayer):
    def __init__(self, input_dim, output_dim):
        """
        params:
        input_dim: dimension of the input
        output_dim: dimension of the output
        """
        super(BayesianLinearLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.frozen = False

        # initialize the weight and bias
        # Sigmas are parametrized using their logs to ensure positivity, like in paper implementation. Unlike the paper implementation, we don't store the log of the variance (=2*log(sigma)) but the log of sigma
        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(output_dim, input_dim))   
        self.bias_mu = nn.Parameter(torch.Tensor(output_dim))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(output_dim))

        self.weight_eps = None
        self.bias_eps = None

    def resample_eps(self):
        """
        resample the eps for weight and bias
        """
        self.weight_eps = torch.randn_like(self.weight_mu)
        self.bias_eps = torch.randn_like(self.bias_mu)

    def zero_eps(self):
        """
        set the eps for weight and bias to zero
        """
        self.weight_eps = torch.zeros_like(self.weight_mu)
        self.bias_eps = torch.zeros_like(self.bias_mu)

    def forward(self, x):
        return F.linear(
            x,
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps,
            bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps
        )

    def KL_const_term(self, prior_layer: nn.Module):
        weight_const_term = -0.5 * self.output_dim * self.input_dim
        bias_const_term = -0.5 * self.output_dim
        return torch.tensor(weight_const_term + bias_const_term, dtype=self.weight_mu.dtype, device=self.weight_mu.device)

    def KL_log_std_term(self, prior_layer: nn.Module):
        weight_log_std_term = (prior_layer.weight_log_sigma - self.weight_log_sigma).sum()
        bias_log_std_term = (prior_layer.bias_log_sigma - self.bias_log_sigma).sum()
        return weight_log_std_term + bias_log_std_term
    
    def KL_mu_diff_term(self, prior_layer: nn.Module):
        weight_mu_diff_term = 0.5 * ((self.weight_mu - prior_layer.weight_mu)**2 * torch.exp(-2 * prior_layer.weight_log_sigma)).sum()
        bias_mu_diff_term = 0.5 * ((self.bias_mu - prior_layer.bias_mu)**2 * torch.exp(-2 * prior_layer.bias_log_sigma)).sum()
        return weight_mu_diff_term + bias_mu_diff_term
    
    def KL_std_quotient_term(self, prior_layer: nn.Module):
        weight_std_quotient_term = 0.5 * (torch.exp(2 * self.weight_log_sigma - 2 * prior_layer.weight_log_sigma)).sum()
        bias_std_quotient_term = 0.5 * (torch.exp(2 * self.bias_log_sigma - 2 * prior_layer.bias_log_sigma)).sum()
        return weight_std_quotient_term + bias_std_quotient_term

    def KL(self, prior_layer: nn.Module, components=False):
        """
        Compute the KL divergence between the distributions parametrized by this model vs by the prior layer
        """
        const_term = self.KL_const_term(prior_layer)
        log_std_term = self.KL_log_std_term(prior_layer)
        mu_diff_term = self.KL_mu_diff_term(prior_layer)
        std_quotient_term = self.KL_std_quotient_term(prior_layer)

        if components:
            return const_term, log_std_term, mu_diff_term, std_quotient_term
        else:
            return const_term + log_std_term + mu_diff_term + std_quotient_term


def resample_bayesian_layers(model: nn.Module):
    """
    resample the eps for all BayesianLayers in the model
    """
    for layer in model.modules():
        if isinstance(layer, BayesianLayer):
            layer.resample_eps()


class VanillaNN(nn.Module):
    def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
        """
        params:
        num_hidden_layers: usually 2
        """
        super(VanillaNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers

        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.intermediate_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers-1)])
        self.last_layer = nn.Linear(hidden_dim, output_dim)


    def forward(self, x, task_idxs=None):
        """
        outputs the logits of the network
        to obtain a probability vector over the possible classes, apply a softmax

        params:
        x: input tensor
        task_idx: tensor containing the indices of this task's output logits
        """

        x = torch.flatten(x, start_dim=1)

        if task_idxs == None:
            task_idxs = torch.arange(self.output_dim)    # if task_idxs is not provided, assume all outputs are needed

        x = F.relu(self.first_layer(x))
        for layer in self.intermediate_layers:
            x = F.relu(layer(x))
        x = self.last_layer(x)
        return x[:,task_idxs]
    

class BayesianNN(nn.Module):
    def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
        """
        params:
        num_hidden_layers: usually 2
        """
        super(BayesianNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers

        self.first_layer = BayesianLinearLayer(input_dim, hidden_dim)
        self.intermediate_layers = nn.ModuleList([BayesianLinearLayer(hidden_dim, hidden_dim) for _ in range(num_hidden_layers-1)])
        self.last_layer = BayesianLinearLayer(hidden_dim, output_dim)

    def forward(self, x, task_idxs=None):
        """
        outputs the logits of the network
        to obtain a probability vector over the possible classes, apply a softmax

        params:
        x: input tensor
        task_idx: tensor containing the indices of this task's output logits
        """

        x = torch.flatten(x, start_dim=1)

        if task_idxs == None:
            task_idxs = torch.arange(self.output_dim)    # if task_idxs is not provided, assume all outputs are needed

        x = F.relu(self.first_layer(x))
        for layer in self.intermediate_layers:
            x = F.relu(layer(x))
        x = self.last_layer(x)
        return x[:,task_idxs]
    
    def initialize_to_prior(self, prior_mu=0, prior_sigma=1):
        """
        Initialize the weights and biases of all BayesianLayers to the prior
        """
        log_prior_sigma = np.log(prior_sigma)

        for module in self.modules():
            if isinstance(module, BayesianLinearLayer):
                module.weight_mu.data = prior_mu * torch.ones_like(module.weight_mu)
                module.weight_log_sigma.data = log_prior_sigma * torch.ones_like(module.weight_log_sigma)
                module.bias_mu.data = prior_mu * torch.ones_like(module.bias_mu)
                module.bias_log_sigma.data = log_prior_sigma * torch.ones_like(module.bias_log_sigma)
    
    def initialize_using_vanilla(self, van_model: VanillaNN, prior_mu=0, prior_sigma=1):
        """
        Initialize the weights and biases of all BayesianLayers to the weights and biases of the corresponding layers in the vanilla model
        And Initialize the unknown weights and biases to the prior
        """
        van_modules = dict(van_model.named_modules())

        for name, module in self.named_modules():
            if isinstance(module, BayesianLinearLayer):
                assert isinstance(van_modules[name], nn.Linear)

                # copy weights of the vanilla model into the means
                module.weight_mu.data = van_modules[name].weight.data
                module.bias_mu.data = van_modules[name].bias.data

                # initialize the sigmas to exp(-3) = 0.05 like in paper implementation
                module.weight_log_sigma.data = -3 * torch.ones_like(module.weight_log_sigma)
                module.bias_log_sigma.data = -3 * torch.ones_like(module.bias_log_sigma)

        # Note: the unknown weight and bias sigmas are also initialized to exp(-3)=0.05, instead of to the prior!!
        # The paper implementation also initializes the means to a truncated normal distribution with std=0.1, but that is not of the essence for this project
        

    def KL(self, prior_model: nn.Module, components=False):
        """
        Compute the KL divergence between the distributions parametrized by this model vs by the prior model
        """

        for param in prior_model.parameters():
            param.requires_grad = False

        # sum KL divergences for all BayesianLayers
        # possible because KL divergence of independent distributions is additive
        if components:
            const_term = 0
            log_std_term = 0
            mu_diff_term = 0
            std_quotient_term = 0

            for layer, prior_layer in zip(self.modules(), prior_model.modules()):
                if isinstance(layer, BayesianLayer):
                    assert isinstance(prior_layer, BayesianLayer)
                    ct,lst,mdt,sqt = layer.KL(prior_layer, components=components)
                    const_term += ct
                    log_std_term += lst
                    mu_diff_term += mdt
                    std_quotient_term += sqt
            
            return const_term, log_std_term, mu_diff_term, std_quotient_term

        else:
            KL = 0
            for layer, prior_layer in zip(self.modules(), prior_model.modules()):
                if isinstance(layer, BayesianLayer):
                    assert isinstance(prior_layer, BayesianLayer)
                    KL += layer.KL(prior_layer, components=components)
            
            return KL

    def resample_eps(self):
        """
        resample the eps for all BayesianLayers in the model
        """
        for layer in self.modules():
            if isinstance(layer, BayesianLayer):
                layer.resample_eps()

    def zero_eps(self):
        """
        set the eps for all BayesianLayers in the model to zero
        """
        for layer in self.modules():
            if isinstance(layer, BayesianLayer):
                layer.zero_eps()
    
    def shared_parameters(self):
        """
        return the shared parameters of the BayesianLayers
        """
        yield from self.first_layer.parameters()
        yield from self.intermediate_layers.parameters()

    def task_specific_parameters(self):
        yield from self.last_layer.parameters()

    def named_shared_parameters(self):
        """
        return the names and shared parameters of the BayesianLayers
        """
        yield from self.first_layer.named_parameters()
        yield from self.intermediate_layers.named_parameters()
    
    def named_task_specific_parameters(self):
        yield from self.last_layer.named_parameters()