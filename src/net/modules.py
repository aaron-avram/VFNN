"""
A file containing all of the project's network modules
"""

import torch
import torch.nn as nn

# Feature dim is always last dim
class BatchNorm1D(nn.Module):
    """
    Batchnorm with respect to a single axis
    """
    def __init__(self, num_features, _momentum=0.9):
        super().__init__()
        self.running_mu = torch.zeros((num_features,))
        self.running_var = torch.ones((num_features,))
        self.gamma = nn.Parameter(torch.ones((num_features,), requires_grad=True))
        self.beta = nn.Parameter(torch.zeros((num_features,), requires_grad=True))
        self.momentum = _momentum
    
    def forward(self, inp: torch.Tensor):
        ndims = inp.ndim - 1
        c = self.running_mu.shape[0]
        if self.training:
            shape = tuple(range(0, ndims))
            mu = torch.mean(inp, dim=shape, keepdim=True)
            var = torch.var(inp, dim=shape, keepdim=True, unbiased=False)
        else:
            mu = self.running_mu.view(*([1] * ndims), c)
            var = self.running_var.view(*([1] * ndims), c)
        
        gamma = self.gamma.view(*([1] * ndims), c)
        beta = self.beta.view(*([1] * ndims), c)

        # Normalize then reparametrize
        x_hat = (inp - mu) / torch.sqrt(var + 1e-12)
        x_new = gamma * x_hat + beta

        # Update running average -- if training
        if self.training:
            self.running_mu = self.momentum * self.running_mu + (1 - self.momentum) * mu.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.squeeze()
        
        return x_new

class Gate(nn.Module):
    """
    A gate in an LSTM cell
    """
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.in_weight = nn.Parameter(torch.randn((in_size, hidden_size), requires_grad=True) / torch.sqrt(torch.tensor(in_size)))
        self.hidden_weight = nn.Parameter(torch.randn((hidden_size, hidden_size), requires_grad=True) / torch.sqrt(torch.tensor(hidden_size)))
        self.bias = nn.Parameter(torch.zeros((hidden_size,), requires_grad=True))
        self.act = torch.sigmoid
    
    def forward(self, x_inp, hidden_inp):
        unact = hidden_inp @ self.hidden_weight + x_inp @ self.in_weight + self.bias
        return self.act(unact)
    
class Cell(nn.Module):
    """
    An LSTM cell
    """
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.forget_gate = Gate(in_size, hidden_size)
        self.input_gate = Gate(in_size, hidden_size)
        self.out_gate = Gate(in_size, hidden_size)
        self.inp_weight = nn.Parameter(torch.randn((in_size, hidden_size), requires_grad=True) / torch.sqrt(torch.tensor(in_size)))
        self.hidden_weight = nn.Parameter(torch.randn((hidden_size, hidden_size), requires_grad=True) / torch.sqrt(torch.tensor(hidden_size)))
        self.bias = nn.Parameter(torch.zeros((hidden_size,), requires_grad=True))
    
    def forward(self, x_inp, hidden_inp, memory_inp):
        forget_gate = self.forget_gate(x_inp, hidden_inp)
        input_gate = self.input_gate(x_inp, hidden_inp)
        out_gate = self.out_gate(x_inp, hidden_inp)

        candidate_mem = torch.tanh(hidden_inp @ self.hidden_weight + x_inp @ self.inp_weight + self.bias)
        new_mem = forget_gate * memory_inp + input_gate * candidate_mem
        new_output = out_gate * torch.tanh(new_mem)
        return (new_output, new_mem)

class LSTM(nn.Module):
    """
    An LSTM block as a network or a component of one
    """
    def __init__(self, in_size, hidden_size, training=True):
        super().__init__()
        self.cell = Cell(in_size, hidden_size)
        self.hidden_size =hidden_size
        self.training = training
    
    def forward(self, inp: torch.Tensor):
        batch_size = inp.shape[0]
        hidden = torch.zeros(batch_size, self.hidden_size, device=inp.device)
        mem = torch.zeros_like(hidden)

        if inp.ndim == 2:
            channels = inp.shape[0]
        elif inp.ndim == 3:
            channels = inp.shape[1]
        else:
            raise ValueError
        
        for t in range(channels):

            # Deal with dimension cases
            if inp.ndim == 2:
                x = inp[t, :]
            else:
                x = inp[:, t, :]
            # Cell update
            hidden, mem = self.cell(x, hidden, mem)
        return hidden

class MLP(nn.Module):
    """
    A multilayer perceptron
    """
    def __init__(self, size: tuple, training=True):
        super().__init__()
        self.layers = nn.Sequential()
        for l1, l2 in zip(size, size[1:]):
            self.layers.append(nn.Linear(l1, l2))
            self.layers.append(nn.ReLU())
        self.layers.pop(-1)
        self.training=training
    
    def forward(self, inp):
        return self.layers.forward(inp).squeeze()

class VFNN(nn.Module):
    """
    The LSTM neural network used in this project for volatility forecasting
    """
    def __init__(self, hidden_size: int, inp_size: int, MLP_size: tuple, training=True):
        super().__init__()
        self.blocks = nn.Sequential()
        lstm = LSTM(in_size=inp_size, hidden_size=hidden_size)
        self.blocks.append(lstm)
        mlp = MLP(MLP_size)
        self.blocks.append(mlp)
        self.training = training

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.blocks.forward(inp)
    