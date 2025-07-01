"""
File containing all of the training functions
"""

import torch
import torch.nn as nn

def sgd(model: nn.Module, xs: torch.Tensor, ys: torch.Tensor, lr: float = 0.1, batch_size: int = 30, steps: int = 1000):
    """
    Perform stochastic gradient descent on model
    """
    lossi = []
    for step in range(steps):
        idx = torch.randint(0, len(xs), (batch_size,))
        x_batch, y_batch = xs[idx], ys[idx]

        logits = model.forward(x_batch)
        loss = nn.functional.cross_entropy(logits, y_batch)

        l2_lambda = 1e-4
        l2_penalty = torch.tensor(0.0)
        for param in model.parameters():
            param.grad = None
            l2_penalty += (param * param).sum()

        loss = loss + l2_lambda * l2_penalty

        loss.backward()
        lossi.append(loss.log10().item())
        

        for param in model.parameters():
            param.grad = param.grad.clamp(-5, 5)
            param.data += -lr * param.grad
    
        if step == steps // 2:
            lr = lr * 0.1

        if step % 100 == 0:
            print(f"Loss: {loss.item()} on step: {step + 1}")
    return lossi