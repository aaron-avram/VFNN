"""
File containing all of the training functions
"""

import torch
import torch.nn as nn

def sgd(model: nn.Module, loss_func: callable, xs: torch.Tensor, ys: torch.Tensor, lr: float = 0.1, batch_size: int = 30, steps: int = 1000):
    """
    Perform stochastic gradient descent on model
    """
    lossi = []
    for step in range(steps):
        idx = torch.randint(0, len(xs), (batch_size,))
        x_batch, y_batch = xs[idx], ys[idx]

        logits = model.forward(x_batch)
        loss = loss_func(logits, y_batch)

        for param in model.parameters():
            param.grad = None

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

def adam(model: nn.Module, loss_func: callable, xs: torch.Tensor, ys: torch.Tensor, lr=1e-3, lambda_=1e-5, batch_size: int = 30, steps: int = 1000):
    """
    Perform adam for model
    """
    lossi = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_)
    for step in range(steps):
        idx = torch.randint(0, len(xs), (batch_size,))
        x_batch, y_batch = xs[idx], ys[idx]

        optimizer.zero_grad()
        preds = model(x_batch)
        loss = loss_func(preds, y_batch)

        loss.backward()
        optimizer.step()

        lossi.append(loss.item())

        if step % 100 == 0:
            print(f"Loss: {loss.item()} on step: {step + 1}")
    return lossi