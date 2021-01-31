---
layout: post
title: PyTorch CrossEntropyLoss for Binary Classification
---

In a binary classification problem, a neural network usually returns a vector of logits of shape [batch_size], while in a multiclass classification problem, logits are represented as a matrix of shape [batch_size, n_classes].

For these tasks, different loss functions are used, and, therefore, the network training pipelines are also different, which is not convenient when you need to test hypotheses for both problem statements (binary/multiclass).

Pipeline schemes:
- binary classification:<br/>
logits (of shape [batch_size]) -> BCEWithLogitsLoss
- multiclass classification:<br/>
logits (of shape [batch_size, n_classes]) -> CrossEntropyLoss

This issue could be solved using logits of shape [batch_size, 2] in the binary classification task with CrossEntropyLoss, using the same interface for both problem statements (binary/multiclass). Why the value of the loss function for different approaches would not differ (up to logits transformation) shown in the gist.

Link to the gist [here](https://gist.github.com/dayyass/f85a339111bbdd1b96e7ce632fe17d90).

```
import torch
import torch.nn as nn

# hyperparams
BATCH_SIZE = 64

# init logits and targets
logits = torch.randn(BATCH_SIZE, 2)  # logits of shape [batch_size, 2]
targets = torch.randint(high=2, size=(BATCH_SIZE,))  # binary targets

# init two type of loss functions
criterion_bce = nn.BCELoss()
criterion_ce = nn.CrossEntropyLoss()

# compute losses
loss_ce = criterion_ce(
    input=logits,
    target=targets,
)
loss_bce = criterion_bce(
    input=torch.softmax(logits, dim=-1)[:, 1],  # transform logits to get equivalent loss value
    target=targets.float(),  # convert int64 targets to float32
)

# two losses are equivalent
torch.testing.assert_allclose(loss_ce, loss_bce)
```
