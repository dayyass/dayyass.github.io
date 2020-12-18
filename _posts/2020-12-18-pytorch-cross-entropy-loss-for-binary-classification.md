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

This issue could be solved using logits of shape [batch_size, 2] in the binary classification task with CrossEntropyLoss, using the same interface for both problem statements (binary/multiclass). Why the value of the loss function for different approaches would not differ (up to logits transformation) shown in the [gist](https://gist.github.com/dayyass/f85a339111bbdd1b96e7ce632fe17d90).
