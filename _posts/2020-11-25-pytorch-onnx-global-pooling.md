---
layout: post
title: PyTorch ONNX GlobalPooling
---

While working on exporting PyTorch models to ONNX, my colleagues and I encountered the problem that ONNX does not support AdaptivePooling.

Therefore, to work around this problem, we decided to replace AdaptivePooling with GlobalPooling, and I wrote helper classes that can be used in PyTorch models.
I hope that for some of you they will make ONNX exporting easier.

Link to code [here](https://gist.github.com/dayyass/6d8f9f85f22a7d8e4179e18f624a652f).

Examples:

```
class GlobalAvgPool2d(nn.Module):
    """
    Reduce mean over last two dimensions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.mean(dim=-1, keepdim=True)
        return x.mean(dim=-2, keepdim=True)


class GlobalMaxPool2d(nn.Module):
    """
    Reduce max over last two dimensions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.max(dim=-1, keepdim=True)[0]
        return x.max(dim=-2, keepdim=True)[0]
```
