---
layout: post
title: PyTorch ONNX GlobalPooling
---

While working on exporting PyTorch models to ONNX, my colleagues and I encountered the problem that ONNX does not support AdaptivePooling.

Therefore, to work around this problem, we decided to replace AdaptivePooling with GlobalPooling, and I wrote [helper classes](https://gist.github.com/dayyass/6d8f9f85f22a7d8e4179e18f624a652f) that can be used in PyTorch models.
I hope that for some of you they will make ONNX exporting easier.
