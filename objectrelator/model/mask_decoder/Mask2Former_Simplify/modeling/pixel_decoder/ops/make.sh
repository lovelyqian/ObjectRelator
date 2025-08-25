#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR


# 清理旧编译文件
# rm -rf build
# rm -rf MultiScaleDeformableAttention.egg-info





# debug
# export CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
# export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
# export TORCH_CUDA_ARCH_LIST="8.0"


python setup.py build install
