#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import torch.nn as nn
from torch.nn import ModuleDict
import torch.nn.functional as F

import numpy as np
from math import ceil, floor
from collections import OrderedDict
from typing import Dict, List, Tuple
from agent_ppo.conf.conf import Config


class Model(nn.Module):
    def __init__(self, device=None):
        super(Model, self).__init__()
        # Feature configure parameter
        # 特征配置参数
        self.model_name = "network_traffic_v1"

        self.device = device
        self.label_size_list = Config.LABEL_SIZE_LIST

        # ========== TODO 5 ==========
        # Improve the PPO shared feature layer and Actor/Critic network structure.
        # Hint: Add suitable hidden layers and keep self.unit_size aligned with the backbone output.
        # 完善 PPO 共享特征层与 Actor/Critic 网络结构。
        # 提示：可补充隐藏层设计，并保持 self.unit_size 与主干网络输出一致。
        self.unit_size = 256
        all_dims = [Config.DIM_OF_OBSERVATION, 512, 256]
        self.main_mlp = MLP(all_dims, "main_mlp")

        # Output label
        # 输出标签
        self.label_mlp = ModuleDict(
            {
                "label{0}_mlp".format(label_index): MLP(
                    [self.unit_size, self.label_size_list[label_index]],
                    "label{0}_mlp".format(label_index),
                )
                for label_index in range(len(self.label_size_list))
            }
        )

        self.value_mlp = MLP([self.unit_size, 64, 1], "value_mlp")

    def forward(self, s, inference=False):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(
                np.array(s, dtype=np.float32),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            s = s.to(torch.float32)

        main_nn = self.main_mlp(s)

        result_list = []
        # Output label
        # 输出标签
        for label_index, label_dim in enumerate(self.label_size_list[:]):
            label_mlp_out = self.label_mlp["label{0}_mlp".format(label_index)](main_nn)
            result_list.append(label_mlp_out)

        # Output value
        # 输出价值
        value_result = self.value_mlp(main_nn)
        result_list.append(value_result)

        # Prepare for inference graph
        # 准备推理图
        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]
        if inference:
            return [logits, value]
        else:
            return result_list

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()


#######################
# Utility functions
# 工具函数
#######################
def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    """Wrapper function to create and initialize a linear layer

    创建和初始化线性层的包装函数

    Args:
        in_features (int): ``in_features``
        输入特征维度
        out_features (int): ``out_features``
        输出特征维度

    Returns:
        nn.Linear: the initialized linear layer
        nn.Linear: 初始化后的线性层
    """
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)

    # Initialize weight and bias
    # 初始化权重和偏置
    nn.init.orthogonal(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)

    return fc_layer


############################
# Building-block classes
# 构建块类
############################
class MLP(nn.Module):
    """A simple multi-layer perceptron

    简单的多层感知机
    """

    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        """Create a MLP object

        创建一个 MLP 对象

        Args:
            fc_feat_dim_list (List[int]): ``in_features`` of the first linear layer followed by
                ``out_features`` of each linear layer
                第一个线性层的 ``in_features`` 以及后续每个线性层的 ``out_features``
            name (str): human-friendly name, serving as prefix of each comprising layers
                易读的名称，作为组成层的前缀
            non_linearity (nn.Module, optional): the activation function to use. Defaults to nn.ReLU.
                使用的激活函数。默认为 nn.ReLU。
            non_linearity_last (bool, optional): whether to append a activation function in the end.
                Defaults to False.
                是否在最后添加激活函数。默认为 False。
        """
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            # No relu for the last fc layer of the mlp unless required
            # 除非需要，否则mlp的最后一个全连接层不使用relu
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
