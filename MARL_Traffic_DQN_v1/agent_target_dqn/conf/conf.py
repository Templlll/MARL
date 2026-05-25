#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class Config:

    # Size of observation
    # observation的维度
    DIM_OF_OBSERVATION = 560
    DIM_OF_ACTION_PHASE = 4
    DIM_OF_ACTION_DURATION = 20
    DIM_SUB_ACTION_MASK = 24

    SOFTMAX = False

    # Algorithm Config
    # 算法的配置
    GAMMA = 0.9
    EPSILON = 0.1

    # ========== TODO 13 ==========
    # Tune the Target DQN hyperparameters.
    # Hint: Focus on learning rate, epsilon range, epsilon decay, and target update frequency.
    # 调优 Target DQN 超参数。
    # 提示：可重点尝试学习率、epsilon 起止值、epsilon 衰减率和目标网络更新频率。
    LR = 1e-3

    START_EPSILON_GREEDY = 1.0
    END_EPSILON_GREEDY = 0.3
    EPSILON_DECAY = 0.995
    LAMBDA = 0.75
    NUMB_HEAD = 2
    TARGET_UPDATE_FREQ = 500

    GRID_WIDTH = 14
    GRID_NUM = 20
    GRID_LENGTH = 5
    MAX_GREEN_DURATION = 40
    MAX_RED_DURATION = 60
