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
    DIM_OF_ACTION_PHASE_1 = 4
    DIM_OF_ACTION_DURATION_1 = 4
    DIM_SUB_ACTION_MASK = 24

    # Number of output heads for policy, 2 heads if including phase and time
    # 策略输出几个头，如果包含相位和时间，则为2个头
    NUMB_HEAD = 2

    GRID_WIDTH = 14
    GRID_NUM = 20
    GRID_LENGTH = 5
    MAX_GREEN_DURATION = 40
    MAX_RED_DURATION = 60

    # ========== TODO 16 ==========
    # Tune the PPO hyperparameters.
    # Hint: Focus on learning rate, gradient clipping range, VALUE_COEF, and exploration-related factors.
    # 调优 PPO 超参数。
    # 提示：可重点尝试学习率、梯度裁剪范围、VALUE_COEF 和探索相关系数。
    INIT_LEARNING_RATE_START = 0.0003
    # Entropy coefficient (encourage exploration). Too large may keep policy near-uniform.
    # 熵系数（鼓励探索）。过大会让策略长期接近均匀随机。
    BETA_START = 0.005
    LOG_EPSILON = 1e-6

    RMSPROP_DECAY = 0.9
    RMSPROP_MOMENTUM = 0.0
    RMSPROP_EPSILON = 0.01
    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001

    LABEL_SIZE_LIST = [DIM_OF_ACTION_PHASE_1, DIM_OF_ACTION_DURATION_1]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    IS_REINFORCE_TASK_LIST = [
        True,
    ] * NUMB_HEAD

    EVAL_FREQ = 5
    GAMMA = 0.99
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 1.0
    VALUE_COEF = 1
