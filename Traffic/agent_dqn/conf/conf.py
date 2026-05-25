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
    # 交通信号控制每次决策间隔较长（数十秒），需要更长远的折扣以传递奖励信号
    GAMMA = 0.99
    EPSILON = 0.1

    # ========== TODO 10 ==========
    # Tune the DQN hyperparameters.
    # Hint: Focus on learning rate, epsilon range, epsilon decay, and target update frequency.
    # 调优 DQN 超参数。
    # 提示：可重点尝试学习率、epsilon 起止值、epsilon 衰减率和目标网络更新频率。
    LR = 1e-4  # 降低学习率，防止Q值震荡发散

    START_EPSILON_GREEDY = 1.0
    END_EPSILON_GREEDY = 0.05
    # 每episode约200步，0.998^200≈0.67，几百个epoch后才能充分探索
    EPSILON_DECAY = 0.998
    LAMBDA = 0.75
    NUMB_HEAD = 2
    # 降低同步频率，让target网络更稳定（每200步同步一次更适合小批量场景）
    TARGET_UPDATE_FREQ = 200

    GRID_WIDTH = 14
    GRID_NUM = 20
    GRID_LENGTH = 5
    MAX_GREEN_DURATION = 40
    MAX_RED_DURATION = 60
