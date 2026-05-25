#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class Config:

    # model
    # 模型配置
    DIM_OF_OBSERVATION = 560
    DIM_OF_ACTION_PHASE = 4
    DIM_OF_ACTION_DURATION = 20
    DIM_SUB_ACTION_MASK = 24

    SOFTMAX = False

    # Algorithm Config
    # 算法的配置
    LR = 3e-4
