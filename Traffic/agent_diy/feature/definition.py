#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import create_cls
import numpy as np
from agent_diy.conf.conf import Config

ObsData = create_cls("ObsData", feature=None, sub_action_mask=None)

ActData = create_cls("ActData", junction_id=None, phase_index=None, duration=None)

# SampleData with dimensions: define dimensions directly, no need for SampleData2NumpyData/NumpyData2SampleData
# SampleData with dimensions: 直接定义维度，不需要 SampleData2NumpyData/NumpyData2SampleData
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,  # 560
    _obs=Config.DIM_OF_OBSERVATION,  # 560
    act=4,
    # action dimension
    # 动作维度
    rew=2,
    # reward dimension
    # 奖励维度
    ret=1,
    # return
    # 回报
    done=1,
    sub_action_mask=Config.DIM_SUB_ACTION_MASK,  # 24
    _sub_action_mask=Config.DIM_SUB_ACTION_MASK,  # 24
)


def sample_process(list_game_data):
    pass
