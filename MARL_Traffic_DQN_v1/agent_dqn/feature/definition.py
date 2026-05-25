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
from agent_dqn.conf.conf import Config
from agent_dqn.feature.traffic_utils import *


# SampleData with dimensions: define dimensions directly, no need for SampleData2NumpyData/NumpyData2SampleData
# SampleData with dimensions: 直接定义维度，不需要 SampleData2NumpyData/NumpyData2SampleData
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,  # 560
    _obs=Config.DIM_OF_OBSERVATION,  # 560
    act=4,
    # [phase(4 choices)]
    # [相位(4个选择)]
    rew=2,
    # [phase_reward, duration_reward]
    # [相位奖励, 持续时间奖励]
    done=1,
    legal_action=4,
    # phase legal actions
    # 相位合法动作
)

ObsData = create_cls("ObsData", feature=None, legal_action=None)

ActData = create_cls("ActData", junction_id=None, phase_index=None, duration=None)


def sample_process(list_game_data):
    r_data = np.array(list_game_data).squeeze()

    sample_datas = []
    for data in r_data:
        legal_action = [data.legal_action[0], data.legal_action[0], data.legal_action[0], data.legal_action[0]]
        sample_data = SampleData(
            obs=data.obs,
            _obs=None,
            act=data.act,
            rew=data.rew,
            done=1 if data.done == 0 else 0,
            legal_action=legal_action,
        )
        sample_datas.append(sample_data)

    for i in range(len(sample_datas) - 1):
        sample_datas[i]._obs = sample_datas[i + 1].obs
    sample_datas[-1]._obs = sample_datas[-1].obs

    if sample_datas[-1].done:
        del sample_datas[-1]

    return sample_datas


def reward_shaping(_obs, act, agent):
    """
    This function is an important function for reward processing, mainly responsible for:
        - Unpacking data, obtaining the data required for reward calculation from _obs
        - Reward calculation, calculating rewards based on the unpacked data
        - Reward concatenation, concatenating all rewards into a list

    Parameters:
        - _obs: The original feature data sent by battlesrv
        - act: The previous act predicted and executed
        - agent: real agent perform action

    Returns:
        - phase reward: The reward corresponding to the action of the phase number
        - duration reward: The reward corresponding to the action of the phase duration
    """
    """
    该函数是奖励处理的重要函数, 主要负责：
        - 数据解包, 从 _obs 获取计算奖励所需要的数据
        - 奖励计算, 根据解包的数据计算奖励
        - 奖励拼接, 将所有的奖励拼接成一个list

    参数：
        - _obs: battlesrv 发送的原始特征数据
        - act: 前一次预测并执行动作
        - agent: 实际执行动作智能体

    返回：
        - phase reward: 对应相位编号动作的奖励
        - duration reward: 对应相位持续时间动作的奖励
    """

    junction_id = 0
    phase_reward, duration_reward = 0, 0

    frame_state = _obs["frame_state"]
    vehicles = frame_state["vehicles"]

    # ========== TODO 9 ==========
    # Improve the reward function design.
    # Hint: Design phase_reward and duration_reward with waiting-time change, best phase matching, and switching penalties.
    # 完善奖励函数设计。
    # 提示：可结合等待时间变化、最佳相位匹配和切换惩罚设计 phase_reward 与 duration_reward。

    if act is None:
        return 0.0, 0.0

    # ---- A) Official-metric proxy reward (delay / queue / waiting_time) ----
    enter_vehicles = [
        v
        for v in vehicles
        if v.get("target_junction", junction_id) == junction_id and on_enter_lane(v)
    ]

    if len(enter_vehicles) > 0:
        avg_delay = float(np.mean([float(v.get("delay", 0.0)) for v in enter_vehicles]))
        avg_wait = float(np.mean([float(v.get("waiting_time", 0.0)) for v in enter_vehicles]))
        queue_len = float(sum(1 for v in enter_vehicles if float(v.get("speed", 0.0)) <= 0.1))
    else:
        avg_delay, avg_wait, queue_len = 0.0, 0.0, 0.0

    score_proxy = (
        1.0 / (1.0 + avg_delay / 9.0)
        + 1.0 / (1.0 + queue_len / 10.0)
        + 1.0 / (1.0 + avg_wait / 8.0)
    )

    prev_score_proxy = float(getattr(agent.preprocess, "prev_score_proxy", score_proxy))
    delta_score = score_proxy - prev_score_proxy
    agent.preprocess.prev_score_proxy = score_proxy

    instant_quality = (score_proxy - 1.5) / 1.5
    base_reward = float(np.tanh(2.0 * delta_score) + 0.2 * instant_quality)

    # ---- B) Phase matching (choose higher-pressure group) ----
    lane_group = get_webster_lane_group()
    group_pressure = {}
    for group_id, lane_ids in lane_group.items():
        pressure = 0.0
        for v in enter_vehicles:
            if v.get("lane", None) not in lane_ids:
                continue
            v_wait = float(v.get("waiting_time", 0.0))
            v_delay = float(v.get("delay", 0.0))
            is_queued = 1.0 if float(v.get("speed", 0.0)) <= 0.1 else 0.0
            pressure += 1.0 * is_queued + 0.05 * v_wait + 0.02 * v_delay
        group_pressure[int(group_id)] = pressure

    best_phase = max(group_pressure, key=group_pressure.get) if group_pressure else 0
    chosen_phase = int(act[1]) if len(act) > 1 and act[1] is not None else 0
    chosen_duration = int(act[2]) if len(act) > 2 and act[2] is not None else 0

    max_pressure = float(group_pressure.get(best_phase, 0.0))
    chosen_pressure = float(group_pressure.get(chosen_phase, 0.0))

    if max_pressure > 0.0:
        phase_match = (chosen_pressure / (max_pressure + 1e-6)) - 0.5
    else:
        phase_match = 0.0

    # ---- C) Switching penalties (penalize green interval < 8s) ----
    frame_time = float(frame_state.get("frame_time", 0.0))
    time_scale = 1000.0 if frame_time > 1.0e4 else 1.0

    last_phase = getattr(agent.preprocess, "last_phase_index", None)
    last_switch_time = getattr(agent.preprocess, "last_switch_time", None)

    fast_switch_penalty = 0.0
    if last_phase is not None and chosen_phase != last_phase:
        fast_switch_penalty -= 0.02
        if last_switch_time is not None:
            interval_sec = (frame_time - float(last_switch_time)) / time_scale
            if interval_sec < 8.0:
                fast_switch_penalty -= 0.15
        agent.preprocess.last_switch_time = frame_time
    elif last_switch_time is None:
        agent.preprocess.last_switch_time = frame_time

    agent.preprocess.last_phase_index = chosen_phase

    # ---- D) Duration matching: longer green when pressure higher ----
    if max_pressure > 0.0:
        pressure_ratio = float(np.clip(chosen_pressure / (max_pressure + 1e-6), 0.0, 1.0))
    else:
        pressure_ratio = 0.0
    target_duration = int(np.round(pressure_ratio * (Config.DIM_OF_ACTION_DURATION - 1)))
    duration_gap = abs(chosen_duration - target_duration)
    duration_match = 0.5 - float(duration_gap) / max(1.0, (Config.DIM_OF_ACTION_DURATION - 1))

    # ---- E) Two-head rewards ----
    # phase head: emphasize phase matching; duration head: emphasize duration matching.
    phase_reward = base_reward + 0.3 * phase_match + fast_switch_penalty
    duration_reward = base_reward + 0.3 * duration_match + fast_switch_penalty

    return float(phase_reward), float(duration_reward)
