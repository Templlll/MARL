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
from agent_ppo.conf.conf import Config
from agent_ppo.feature.traffic_utils import *


ObsData = create_cls("ObsData", feature=None, legal_action=None, sub_action_mask=None)

ActData = create_cls("ActData", junction_id=None, action=None, d_action=None, prob=None, value=None)

# SampleData with dimensions: define dimensions directly, no need for SampleData2NumpyData/NumpyData2SampleData
# SampleData with dimensions: 直接定义维度，不需要 SampleData2NumpyData/NumpyData2SampleData
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,  # 560
    legal_action=Config.DIM_OF_ACTION_PHASE_1 + Config.DIM_OF_ACTION_DURATION_1,  # 8
    act=Config.NUMB_HEAD,  # 2
    reward=1,
    reward_sum=1,
    done=1,
    value=1,
    next_value=1,
    advantage=1,
    prob=Config.DIM_OF_ACTION_PHASE_1 + Config.DIM_OF_ACTION_DURATION_1,  # 8
    sub_action=Config.NUMB_HEAD,  # 2
    is_train=1,
)


def sample_process(list_sample_data):
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_reward(list_sample_data)

    return list_sample_data


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

    # ========== TODO 15 ==========
    # Improve the reward function design.
    # Hint: Build the reward with waiting-time change, phase matching, traffic efficiency, and switching penalties.
    # 完善奖励函数设计。
    # 提示：可结合等待时间变化、相位匹配、通行效率和切换惩罚构造奖励。

    if act is None:
        return 0.0

    # ---- A) Align with official metrics (delay / queue / waiting_time) ----
    # 官方评分核心：平均延误、排队长度、等待时间 + 频繁切换惩罚。
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

    # Dense reward: combine instantaneous quality + improvement.
    # 稠密奖励：兼顾“当前好坏”与“相对上一步的改善”。
    instant_quality = (score_proxy - 1.5) / 1.5  # roughly in [-1, 1]
    metrics_reward = float(np.tanh(2.0 * delta_score) + 0.2 * instant_quality)

    # ---- B) Phase matching via lane-group pressure (based on delay/wait/queue) ----
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

    chosen_phase = int(act[1]) if len(act) > 1 and act[1] is not None else 0
    chosen_duration = int(act[2]) if len(act) > 2 and act[2] is not None else 5

    best_phase = max(group_pressure, key=group_pressure.get) if group_pressure else 0
    max_pressure = float(group_pressure.get(best_phase, 0.0))
    chosen_pressure = float(group_pressure.get(chosen_phase, 0.0))

    if max_pressure > 0.0:
        phase_reward = (chosen_pressure / (max_pressure + 1e-6)) - 0.5
    else:
        phase_reward = 0.0

    # ---- C) Switching penalties (official: penalize green intervals < 8s) ----
    frame_time = float(frame_state.get("frame_time", 0.0))
    time_scale = 1000.0 if frame_time > 1.0e4 else 1.0  # heuristically treat as ms when large

    last_phase = getattr(agent.preprocess, "last_phase_index", None)
    last_switch_time = getattr(agent.preprocess, "last_switch_time", None)

    switch_penalty = 0.0
    if last_phase is not None and chosen_phase != last_phase:
        switch_penalty -= 0.02
        if last_switch_time is not None:
            interval_sec = (frame_time - float(last_switch_time)) / time_scale
            if interval_sec < 8.0:
                switch_penalty -= 0.15
        agent.preprocess.last_switch_time = frame_time
    elif last_switch_time is None:
        agent.preprocess.last_switch_time = frame_time

    agent.preprocess.last_phase_index = chosen_phase

    # ---- D) Duration matching: longer green when pressure higher ----
    if max_pressure > 0.0:
        pressure_ratio = float(np.clip(chosen_pressure / (max_pressure + 1e-6), 0.0, 1.0))
    else:
        pressure_ratio = 0.0
    target_duration = (int(np.round(pressure_ratio * (Config.DIM_OF_ACTION_DURATION_1 - 1))) + 1) * 5
    duration_gap = abs(chosen_duration - target_duration)
    duration_reward = -float(duration_gap) / 15.0

    # ---- E) Combine (keep scale moderate) ----
    total_reward = 0.6 * metrics_reward + 0.25 * phase_reward + 0.1 * duration_reward + switch_penalty

    return float(total_reward)


def _calc_reward(list_sample_data):
    """
    Calculate cumulated reward and advantage with GAE.
    reward_sum: used for value loss
    advantage: used for policy loss
    V(s) here is a approximation of target network

    使用 GAE 计算累积奖励和优势函数。
    reward_sum: 用于价值损失
    advantage: 用于策略损失
    V(s) 这里是目标网络的近似值
    """

    gae, last_gae = 0.0, 0.0
    gamma, lamda = Config.GAMMA, Config.LAMDA
    for rl_info in reversed(list_sample_data):
        delta = -rl_info.value + rl_info.reward + gamma * rl_info.next_value
        gae = gae * gamma * lamda + delta
        rl_info.advantage = gae
        rl_info.reward_sum = gae + rl_info.value
