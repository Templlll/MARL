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

    # ---- A) 收集进口车道车辆信息 ----
    enter_vehicles = [
        v
        for v in vehicles
        if v.get("target_junction", junction_id) == junction_id and on_enter_lane(v)
    ]

    if len(enter_vehicles) > 0:
        avg_wait = float(np.mean([float(v.get("waiting_time", 0.0)) for v in enter_vehicles]))
        avg_delay = float(np.mean([float(v.get("delay", 0.0)) for v in enter_vehicles]))
        queue_len = float(sum(1 for v in enter_vehicles if float(v.get("speed", 0.0)) <= 0.1))
        n_veh = float(len(enter_vehicles))
    else:
        avg_wait, avg_delay, queue_len, n_veh = 0.0, 0.0, 0.0, 0.0

    # ---- B) 基础奖励：等待时间变化量（差分奖励，信号稠密） ----
    # 用等待时间的减少量直接作为奖励核心，与官方评分指标对齐
    prev_avg_wait = float(getattr(agent.preprocess, "prev_avg_wait", avg_wait))
    prev_queue_len = float(getattr(agent.preprocess, "prev_queue_len", queue_len))

    # 等待时间下降 → 正奖励；上升 → 负奖励（tanh压缩防止量级爆炸）
    delta_wait = prev_avg_wait - avg_wait        # 等待时间减少为正
    delta_queue = prev_queue_len - queue_len     # 排队减少为正

    agent.preprocess.prev_avg_wait = avg_wait
    agent.preprocess.prev_queue_len = queue_len

    # 归一化：等待时间典型范围约0-60s，排队约0-14辆
    base_reward = float(np.tanh(delta_wait / 10.0) + 0.5 * np.tanh(delta_queue / 5.0))

    # ---- C) 相位匹配奖励：选择压力最大的相位组 ----
    lane_group = get_webster_lane_group()
    group_pressure = {}
    for group_id, lane_ids in lane_group.items():
        pressure = 0.0
        for v in enter_vehicles:
            if v.get("lane", None) not in lane_ids:
                continue
            # 排队车辆权重高，等待时间作为辅助信号
            is_queued = 1.0 if float(v.get("speed", 0.0)) <= 0.1 else 0.3
            v_wait = float(v.get("waiting_time", 0.0))
            pressure += is_queued + 0.02 * v_wait
        group_pressure[int(group_id)] = pressure

    chosen_phase = int(act[1]) if len(act) > 1 and act[1] is not None else 0
    chosen_duration = int(act[2]) if len(act) > 2 and act[2] is not None else 0

    if group_pressure:
        best_phase = max(group_pressure, key=group_pressure.get)
        max_pressure = float(group_pressure[best_phase])
        chosen_pressure = float(group_pressure.get(chosen_phase, 0.0))
        # 归一化到 [-0.5, 0.5]
        phase_match = (chosen_pressure / (max_pressure + 1e-6)) - 0.5 if max_pressure > 0 else 0.0
    else:
        best_phase, max_pressure, chosen_pressure, phase_match = 0, 0.0, 0.0, 0.0

    # ---- D) 频繁切换惩罚（绿灯时长 < 8s 时惩罚）----
    frame_time = float(frame_state.get("frame_time", 0.0))
    # 判断时间单位：若frame_time > 1e4则推断单位为毫秒
    time_scale = 1000.0 if frame_time > 1.0e4 else 1.0

    last_phase = getattr(agent.preprocess, "last_phase_index", None)
    last_switch_time = getattr(agent.preprocess, "last_switch_time", None)

    switch_penalty = 0.0
    if last_phase is not None and chosen_phase != last_phase:
        # 发生相位切换：轻微惩罚
        switch_penalty -= 0.05
        if last_switch_time is not None:
            interval_sec = (frame_time - float(last_switch_time)) / time_scale
            # 绿灯时长过短（< 8s）额外惩罚，对应官方评分规则
            if interval_sec < 8.0:
                switch_penalty -= 0.2
        agent.preprocess.last_switch_time = frame_time
    elif last_switch_time is None:
        agent.preprocess.last_switch_time = frame_time

    agent.preprocess.last_phase_index = chosen_phase

    # ---- E) 时长匹配奖励：压力越大应给越长的绿灯 ----
    # DIM_OF_ACTION_DURATION=20，动作索引0-19对应时长步（每步2s，共0-38s）
    if max_pressure > 0.0:
        pressure_ratio = float(np.clip(chosen_pressure / (max_pressure + 1e-6), 0.0, 1.0))
    else:
        pressure_ratio = 0.5  # 无车时取中间时长即可
    target_duration_idx = int(np.round(pressure_ratio * (Config.DIM_OF_ACTION_DURATION - 1)))
    duration_gap = abs(chosen_duration - target_duration_idx)
    # 归一化到 [-0.5, 0.5]，量纲与 phase_match 对称
    duration_match = 0.5 - float(duration_gap) / float(Config.DIM_OF_ACTION_DURATION - 1)

    # ---- F) 双头奖励合成 ----
    # phase_head 强调相位匹配；duration_head 强调时长匹配；两头都包含基础奖励和切换惩罚
    phase_reward = 0.6 * base_reward + 0.4 * phase_match + switch_penalty
    duration_reward = 0.6 * base_reward + 0.4 * duration_match + switch_penalty

    return float(phase_reward), float(duration_reward)
