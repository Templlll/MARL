#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time
from agent_target_dqn.feature.definition import *
from common_python.utils.common_func import Frame
from tools.train_env_conf_validate import read_usr_conf
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env, agent = envs[0], agents[0]
    epoch_num = 100000
    episode_num_every_epoch = 1
    last_save_model_time = 0

    # Initializing monitoring data
    # 监控数据初始化
    monitor_data = {
        "reward": 0,
    }
    last_report_monitor_time = time.time()

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_target_dqn/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error(f"usr_conf is None, please check agent_target_dqn/conf/train_env_conf.toml")
        return

    for epoch in range(epoch_num):
        epoch_total_rew = 0

        data_length = 0
        for g_data in run_episodes(episode_num_every_epoch, env, agent, usr_conf, logger):
            data_length += len(g_data)
            total_rew = []
            for data in g_data:
                total_rew.append(data.rew[0])

            total_rew = sum(total_rew)
            epoch_total_rew += total_rew
            agent.send_sample_data(g_data)
            g_data.clear()

        avg_step_reward = 0
        if data_length:
            avg_step_reward = f"{(epoch_total_rew/data_length):.2f}"

        # save model file
        # 保存model文件
        now = time.time()
        if now - last_save_model_time >= 1800:
            agent.save_model()
            last_save_model_time = now

        # Reporting training progress
        # 上报训练进度
        if now - last_report_monitor_time > 60:
            monitor_data["reward"] = avg_step_reward
            if monitor:
                monitor.put_data({os.getpid(): monitor_data})
                last_report_monitor_time = now

        logger.info(f"Avg Step Reward: {avg_step_reward}, Epoch: {epoch}, Data Length: {data_length}")


def run_episodes(n_episode, env, agent, usr_conf, logger):
    try:
        train_test_quick_stop = os.environ.get("is_train_test", "False").lower() == "true"
        for _ in range(n_episode):
            collector = list()
            predict_cnt = 0

            # Retrieving training metrics
            # 获取训练中的指标
            training_metrics = get_training_metrics()
            if training_metrics:
                logger.info(f"training_metrics is {training_metrics}")

            # At the start of each environment, loading the latest model file
            # 每次对局开始时, 加载最新model文件
            agent.load_model(id="latest")

            # Reset the environment and get the initial extra_info
            # 重置环境, 并获取初始状态
            env_obs = env.reset(usr_conf=usr_conf)
            # Disaster recovery
            # 容灾
            if handle_disaster_recovery(env_obs, logger):
                break

            agent.reset(env_obs)
            obs = env_obs["observation"]
            extra_info = env_obs["extra_info"]

            # Record the last_predict_act
            # 记录上次预测的动作
            last_predict_act = None

            done = False
            while not done:
                need_to_predict = obs["legal_action"][0] != 0
                if need_to_predict:
                    if len(collector) > 0:
                        # Calculate reward Rewards
                        # 计算奖励
                        reward = reward_shaping(obs, last_predict_act, agent)
                        collector[-1].rew = reward

                    # Feature processing
                    # 特征处理
                    obs_data = agent.observation_process(obs, extra_info)
                    # Agent makes a prediction to get the next frame's action
                    # Agent 进行推理, 获取下一帧的预测动作
                    act_data = agent.predict(list_obs_data=[obs_data])

                    # Unpack ActData into actions
                    # ActData 解包成动作
                    act = agent.action_process(act_data[0])
                    predict_cnt += 1
                else:
                    # No need to predict
                    # 不需要预测的情况
                    agent.preprocess.update_traffic_info(obs, extra_info)
                    act = [None, None, None]

                # Interact with the environment, execute actions, get the next extra_info
                # 与环境交互, 执行动作, 获取下一步的状态, 如果遇到不需要预测的帧，则env.step直到得到需要预测的帧
                env_reward, env_obs = env.step(act)
                # Disaster recovery
                # 容灾
                if handle_disaster_recovery(env_obs, logger):
                    if len(collector) > 10:
                        collector = sample_process(collector)
                        yield collector
                    break

                frame_no = env_obs["frame_no"]
                _obs = env_obs["observation"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                extra_info = env_obs["extra_info"]
                logger.info(f"current frame_no is {frame_no}, predict_cnt is {predict_cnt}")

                # Determine if the environment is over
                # 判断环境结束
                done = terminated or truncated or (train_test_quick_stop and len(collector) > 1)
                if truncated:
                    logger.info(f"truncated is True, frame_no is {frame_no}, so this episode timeout")
                elif terminated:
                    logger.info(f"terminated is True, frame_no is {frame_no}, so this episode reach the end")

                # Save samples only when predicting
                # 只有预测步才保存样本
                if need_to_predict:
                    # Construct environment frames to prepare for sample construction
                    # 构造环境帧，为构造样本做准备
                    frame = Frame(
                        obs=obs_data.feature,
                        act=act,
                        rew=None,
                        done=0,
                        legal_action=obs["legal_action"],
                    )

                    collector.append(frame)

                # Status update
                # 状态更新
                obs = _obs
                if need_to_predict:
                    last_predict_act = act

                # Perform sample processing and return samples for training
                # 进行样本处理并将样本返回进行训练
                if done:
                    if len(collector) > 1:
                        # Calculate reward Rewards include phase_reward and duration_reward
                        # 奖励有phase_reward和duration_reward
                        reward = reward_shaping(_obs, last_predict_act, agent)
                        collector[-1].done = 1
                        collector[-1].rew = reward
                        collector = sample_process(collector)
                        yield collector
                    break

    except Exception as e:
        logger.error(f"run_episodes error")
        raise RuntimeError(f"run_episodes error")
