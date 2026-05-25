#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import Frame
from tools.train_env_conf_validate import read_usr_conf
from agent_diy.feature.definition import (
    sample_process,
)
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env, agent = envs[0], agents[0]

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error(f"usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    # Retrieving training metrics
    # 获取训练中的指标
    training_metrics = get_training_metrics()
    if training_metrics:
        logger.info(f"training_metrics is {training_metrics}")

    # Please write your DIY training process below.
    # 请在下方写你DIY的训练流程

    # At the start of each environment, support loading the latest model file
    # 每次对局开始时, 支持加载最新model文件, 该调用会从远程的训练节点加载最新模型
    agent.load_model(id="latest")

    # model saving
    # 保存模型
    agent.save_model()

    return
