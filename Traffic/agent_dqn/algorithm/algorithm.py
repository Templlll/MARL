#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import os
import time
import numpy as np
from agent_dqn.conf.conf import Config
from copy import deepcopy
import torch.nn.functional as F


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model

        self.optim = optimizer
        self.logger = logger
        self.monitor = monitor

        self.num_head = Config.NUMB_HEAD
        self._gamma = Config.GAMMA

        self.target_model = deepcopy(self.model)
        self.target_model.to(self.device)
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad_(False)
        self.update_target_q(force=True)

        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        # Convert list of SampleData to tensor batch
        # 将 SampleData 数组 转换为 tensor batch
        obs = torch.stack([frame.obs for frame in list_sample_data]).to(self.device)
        action = torch.stack(
            [frame.act if not any(np.isinf(frame.act)) else [0] * len(frame.act) for frame in list_sample_data]
        ).to(self.device)
        rew = torch.stack([frame.rew for frame in list_sample_data]).to(self.device)
        _obs = torch.stack([frame._obs for frame in list_sample_data]).to(self.device)
        not_done = torch.stack([frame.done for frame in list_sample_data]).to(self.device)

        # Main implementation of the multi-head output DQN algorithm
        # 多头输出dqn算法的主要实现
        self.target_model.eval()

        with torch.no_grad():
            # Calculate the target Q-values for each head
            # 计算各个头的目标q值
            q_targets = []
            for head_idx in range(self.num_head):
                q_targets_head = (
                    rew[:, head_idx].unsqueeze(1)
                    + self._gamma
                    * (self.target_model(_obs)[0][head_idx]).max(1)[0].unsqueeze(1)
                    * not_done[:, None]
                )
                q_targets.append(q_targets_head)
            q_targets = torch.cat(q_targets, dim=1)

        # Calculate the Q-values for each head
        # 计算各个头的q值
        self.model.train()
        q_values = []
        for head_idx in range(self.num_head):
            q_values_head = self.model(obs)[0][head_idx].gather(1, action[:, head_idx + 1].long().unsqueeze(1))
            q_values.append(q_values_head)
        q_values = torch.cat(q_values, dim=1)

        self.optim.zero_grad()
        loss = F.mse_loss(q_targets.float(), q_values.float())
        loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
        self.optim.step()
        self.train_step += 1

        if Config.TARGET_UPDATE_FREQ and (self.train_step % int(Config.TARGET_UPDATE_FREQ) == 0):
            self.update_target_q()

        value_loss = loss.detach().item()
        target_q_value = q_targets.mean().detach().item()
        q_value = q_values.mean().detach().item()

        # Periodically report monitoring
        # 按照间隔上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "value_loss": value_loss,
                "target_q_value": target_q_value,
                "q_value": q_value,
                "model_grad_norm": model_grad_norm,
            }
            self.monitor.put_data({os.getpid(): monitor_data})
            self.logger.info(
                f"value_loss: {value_loss}, target_q_value: {target_q_value},\
                                q_value: {q_value},\
                                model_grad_norm: {model_grad_norm}"
            )
            self.last_report_monitor_time = now

    def update_target_q(self, force: bool = False):
        if self.target_model is self.model:
            if not force:
                return
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad_(False)
