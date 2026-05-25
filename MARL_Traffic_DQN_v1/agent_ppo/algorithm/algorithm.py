###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import torch.nn as nn
import os
import time
import numpy as np
from agent_ppo.conf.conf import Config
import torch.nn.functional as F


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model

        self.optimizer = optimizer
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.logger = logger
        self.monitor = monitor

        self.num_head = Config.NUMB_HEAD
        self._gamma = Config.GAMMA

        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.m_var_beta = Config.BETA_START
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.var_beta = self.m_var_beta

        self.last_report_monitor_time = 0
        self.train_step = 0

    def learn(self, list_sample_data):
        # Convert list of SampleData to tensor batch
        # 将 SampleData 数组 转换为 tensor batch
        batch_size = len(list_sample_data)

        obs = torch.stack([frame.obs for frame in list_sample_data]).to(self.model.device)
        legal_action = torch.stack([frame.legal_action for frame in list_sample_data]).to(self.model.device)
        sub_action = torch.stack([frame.sub_action for frame in list_sample_data]).to(self.model.device)
        act = torch.stack([frame.act for frame in list_sample_data]).to(self.model.device)
        prob = torch.stack([frame.prob for frame in list_sample_data]).to(self.model.device)
        reward = torch.stack([frame.reward for frame in list_sample_data]).to(self.model.device)
        reward_sum = torch.stack([frame.reward_sum for frame in list_sample_data]).to(self.model.device)
        advantage = torch.stack([frame.advantage for frame in list_sample_data]).to(self.model.device)
        value = torch.stack([frame.value for frame in list_sample_data]).to(self.model.device)
        next_value = torch.stack([frame.next_value for frame in list_sample_data]).to(self.model.device)
        is_train = torch.stack([frame.is_train for frame in list_sample_data]).to(self.model.device)

        data_list = [
            obs,
            legal_action,
            sub_action,
            act,
            prob,
            reward,
            reward_sum,
            advantage,
            value,
            next_value,
            is_train,
        ]

        # Configure model before prediction
        # 预测前先对model进行设置
        self.model.set_train_mode()
        self.optimizer.zero_grad()

        rst_list = self.model(obs)
        total_loss, info_list = self.calculate_loss(data_list, rst_list)
        results = {}

        results["total_loss"] = total_loss.item()

        total_loss.backward()

        # Gradient clipping
        # 梯度裁剪
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

        self.optimizer.step()
        self.train_step += 1

        _info_list = []
        for info in info_list:
            if isinstance(info, list):
                _info = [i.item() for i in info]
            else:
                _info = info.item()
            _info_list.append(_info)

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            _, (value_loss, policy_loss, entropy_loss) = _info_list
            results["value_loss"] = round(value_loss, 2)
            results["policy_loss"] = round(policy_loss, 2)
            results["entropy_loss"] = round(entropy_loss, 2)

            self.logger.info(
                f"policy_loss: {round(policy_loss, 2)}, value_loss: {round(value_loss, 2)}, entropy_loss: {round(entropy_loss, 2)}"
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})

            self.last_report_monitor_time = now

    def calculate_loss(self, list_sample_data, model_output_data):
        (
            obs,
            legal_action,
            sub_action,
            act,
            prob,
            reward,
            reward_sum,
            advantage,
            value,
            next_value,
            is_train,
        ) = list_sample_data

        reward = reward_sum * Config.VALUE_COEF + reward * (1 - Config.VALUE_COEF)
        legal_action_flag_list = torch.split(legal_action, self.label_size_list, dim=1)
        usq_label_list = list()
        for shape_index in range(len(self.label_size_list)):
            usq_label_list.append(act[:, shape_index])
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = usq_label_list[shape_index].reshape(-1, 1).long()
        # Process probability
        # 处理概率
        sum_ls_list = [sum(self.label_size_list[0:i]) for i in range(len(self.label_size_list))]

        old_label_probability_list = list()
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list.append(
                prob[:, sum_ls_list[shape_index] : sum_ls_list[shape_index] + self.label_size_list[shape_index]]
            )
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.label_size_list[shape_index]
            )
        usq_weight_list = list()
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list.append(sub_action[:, shape_index])
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(-1, 1)

        label_list = []
        for ele in usq_label_list:
            label_list.append(ele.squeeze(dim=1))
        weight_list = []
        for weight in usq_weight_list:
            weight_list.append(weight.squeeze(dim=1))

        label_result = model_output_data[:-1]

        value_result = model_output_data[-1]

        # Loss of value network
        # 价值网络损失
        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        self.value_cost = 0.5 * torch.mean(torch.square(reward - fc2_value_result_squeezed), dim=0)

        # ========== TODO 6 ==========
        # Implement the PPO policy loss (clip loss).
        # Hint: Compute the probability ratio, clipping term, and advantage-weighted loss.
        # 实现 PPO 策略损失（clip loss）。
        # 提示：需要计算概率比、裁剪项以及基于 advantage 的加权损失。
        advantage = advantage.squeeze(dim=1) if advantage.dim() > 1 else advantage
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        advantage = advantage.detach()

        new_log_prob = 0.0
        old_log_prob = 0.0
        entropy_list = []

        for head_idx, head_logits in enumerate(label_result):
            head_legal_action = legal_action_flag_list[head_idx].float()
            if torch.count_nonzero(head_legal_action, dim=1).min().item() == 0:
                head_legal_action = torch.where(
                    torch.sum(head_legal_action, dim=1, keepdim=True) > 0,
                    head_legal_action,
                    torch.ones_like(head_legal_action),
                )

            masked_logits = head_logits.masked_fill(head_legal_action <= 0, -1e9)
            current_prob = torch.softmax(masked_logits, dim=1)
            old_prob = torch.clamp(old_label_probability_list[head_idx], min=self.min_policy)

            chosen_label = usq_label_list[head_idx]
            current_selected_prob = torch.gather(current_prob, 1, chosen_label)
            old_selected_prob = torch.gather(old_prob, 1, chosen_label)

            new_log_prob = new_log_prob + torch.log(current_selected_prob)
            old_log_prob = old_log_prob + torch.log(old_selected_prob)

            entropy_list.append(-torch.sum(current_prob * torch.log(current_prob.clamp_min(self.min_policy)), dim=1))

        ratio = torch.exp(new_log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_1 = ratio.squeeze(dim=1) * advantage
        surrogate_2 = clipped_ratio.squeeze(dim=1) * advantage
        self.policy_cost = -torch.mean(torch.min(surrogate_1, surrogate_2), dim=0)

        # ========== TODO 7 ==========
        # Implement the PPO entropy loss.
        # Hint: Compute entropy from the action distribution to encourage exploration.
        # 实现 PPO 熵损失。
        # 提示：可根据动作概率分布计算熵，用于鼓励探索。
        self.entropy_cost = torch.mean(torch.stack(entropy_list, dim=0)) if entropy_list else torch.tensor(0.0, device=self.device)

        # ========== TODO 8 ==========
        # Combine the final loss.
        # Hint: You can weight value_cost, policy_cost, and entropy regularization together.
        # 组合总损失。
        # 提示：可按 value_cost、policy_cost 和熵正则项加权求和。
        self.loss = self.value_cost + self.policy_cost - self.var_beta * self.entropy_cost

        return self.loss, [
            self.loss,
            [self.value_cost, self.policy_cost, self.entropy_cost],
        ]
