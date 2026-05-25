#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from kaiwudrl.interface.agent import BaseAgent
from agent_ppo.model.model import Model
from agent_ppo.feature.definition import *
from agent_ppo.conf.conf import Config
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.feature.preprocessor import FeatureProcess
from torch.distributions import Categorical


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        initial_lr = Config.INIT_LEARNING_RATE_START
        parameters = self.model.parameters()

        # ========== TODO 17 ==========
        # Compare and choose a suitable PPO optimizer.
        # Hint: Try RMSprop, Adam, or SGD together with INIT_LEARNING_RATE_START tuning.
        # 比较并选择合适的 PPO 优化器。
        # 提示：可尝试 RMSprop、Adam 或 SGD，并结合 INIT_LEARNING_RATE_START 调整。
        self.optimizer = torch.optim.Adam(params=parameters, lr=initial_lr, eps=1e-5)
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.logger = logger
        self.monitor = monitor
        self.preprocess = FeatureProcess(logger)
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, self.logger, self.monitor)
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs):
        self.preprocess.reset()

    def __predict_detail(self, list_obs_data, exploit_flag=False):
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        self.model.set_eval_mode()

        s = torch.tensor(feature).view(1, Config.DIM_OF_OBSERVATION).float().to(self.device)
        with torch.no_grad():
            output_list = self.model(s, inference=True)

        np_output = []
        for output in output_list:
            np_output.append(output.numpy())

        logits, value = np_output[:2]

        list_act_data = list()
        for i in range(len(legal_action)):
            prob, action, d_action = self._sample_masked_action(logits[i], np.array(legal_action[i], dtype=np.float32))
            list_act_data.append(ActData(junction_id=0, action=action, d_action=d_action, prob=prob, value=value))
        return list_act_data

    def predict(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=False)

    def exploit(self, observation):
        obs_data = self.observation_process(observation["obs"], observation["extra_info"])
        if not obs_data:
            return [[None, None, None]]
        act_data = self.__predict_detail([obs_data], exploit_flag=True)
        act = self.action_process(act_data[0], False)
        return act

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict = self.model.state_dict()
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)

        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.model.device))

        self.logger.info(f"load model {model_file_path} successfully")

    def observation_process(self, raw_obs, extra_info=None):
        """
        This function is an important function for feature processing, mainly responsible for:
            - Parsing raw data from proto data
            - Calculating features through raw data to obtain multiple feature vectors
            - Concatenation of features
            - Labeling of legal actions

        Args:
            - raw_obs: Raw feature data sent by battlesrv

        Returns:
            - ObsData: A variable containing observation, legal_action and sub_action_mask
        """
        """
            该函数是特征处理的重要函数, 主要负责：
                - 从 proto 数据中解析原始数据
                - 通过原始数据计算特征, 得到多个特征向量
                - 特征的拼接
                - 合法动作的标注

            参数：
                - raw_obs: battlesrv 发送的原始特征数据

            返回：
                - ObsData: 包含 observation, legal_action 与 sub_action_mask 的变量
        """

        # User-defined section, can record or update traffic information per frame.
        # 用户自定义部分, 可每帧对交通信息进行记录或更新
        self.preprocess.update_traffic_info(raw_obs, extra_info)

        # Note: The unpacking of the following raw data is for example purposes only,
        # please modify according to the actual situation
        # 注意: 以下原始数据的解包为示例, 请根据实际情况修改
        frame_state = raw_obs["frame_state"]

        # Parse frame_state
        # 解析 frame_state
        _, _, vehicles = (
            frame_state["frame_no"],
            frame_state["frame_time"],
            frame_state["vehicles"],
        )

        # Divide the lane into several grids along the lane direction and the vehicle driving direction
        # 沿车道方向和车辆行驶方向将车道划分为数个栅格
        speed_dict = {}
        position_dict = {}
        for junction_id in self.preprocess.junction_dict.keys():
            speed_dict[junction_id] = np.zeros((Config.GRID_WIDTH, Config.GRID_NUM))
            position_dict[junction_id] = np.zeros((Config.GRID_WIDTH, Config.GRID_NUM))

        # The default value of junction_id in a single intersection scenario is 0
        # 单交叉口场景junction_id默认为0
        junction_id = 0

        # Initialize state-related variables to prevent errors when there are no vehicles in the traffic scenario
        # 初始化状态相关变量, 防止交通场景内车辆为空时报错
        position = list(position_dict[junction_id].astype(int).flatten())
        speed = list(speed_dict[junction_id].flatten())

        for vehicle in vehicles:
            # Only count vehicles on the enter lane
            # 仅统计位于进口车道上的车辆信息
            if on_enter_lane(vehicle):
                # Convert the vehicle x,y coordinates to grid coordinates. Here,
                # get_lane_code maps the lane number to integers 0-13, corresponding to 14 import lanes
                # 将车辆x,y坐标转化为栅格坐标, 此处get_lane_code将车道编号映射至整数0-13, 对应14条进口车道
                x_pos = get_lane_code(vehicle)
                y_pos = int((vehicle["position_in_lane"]["y"] / 1) // Config.GRID_LENGTH)

                if y_pos >= Config.GRID_NUM:
                    continue

                speed_dict[vehicle["target_junction"]][x_pos, y_pos] = float(
                    vehicle["speed"] / self.preprocess.vehicle_configs_dict[vehicle["v_config_id"]]["max_speed"]
                )
                position_dict[vehicle["target_junction"]][x_pos, y_pos] = 1
            else:
                continue

        position = list(position_dict[junction_id].astype(int).flatten())
        speed = list(speed_dict[junction_id].flatten())

        # Integrate all state quantities into the observation
        # 将所有状态量整合在observation中
        observation = position + speed

        return ObsData(
            feature=observation,
            legal_action=[
                1,
            ]
            * (Config.DIM_OF_ACTION_PHASE_1 + Config.DIM_OF_ACTION_DURATION_1),
            sub_action_mask=[
                1,
            ]
            * Config.NUMB_HEAD,
        )

    def action_process(self, act_data, is_stochastic=True):
        junction_id = act_data.junction_id
        if is_stochastic:
            action = act_data.action
        else:
            action = act_data.d_action

        action_p = action[0]
        action_d = (action[1] + 1) * 5

        return [junction_id, action_p, action_d]

    def _sample_masked_action(self, logits, legal_action):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional []

        从预测的logits和合法动作中采样动作
        返回: 概率、随机动作和确定性动作（包含额外的[]）
        """
        prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        for index in range(0, len(self.label_size_list)):
            # Count non-zero elements. If all actions are illegal, then True
            # 统计非0元素数量，若全部动作都非法，则True
            if np.count_nonzero(legal_actions[index]) == 0:  # np.sum(~np.isnan(probs)) == 0:
                probs = [
                    0,
                ] * self.label_size_list[index]
                sample_action = 0
                d_action = 0
            else:
                probs = self._legal_soft_max(logits_split[index], legal_actions[index])
                sample_action = self._legal_sample(probs, use_max=False)
                d_action = self._legal_sample(probs, use_max=True)
            action_list.append(sample_action)
            d_action_list.append(d_action)
            prob_list += list(probs)

        return prob_list, action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        # Large and small constants for numerical stability
        # 用于数值稳定性的大小常量
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        # Not necessary max clip 1
        # 最大值裁剪1不是必需的
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        """
        Sample with probability, input probs should be 1D array

        按概率采样，输入的probs应该是一维数组
        """
        if use_max:
            return np.argmax(probs)

        return np.argmax(np.random.multinomial(1, probs, size=1))
