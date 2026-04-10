#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Gao Fang
@contact: gaofang@cetc.com.cn
@software: PyCharm
@file: dqn.py
@time: 2018/7/25 0025 13:38
@desc: 
"""
import os

import torch
import torch.nn as nn
import numpy as np


class NetFighter(nn.Module):
    def __init__(self, n_actions):
        super(NetFighter, self).__init__()
        # Compact CNN encoder with progressive down-sampling.
        self.conv = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=5, stride=2, padding=2),   # 50x50
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 25x25
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 13x13
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),# 7x7
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.info_fc = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
        )
        self.feature_fc = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        # Dueling heads: Q(s,a)=V(s)+A(s,a)-mean(A)
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, img, info):
        # Normalize image and info scales to stabilize optimization.
        img = img.float() / 255.0
        info = info.float()
        info_scale = info.new_tensor([360.0, 10.0, 10.0, 1500.0, 20.0, 180.0])
        info = info / info_scale

        img_feature = self.conv(img).view(img.size(0), -1)
        info_feature = self.info_fc(info)
        combined = torch.cat((img_feature, info_feature), dim=1)
        feature = self.feature_fc(combined)
        value = self.value_head(feature)
        advantage = self.adv_head(feature)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

# Deep Q Network off-policy
class RLFighter:
    def __init__(
            self,
            n_actions,
            learning_rate=0.001,
            reward_decay=0.95,
            e_greedy=1.0,
            replace_target_iter=100,
            memory_size=2000,
            batch_size=32,
            e_greedy_increment=-0.00005,
            model_dir='model/simple',
            double_dqn=True,
            grad_norm_clip=10.0,
            reward_clip=3000.0,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = float(e_greedy)
        self.model_dir = model_dir
        self.double_dqn = bool(double_dqn)
        self.grad_norm_clip = grad_norm_clip
        self.reward_clip = reward_clip
        self.s_screen_memory = []
        self.s_info_memory = []
        self.a_memory = []
        self.r_memory = []
        self.done_memory = []
        self.s__screen_memory = []
        self.s__info_memory = []
        self.memory_counter = 0
        self._memory_initialized = False
        self._memory_index = 0

        self.gpu_enable = torch.cuda.is_available()

        # total learning step
        self.learn_step_counter = 0

        self.cost_his = []
        self.eval_net, self.target_net = NetFighter(self.n_actions), NetFighter(self.n_actions)
        if self.gpu_enable:
            print('GPU Available!!')
            self.eval_net = self.eval_net.cuda()
            self.target_net = self.target_net.cuda()
        self.loss_func = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr, eps=1e-5)
        os.makedirs(self.model_dir, exist_ok=True)


    def _init_memory(self, s, s_):
        screen_shape = tuple(s['screen'].shape)
        info_shape = tuple(s['info'].shape)
        next_screen_shape = tuple(s_['screen'].shape)
        next_info_shape = tuple(s_['info'].shape)

        self.s_screen_memory = np.zeros((self.memory_size,) + screen_shape, dtype=np.float32)
        self.s_info_memory = np.zeros((self.memory_size,) + info_shape, dtype=np.float32)
        self.a_memory = np.zeros((self.memory_size,), dtype=np.int64)
        self.r_memory = np.zeros((self.memory_size,), dtype=np.float32)
        self.done_memory = np.zeros((self.memory_size,), dtype=np.float32)
        self.s__screen_memory = np.zeros((self.memory_size,) + next_screen_shape, dtype=np.float32)
        self.s__info_memory = np.zeros((self.memory_size,) + next_info_shape, dtype=np.float32)
        self._memory_initialized = True

    def store_transition(self, s, a, r, s_, done=False):
        if not self._memory_initialized:
            self._init_memory(s, s_)

        idx = self._memory_index
        self.s_screen_memory[idx] = s['screen']
        self.s_info_memory[idx] = s['info']
        self.a_memory[idx] = int(np.asarray(a).reshape(-1)[0])
        self.r_memory[idx] = float(r)
        self.done_memory[idx] = 1.0 if done else 0.0
        self.s__screen_memory[idx] = s_['screen']
        self.s__info_memory[idx] = s_['info']

        self._memory_index = (self._memory_index + 1) % self.memory_size
        self.memory_counter += 1

    def load(self,path):
        if self.gpu_enable:
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location='cpu')
        try:
            self.eval_net.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                'Checkpoint is incompatible with current DQN architecture. '
                'Use --fresh_start to train from scratch with the new model.'
            ) from exc

    def choose_action(self, img_obs, info_obs):
        actions = self.choose_action_batch([img_obs], [info_obs])
        action = np.zeros(1, dtype=np.int32)
        action[0] = int(actions[0])
        return action

    def choose_action_batch(self, img_obs_batch, info_obs_batch):
        """
        Batch epsilon-greedy action selection.
        :param img_obs_batch: list/array of [C,H,W]
        :param info_obs_batch: list/array of [F]
        :return: np.ndarray shape [B], dtype int32
        """
        if len(img_obs_batch) == 0:
            return np.empty((0,), dtype=np.int32)

        img_obs = torch.from_numpy(np.asarray(img_obs_batch, dtype=np.float32))
        info_obs = torch.from_numpy(np.asarray(info_obs_batch, dtype=np.float32))
        if self.gpu_enable:
            img_obs = img_obs.cuda(non_blocking=True)
            info_obs = info_obs.cuda(non_blocking=True)

        batch_size = img_obs.shape[0]
        random_mask = (np.random.uniform(size=batch_size) < self.epsilon)
        actions = np.random.randint(0, self.n_actions, size=batch_size, dtype=np.int32)

        greedy_indices = np.where(~random_mask)[0]
        if greedy_indices.size > 0:
            greedy_idx_tensor = torch.from_numpy(greedy_indices).long()
            if self.gpu_enable:
                greedy_idx_tensor = greedy_idx_tensor.cuda(non_blocking=True)
            prev_mode = self.eval_net.training
            self.eval_net.eval()
            with torch.no_grad():
                q_values = self.eval_net(
                    img_obs.index_select(0, greedy_idx_tensor),
                    info_obs.index_select(0, greedy_idx_tensor)
                )
                greedy_actions = torch.argmax(q_values, dim=1)
            self.eval_net.train(prev_mode)
            if self.gpu_enable:
                greedy_actions = greedy_actions.cpu()
            actions[greedy_indices] = greedy_actions.numpy().astype(np.int32, copy=False)

        return actions

    def learn(self):
        if not self._memory_initialized:
            return
        current_size = min(self.memory_counter, self.memory_size)
        if current_size == 0:
            return

        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')
            step_counter_str = '%09d' % self.learn_step_counter
            torch.save(self.target_net.state_dict(), os.path.join(self.model_dir, 'model_' + step_counter_str + '.pkl'))

        # Mini-batch sample from replay buffer.
        batch_size = min(self.batch_size, current_size)
        sample_index = np.random.choice(current_size, size=batch_size, replace=False)
        s_screen_mem = torch.from_numpy(self.s_screen_memory[sample_index])
        s_info_mem = torch.from_numpy(self.s_info_memory[sample_index])
        a_mem = torch.from_numpy(self.a_memory[sample_index]).long().view(batch_size, 1)
        r_mem = torch.from_numpy(self.r_memory[sample_index]).view(batch_size, 1)
        done_mem = torch.from_numpy(self.done_memory[sample_index]).view(batch_size, 1)
        s__screen_mem = torch.from_numpy(self.s__screen_memory[sample_index])
        s__info_mem = torch.from_numpy(self.s__info_memory[sample_index])
        if self.gpu_enable:
            s_screen_mem = s_screen_mem.cuda()
            s_info_mem = s_info_mem.cuda()
            a_mem = a_mem.cuda()
            r_mem = r_mem.cuda()
            done_mem = done_mem.cuda()
            s__screen_mem = s__screen_mem.cuda()
            s__info_mem = s__info_mem.cuda()

        if self.reward_clip is not None and self.reward_clip > 0:
            r_mem = torch.clamp(r_mem, min=-float(self.reward_clip), max=float(self.reward_clip))

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(s_screen_mem, s_info_mem).gather(1, a_mem)  # shape (batch, 1)
        with torch.no_grad():
            if self.double_dqn:
                next_act = self.eval_net(s__screen_mem, s__info_mem).argmax(dim=1, keepdim=True)
                q_next = self.target_net(s__screen_mem, s__info_mem).gather(1, next_act)
            else:
                q_next = self.target_net(s__screen_mem, s__info_mem).max(1)[0].view(batch_size, 1)
            q_target = r_mem + (1.0 - done_mem) * self.gamma * q_next
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_norm_clip is not None and self.grad_norm_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=float(self.grad_norm_clip))
        self.optimizer.step()

        self.cost_his.append(loss.item())

        # Decrease epsilon during training and clamp into [epsilon_min, epsilon_max].
        self.epsilon = self.epsilon + self.epsilon_increment
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
        elif self.epsilon > self.epsilon_max:
            self.epsilon = self.epsilon_max
        if self.learn_step_counter % 100 == 0:
            print(f'epsilon={self.epsilon:.5f}')
        self.learn_step_counter += 1


class NetDetector(nn.Module):
    def __init__(self, n_actions):
        super(NetDetector, self).__init__()
        self.conv1 = nn.Sequential(     # 100 * 100 * 3
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(     # 50 * 50 * 16
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 25 * 25 * 32
        )
        self.info_fc = nn.Sequential(
            nn.Linear(3, 256),
            nn.Tanh(),
        )
        self.feature_fc = nn.Sequential(    # 25 * 25 * 32 + 256
            nn.Linear((25 * 25 * 32 + 256), 512),
            nn.ReLU(),
        )
        self.decision_fc = nn.Linear(512, n_actions)

    def forward(self, img, info):
        img_feature = self.conv1(img)
        img_feature = self.conv2(img_feature)
        info_feature = self.info_fc(info)
        combined = torch.cat((img_feature.view(img_feature.size(0), -1), info_feature.view(info_feature.size(0), -1)),
                             dim=1)
        feature = self.feature_fc(combined)
        action = self.decision_fc(feature)
        return action
