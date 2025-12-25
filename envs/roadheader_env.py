import mujoco
import mujoco.viewer
import numpy as np
from typing import Tuple

# 引入我们刚才定义的 Robot 类
from core import RoadheaderModel

class RoadheaderEnv:
    """
    符合 RL 标准的仿真环境。
    主要方法: reset(), step(action), render()
    """
    def __init__(self, xml_path: str):
        self.robot = RoadheaderModel(xml_path)
        self.model = self.robot.model
        self.data = self.robot.data
        
        self.viewer = None
        self.dt = self.model.opt.timestep

    def reset(self):
        """重置环境到初始状态"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """环境推演一步"""
        # 1. 动作映射: [-1, 1] -> 物理单位
        self._apply_action(action)
        
        # 2. 物理步进
        mujoco.mj_step(self.model, self.data)
        
        # 3. 获取反馈
        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        done = False
        info = {}
        
        return obs, reward, done, info

    def _apply_action(self, action: np.ndarray):
        """将归一化的动作 [-1, 1] 转换为实际控制信号"""
        if len(action) != self.robot.n_actions:
            # 简单的防错，实际中可能需要 log warning
            return 

        idx = 0
        for name, info in self.robot.actuators.items():
            low, high = info.ctrl_range
            center = (high + low) / 2.0
            span = (high - low) / 2.0
            phys_ctrl = center + action[idx] * span
            
            self.data.ctrl[info.id] = phys_ctrl
            idx += 1
            
        self._set_rotators(10.0)

    def _set_rotators(self, speed):
        try:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_cutter_spin")
            if aid != -1: self.data.ctrl[aid] = speed
        except: pass

    def _get_obs(self) -> np.ndarray:
        """获取观测向量 (State)"""
        # 这里用到了 np.concatenate 将位置和速度拼成一个长向量
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def _compute_reward(self, obs, action) -> float:
        """奖励函数设计"""
        return 0.0

    def render(self):
        """渲染画面"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -20
            self.viewer.cam.distance = 8.0
            self.viewer.cam.lookat[:] = [0, 0, 1.0]
        
        self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()