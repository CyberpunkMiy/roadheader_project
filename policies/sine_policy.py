import math
import numpy as np

class SineWavePolicy:
    """
    一个“假”的智能体，使用正弦波输出动作 [-1, 1]。
    用于测试环境动力学。
    """
    def __init__(self, freq=0.5):
        self.freq = freq
        
    def predict(self, obs, t) -> np.ndarray:
        """
        输入: 观测, 时间
        输出: 动作 [-1, 1] 之间
        """
        val = math.sin(2 * math.pi * self.freq * t)
        
        # 假设有4个主要动作，加上相位差让它动得更有机一点
        actions = np.array([
            val,                                    # Act 1
            math.sin(2 * math.pi * 0.5 * t + 1.0),  # Act 2
            math.sin(2 * math.pi * 0.2 * t),        # Act 3
            math.sin(2 * math.pi * 1.0 * t)         # Act 4
        ])
        
        return np.clip(actions, -1.0, 1.0)