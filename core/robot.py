import mujoco
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class ActuatorInfo:
    id: int
    name: str
    ctrl_range: Tuple[float, float]

class RoadheaderModel:
    """
    负责底层 MuJoCo 模型的加载、驱动器信息的解析。
    不包含业务逻辑，只负责“硬件”接口。
    """
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
        except ValueError:
            raise FileNotFoundError(f"无法加载模型: {xml_path}")
        
        # 解析驱动器信息
        self.actuators = self._parse_actuators()
        self.n_actions = len(self.actuators)

    def _parse_actuators(self) -> Dict[str, ActuatorInfo]:
        """自动读取 XML 中的 ctrlrange，为 RL 做归一化准备"""
        # 请确保 xml 中有这些名字，或者根据实际情况修改列表
        target_names = ["act_left_right", "act_up_down", "act_front_back", "act_shovel"]
        infos = {}
        for name in target_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid != -1:
                cr = self.model.actuator_ctrlrange[aid]
                infos[name] = ActuatorInfo(id=aid, name=name, ctrl_range=(cr[0], cr[1]))
        return infos
    
if __name__ == "__main__":
    # 测试代码
    model = RoadheaderModel("../myrobot.xml")
    for name, info in model.actuators.items():
        print(f"Actuator: {name}, ID: {info.id}, Control Range: {info.ctrl_range}")