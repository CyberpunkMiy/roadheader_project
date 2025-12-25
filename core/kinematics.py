import mujoco
import numpy as np
from typing import Tuple, Optional

class RoadheaderIK:
    """
    掘进机逆运动学求解器 (Inverse Kinematics)
    基于 MuJoCo 的 Jacobian 矩阵进行数值迭代求解 (Damped Least Squares).
    """
    def __init__(self, model, data, site_name="cutter_tip"):
        self.model = model
        self.data = data
        
        # 获取截割头尖端的 site ID
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id == -1:
            raise ValueError(f"XML中未找到 site: {site_name}")

        # 获取需要参与 IK 计算的关节 ID 列表 (排除履带等无关关节)
        # 这里假设我们要控制的是: act_left_right (左右), act_up_down (升降), act_front_back (伸缩)
        # 注意：你需要根据实际关节名字修改这里
        joint_names = ["gangti_self_joint_left", "huosai_front_joint", "jiege_shengsuo_joint"]
        self.dof_ids = []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid != -1:
                # 获取该关节在 qpos 数组中的地址
                qpos_addr = model.jnt_qposadr[jid]
                self.dof_ids.append(qpos_addr)
        
        self.dof_indices = np.array(self.dof_ids, dtype=int)

    def solve(self, target_pos: np.ndarray, max_steps=50, tol=0.01) -> Optional[np.ndarray]:
        """
        输入: 目标坐标 [x, y, z]
        输出: 对应的驱动器控制值 (ctrl)
        """
        # 1. 备份当前状态 (因为计算 IK 会移动机器人，算完要复原)
        qpos_backup = self.data.qpos.copy()
        
        success = False
        
        # 2. 迭代求解 (Newton-Raphson method)
        for _ in range(max_steps):
            # 获取当前末端位置
            current_pos = self.data.site_xpos[self.site_id]
            print(f"Current Pos: {current_pos}, Target Pos: {target_pos}")
            
            # 计算误差 (Error)
            error = target_pos - current_pos
            
            # 如果误差足够小，说明找到了
            if np.linalg.norm(error) < tol:
                success = True
                break

            # 计算雅可比矩阵 (Jacobian)
            # J 的形状通常是 (3, nv) -> (3, 自由度数量)
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site_id)
            
            # 只取我们关心的那几个关节的 Jacobian 列
            # 假设我们只控制特定的几个关节
            j_reduced = jacp[:, self.dof_indices]

            # 求解关节增量: delta_q = J_inverse * error
            # 使用伪逆 (Pseudo-Inverse) 解决奇异性问题
            delta_q = np.linalg.pinv(j_reduced) @ error
            
            # 更新关节角度
            self.data.qpos[self.dof_indices] += delta_q * 0.5 # 0.5 是步长，防震荡
            
            # 必须调用这个来更新几何体位置，否则下一轮 Jacobian 是错的
            mujoco.mj_kinematics(self.model, self.data)

        # 3. 获取解出来的目标 Joint Positions
        target_qpos = self.data.qpos.copy()
        
        # 4. [关键一步] 将 Joint Positions 转换为 Actuator Control
        # 很多时候，液压缸长度 != 关节角度，需要 MuJoCo 帮我们算
        mujoco.mj_kinematics(self.model, self.data) # 确保状态是最新的
        mujoco.mj_transmission(self.model, self.data) # 计算驱动器长度
        
        # 获取此时的驱动器长度作为目标控制值
        # 注意：这里假设你的 actuators 和关节是一一对应的，如果不对应需要按 ID 映射
        # 简单起见，我们提取所有相关 actuator 的长度
        target_ctrl = self.data.actuator_length.copy()
        
        # 5. 恢复机器人到原来的状态 (不影响仿真物理进程)
        self.data.qpos[:] = qpos_backup
        mujoco.mj_kinematics(self.model, self.data)
        
        if success:
            return target_ctrl # 返回计算出的控制信号
        else:
            return None # 求解失败（目标太远够不着）

if __name__ == "__main__":
    import os
    # 简单的测试代码
    # 假设 xml 文件在项目根目录，而当前脚本在 core/ 目录下
    xml_path = "../myrobot.xml" 
    
    if not os.path.exists(xml_path):
        # 如果直接在 core 目录下运行，尝试找上一级
        # 如果在根目录运行 (python core/kinematics.py)，路径可能需要调整
        # 这里做一个简单的路径检查
        if os.path.exists("myrobot.xml"):
            xml_path = "myrobot.xml"
        else:
            print(f"Warning: XML file not found at {xml_path} or myrobot.xml")

    try:
        print(f"Loading model from {xml_path}...")
        m = mujoco.MjModel.from_xml_path(xml_path)
        d = mujoco.MjData(m)
        
        # 初始化 IK 求解器
        # 注意：确保 xml 中包含 site_name="cutter_tip" 以及代码中硬编码的关节名称
        ik_solver = RoadheaderIK(m, d, site_name="cutter_tip")
        print("IK Solver initialized.")
        
        # 进行一次正向运动学计算，获取当前末端位置
        mujoco.mj_kinematics(m, d)
        current_tip = d.site_xpos[ik_solver.site_id].copy()
        print(f"Current Tip Position: {current_tip}")
        
        # 定义一个测试目标点：在当前位置基础上，X轴向前移动 0.2 米
        target = current_tip + np.array([0.01, 0.0, 0.0]) 
        print(f"Target Position: {target}")
        
        # 求解
        ctrl = ik_solver.solve(target)
        
        if ctrl is not None:
            print("IK Solution Found!")
            print(f"Target Control Signals (Actuator Lengths): {ctrl}")
            # 注意：这里返回的是所有 actuator 的长度，你需要根据 ID 对应到具体的驱动器
        else:
            print("IK Solver failed to find a solution (target might be out of reach).")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()