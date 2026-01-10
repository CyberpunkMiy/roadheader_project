import mujoco
import numpy as np
from typing import Tuple, Optional

class RoadheaderIK:
    """
    掘进机逆运动学求解器 (Inverse Kinematics)
    """
    def __init__(self, model, data, site_name="cutter_tip"):
        self.model = model
        self.data = data
        
        # 1. [修改] 必须先运行一次前向动力学，确保所有关节轴向(xaxis)和几何关系被初始化
        #    如果没有这行，mj_jacSite 读到的初始数据可能是 0
        mujoco.mj_forward(self.model, self.data)

        # 获取截割头尖端的 site ID
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id == -1:
            raise ValueError(f"XML中未找到 site: {site_name}")

        # 关节列表 (根据 debug_kinematics 确认的名称)
        joint_names = ["gangti_self_joint_left", "huosai_front_joint", "jiege_shengsuo_joint"]
        
        # 2. [修改] 分别存储 qpos_adr (用于改数值) 和 dof_adr (用于查 Jacobian)
        self.qpos_indices = [] # 对应 data.qpos
        self.dof_indices = []  # 对应 Jacobian 的列
        
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid != -1:
                # 记录位置地址 (修改 qpos 用)
                self.qpos_indices.append(model.jnt_qposadr[jid])
                # 记录自由度地址 (切片 Jacobian 用)
                self.dof_indices.append(model.jnt_dofadr[jid])
            else:
                print(f"Warning: Joint {name} not found in model.")
        
        self.qpos_indices = np.array(self.qpos_indices, dtype=int)
        self.dof_indices = np.array(self.dof_indices, dtype=int)

    def solve(self, target_pos: np.ndarray, max_steps=50, tol=0.01) -> Optional[np.ndarray]:
        """
        输入: 目标坐标 [x, y, z]
        输出: 对应的驱动器控制值 (ctrl)
        """
        # 备份当前状态
        qpos_backup = self.data.qpos.copy()
        success = False
        
        # 迭代求解 (Newton-Raphson)
        for i in range(max_steps):
            # 获取当前末端位置
            current_pos = self.data.site_xpos[self.site_id]
            error = target_pos - current_pos
            
            # 判断误差
            if np.linalg.norm(error) < tol:
                success = True
                break

            # 计算雅可比矩阵
            jacp = np.zeros((3, self.model.nv))
            # jacr = np.zeros((3, self.model.nv)) # 如果需要姿态控制则启用
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site_id)
            
            # 3. [修改] 使用 dof_indices 切片 Jacobian
            j_reduced = jacp[:, self.dof_indices]

            # 求解关节增量: delta_q = J_pinv * error
            # 增加阻尼项 (damping) 0.01 避免奇异值爆炸
            n_dof = len(self.dof_indices)
            lambda_dls = 0.01
            # DLS 公式: J^T * (J * J^T + lambda^2 * I)^-1 * error
            # 或者直接用 np.linalg.pinv (简单粗暴，通常够用)
            delta_q = np.linalg.pinv(j_reduced, rcond=1e-2) @ error
            
            # 4. [修改] 使用 qpos_indices 更新关节角度
            # 步长 0.2 比较保守，防止并在闭链结构中震荡
            self.data.qpos[self.qpos_indices] += delta_q * 0.2
            
            # 5. [修改] 关键点：这里用 mj_forward 而不是 mj_kinematics
            #    因为你的模型有很多 equality constraints (闭链)，
            #    forward 会计算约束力并修正位置，让物理状态更合法
            mujoco.mj_forward(self.model, self.data)

        # 提取结果
        target_ctrl = None
        if success:
            print(f"IK Converged in {i} steps.")
            # 计算传动，获取 actuator 的长度
            mujoco.mj_transmission(self.model, self.data)
            
            # 注意：这里简单粗暴返回了所有 actuators 的长度
            # 实际项目中，你可能需要根据 actuator 的名字来筛选对应的 4 个驱动器
            target_ctrl = self.data.actuator_length.copy()
        else:
            print("IK Failed to converge.")

        # 恢复状态 (如果不希望 IK 改变当前仿真画面的话)
        self.data.qpos[:] = qpos_backup
        mujoco.mj_forward(self.model, self.data)
        
        return target_ctrl

if __name__ == "__main__":
    import os
    import time
    import mujoco.viewer  # 引入渲染器

    xml_path = "../myrobot.xml" 
    
    if os.path.exists(xml_path):
        print(f"Loading model from {xml_path}...")
        m = mujoco.MjModel.from_xml_path(xml_path)
        d = mujoco.MjData(m)
        
        # 1. 初始刷新
        mujoco.mj_forward(m, d)
        
        # 初始化 IK
        ik_solver = RoadheaderIK(m, d, site_name="cutter_tip")
        
        # 获取起点
        current_tip = d.site_xpos[ik_solver.site_id].copy()
        
        # 设定目标：向上抬 0.1 米，向前伸 0.1 米
        target = current_tip + np.array([-0.5, -0.3, -0.5]) 
        
        print("计算 IK 中...")
        ctrl = ik_solver.solve(target)
        
        if ctrl is not None:
            print("✅ IK 成功! 准备演示动画...")
            
            # === 核心修改：打开可视化窗口 ===
            # launch_passive 适合在脚本中弹窗查看
            with mujoco.viewer.launch_passive(m, d) as viewer:
                # 设置摄像机位置 (根据你的模型大小调整)
                viewer.cam.lookat[:] = [0, 0, 1]   # 镜头焦点：看向坐标 [0, 0, 1] (大概是机身中心)
                viewer.cam.distance = 8.0          # 镜头距离：拉远到 8 米 (根据你的模型大小调整)
                viewer.cam.elevation = -30         # 镜头仰角：-30度 (俯视)
                viewer.cam.azimuth = 135           # 镜头水平角：135度 (侧后方视角)
                
                # 1. 先展示一下起始位置 (停顿 1 秒)
                start_time = time.time()
                while time.time() - start_time < 1.0:
                    viewer.sync()
                    time.sleep(0.01)
                
                print("开始移动到目标位置...")
                
                # 2. 应用计算出的控制量
                # 注意：假设前 4 个是位置驱动器 (Position Actuators)
                # 如果你的 act_left_right, act_up_down, act_front_back, act_shovel 分别是 0,1,2,3
                d.ctrl[0] = ctrl[0]
                d.ctrl[1] = ctrl[1]
                d.ctrl[2] = ctrl[2]
                d.ctrl[3] = ctrl[3]
                
                # 3. 在循环中执行物理仿真，让大家看到它慢慢动过去
                # 运行 3000 步 (3秒)，让 PID 控制器有时间把关节推到位
                for _ in range(3000):
                    mujoco.mj_step(m, d)
                    viewer.sync() # 刷新画面
                    
                    # 稍微加点延时，不然电脑太快一瞬间就结束了
                    # time.sleep(0.001) 
                
                print(f"移动完成。实际末端位置: {d.site_xpos[ik_solver.site_id]}")
                print(f"目标位置: {target}")
                
                # 4. 保持窗口打开，直到你手动关闭
                print("演示结束，请手动关闭窗口。")
                while viewer.is_running():
                    time.sleep(0.1)
                    
        else:
            print("❌ IK 求解失败")
    else:
        print(f"Error: {xml_path} not found.")