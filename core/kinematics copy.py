import mujoco
import numpy as np
from typing import Tuple, Optional, List

class RoadheaderIK:
    """
    针对 EBZ260H 掘进机的逆运动学求解器 (修复版)
    包含：自动驱动器映射、关节限位保护、正确的索引处理
    """
    def __init__(self, model, data, site_name="cutter_tip", joint_names: List[str] = None):
        self.model = model
        self.data = data
        self.site_name = site_name
        
        # 1. 确定末端执行器
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id == -1:
            raise ValueError(f"XML中未找到 site: {site_name}")

        # 2. 如果没传关节名，默认使用掘进机常见的三大关节
        if joint_names is None:
            # 请确保这些名字与您的 XML 完全一致
            joint_names = ["gangti_self_joint_left", "huosai_front_joint", "jiege_shengsuo_joint"]
        
        self.joint_names = joint_names
        
        # 3. 初始化索引列表
        self.joint_ids = []      # 关节本身的 ID (0, 1, 2...)
        self.qpos_indices = []   # qpos 数组中的起始索引 (用于读写角度)
        self.dof_indices = []    # Jacobian / qvel 数组中的起始索引 (用于计算)
        self.actuator_ids = []   # 对应的驱动器 ID
        
        # 4. 构建映射关系
        print(f"--- 初始化 IK 求解器: {site_name} ---")
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"未找到关节: {name}")
            
            self.joint_ids.append(jid)
            self.qpos_indices.append(model.jnt_qposadr[jid])
            self.dof_indices.append(model.jnt_dofadr[jid])
            
            # 自动查找对应的驱动器
            act_id = self._find_actuator_for_joint(jid)
            self.actuator_ids.append(act_id)
            
            act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id) if act_id != -1 else "None"
            print(f"Joint: {name} -> Actuator: {act_name} (ID: {act_id})")

    def _find_actuator_for_joint(self, joint_id):
        """
        遍历模型中所有 Actuator，找到驱动指定 Joint 的那个。
        原理：model.actuator_trnid[act_id, 0] 存储了该驱动器作用的 Joint ID。
        """
        for i in range(self.model.nu):
            # trnid 记录了传动链的目标。对于普通关节传动，第一个参数就是 joint_id
            if self.model.actuator_trnid[i, 0] == joint_id:
                return i
        return -1 # 未找到对应驱动器

    def solve(self, target_pos: np.ndarray, max_steps=100, tol=0.005) -> Optional[np.ndarray]:
        """
        执行 IK 解算
        返回: 对应 Actuator 的目标长度/位置 (Control Values)
        """
        # 备份当前状态
        qpos_backup = self.data.qpos.copy()
        
        success = False
        
        # 转换为 numpy 数组方便计算
        dof_idx_arr = np.array(self.dof_indices, dtype=int)
        
        for step in range(max_steps):
            # 1. 计算误差
            current_pos = self.data.site_xpos[self.site_id]
            error = target_pos - current_pos
            
            if np.linalg.norm(error) < tol:
                success = True
                break

            # 2. 计算雅可比
            jacp = np.zeros((3, self.model.nv))
            # jacr (旋转) 暂时不需要，除非你也要控制截割头的朝向
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site_id)
            
            # 3. 提取相关关节的 Jacobian (3 x N_joints)
            j_reduced = jacp[:, dof_idx_arr]

            # 4. 求解 delta_q (使用阻尼最小二乘法 DLS 增强稳定性)
            # lambda_dls 是阻尼系数，防止在奇异点附近动作过大
            lambda_dls = 0.01
            j_t = j_reduced.T
            # DLS 公式: J^T * (J * J^T + lambda^2 * I)^-1 * error
            # 这里简化直接用伪逆，但步长给小一点
            delta_q = np.linalg.pinv(j_reduced, rcond=1e-2) @ error
            
            # 5. 更新关节角度 (带限位保护)
            for i, q_idx in enumerate(self.qpos_indices):
                # 获取该关节的物理限位
                jid = self.joint_ids[i]
                limit_min, limit_max = self.model.jnt_range[jid]
                
                # 计算新角度
                new_val = self.data.qpos[q_idx] + delta_q[i] * 0.2  # 0.2 是较保守的学习率
                
                # 强制 Clip 防止穿模/液压缸爆炸
                self.data.qpos[q_idx] = np.clip(new_val, limit_min, limit_max)
            
            # 6. 刷新运动学，让修改后的 qpos 生效
            mujoco.mj_kinematics(self.model, self.data)

        # 解算完成，提取结果
        if success:
            print(f"IK Converged in {step} steps.")
            # 此时 qpos 已经是目标姿态，我们需要计算对应的驱动器长度
            # 必须调用 transmission 来更新 actuator_length
            mujoco.mj_transmission(self.model, self.data)
            
            target_actuator_values = []
            for i, act_id in enumerate(self.actuator_ids):
                if act_id != -1:
                    # 获取该驱动器的当前长度（作为控制目标）
                    val = self.data.actuator_length[act_id]
                    target_actuator_values.append(val)
                else:
                    # 如果某个关节没有驱动器（是被动的），填 0 或保持原样
                    target_actuator_values.append(0.0)
            
            # 恢复现场 (以免影响主仿真循环)
            self.data.qpos[:] = qpos_backup
            mujoco.mj_kinematics(self.model, self.data)
            
            return np.array(target_actuator_values)
        else:
            print("IK Failed to converge.")
            # 恢复现场
            self.data.qpos[:] = qpos_backup
            mujoco.mj_kinematics(self.model, self.data)
            return None

# --- 测试代码 ---
if __name__ == "__main__":
    import os
    xml_path = "../myrobot.xml" # 请确保路径正确
    
    if os.path.exists(xml_path):
        m = mujoco.MjModel.from_xml_path(xml_path)
        d = mujoco.MjData(m)
        # mujoco.mj_kinematics(m, d)
        mujoco.mj_forward(m, d)
        
        # 你的关节名称列表
        my_joints = ["gangti_self_joint_left", "huosai_front_joint", "jiege_shengsuo_joint"] 

        try:
            # 1. 初始化求解器
            ik = RoadheaderIK(m, d, site_name="cutter_tip", joint_names=my_joints)
            print("IK Solver initialized.")

            # ==========================================
            #  在此处插入你的测试代码 (替换掉原来的逻辑)
            # ==========================================
            print("\n--- 开始: 物理可达性测试 (Sanity Check) ---")
            
            # A. 记下当前其实状态
            q_origin = d.qpos.copy()
            print(f"原始 Qpos (部分): {q_origin[ik.qpos_indices]}")

            # B. 手动让升降关节动一点点 (生成一个绝对合法的目标)
            # 注意：索引 1 对应 my_joints 里的第二个关节 (huosai_front_joint)
            test_joint_idx = 1 
            offset_val = 0.1 # 移动 0.1 (如果是移动关节是米，旋转关节是弧度)
            
            print(f"正在手动移动关节: {my_joints[test_joint_idx]} 偏移 {offset_val}")
            d.qpos[ik.qpos_indices[test_joint_idx]] += offset_val 
            
            # 刷新运动学，获取这个新姿态下的末端坐标
            mujoco.mj_kinematics(m, d)
            achievable_target = d.site_xpos[ik.site_id].copy()
            print(f"生成的【绝对可达】目标点: {achievable_target}")

            # C. 恢复机器人到原状，准备让 IK 自己算回去
            d.qpos[:] = q_origin
            mujoco.mj_kinematics(m, d)
            print("机器人已复位，准备开始 IK 解算...")

            # D. 让 IK 去解这个 achievable_target
            # 理论上：如果算法是对的，它应该能算出让该关节移动 +0.1 的结果
            ctrl_cmds = ik.solve(achievable_target)
            
            if ctrl_cmds is not None:
                print("\n>>> 测试成功！IK 收敛了。")
                print(f"计算出的控制量: {ctrl_cmds}")
                
                # 验证一下：看看算出来的 qpos 和我们手动移的是不是差不多
                # 我们之前手动移了第2个关节 +0.1，看看 IK 算完是不是也差不多加了 0.1
                current_q = d.qpos[ik.qpos_indices]
                print(f"IK 解算后的关节角度: {current_q}")
                print(f"期望的关节角度 (原值 + 偏移): \n{q_origin[ik.qpos_indices] + [0, offset_val, 0]}")
            else:
                print("\n>>> 测试失败。即使是物理可达的点，IK 也没算出来。")
                print("可能原因：Jacobian 索引错误、步长太大震荡、或者陷入局部极小值。")

            # ==========================================

        except ValueError as e:
            print(f"初始化错误: {e}")
    else:
        print("未找到 XML 文件。")