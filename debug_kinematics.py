import mujoco
import numpy as np
import os

def debug_robot():
    xml_path = "myrobot.xml"
    if not os.path.exists(xml_path):
        print(f"Error: 找不到 {xml_path}")
        return

    print(f"Loading {xml_path}...")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    # 关节名称列表 (请确保和您的一致)
    joint_names = ["gangti_self_joint_left", "huosai_front_joint", "jiege_shengsuo_joint"]
    site_name = "cutter_tip"
    
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    print(f"\nTarget Site: {site_name} (ID: {site_id})")
    
    print("\n=== 1. 关节结构分析 ===")
    dof_indices = []
    qpos_indices = []
    
    for name in joint_names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            print(f"❌ 找不到关节: {name}")
            continue
            
        jtype = m.jnt_type[jid]
        qadr = m.jnt_qposadr[jid]
        dadr = m.jnt_dofadr[jid]
        
        # 解析关节类型
        type_str = "Unknown"
        if jtype == mujoco.mjtJoint.mjJNT_FREE: type_str = "FREE (6DOF)"
        elif jtype == mujoco.mjtJoint.mjJNT_BALL: type_str = "BALL (3DOF)"
        elif jtype == mujoco.mjtJoint.mjJNT_SLIDE: type_str = "SLIDE (1DOF)"
        elif jtype == mujoco.mjtJoint.mjJNT_HINGE: type_str = "HINGE (1DOF)"
        
        print(f"Joint: {name:25} | ID: {jid} | Type: {type_str} | QposAdr: {qadr} | DofAdr: {dadr}")
        
        dof_indices.append(dadr)
        qpos_indices.append(qadr)

    print("\n=== 2. 雅可比矩阵 (Jacobian) 诊断 ===")
    # 刷新运动学
    mujoco.mj_kinematics(m, d)
    
    # 计算雅可比
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
    
    print("检查我们在 IK 中使用的 Jacobian 列 (Translation):")
    for i, name in enumerate(joint_names):
        dof_idx = dof_indices[i]
        col = jacp[:, dof_idx]
        norm = np.linalg.norm(col)
        print(f"Joint {name:25} -> Jacobian Column: {col} (Norm: {norm:.6f})")
        
        if norm < 1e-6:
            print(f"   ⚠️ 警告: 该关节的 Jacobian 几乎为 0！MuJoCo 认为动这个关节不会改变 {site_name} 的位置。")
            print(f"   可能是因为：Site 不在该关节的子树上 (Tree Structure Issue)。")

    print("\n=== 3. 真实物理梯度验证 (Finite Difference) ===")
    # 我们手动动一下关节，看看末端到底动不动
    current_pos = d.site_xpos[site_id].copy()
    epsilon = 1e-4 # 微小扰动
    
    for i, name in enumerate(joint_names):
        q_idx = qpos_indices[i]
        dof_idx = dof_indices[i]
        
        # 备份
        old_val = d.qpos[q_idx]
        
        # 扰动
        d.qpos[q_idx] = old_val + epsilon
        mujoco.mj_kinematics(m, d)
        new_pos = d.site_xpos[site_id].copy()
        
        # 恢复
        d.qpos[q_idx] = old_val
        mujoco.mj_kinematics(m, d)
        
        # 计算实际的 delta
        diff = new_pos - current_pos
        numerical_jac_col = diff / epsilon
        
        analytical_col = jacp[:, dof_idx]
        
        print(f"\n测试关节: {name}")
        print(f"  手动微动 {epsilon} 后，末端位移: {diff}")
        print(f"  数值梯度 (Numerical):  {numerical_jac_col}")
        print(f"  解析梯度 (Analytical): {analytical_col}")
        
        err = np.linalg.norm(numerical_jac_col - analytical_col)
        if err > 1e-3:
             print(f"  ❌ 严重不一致! 误差: {err:.6f}")
             print("  结论: mj_jacSite 计算出的导数与实际物理运动不符。通常是因为模型使用了等式约束(Equality Constraint)而非树状结构。")
        else:
             print(f"  ✅ 一致 (误差: {err:.6f})")

if __name__ == "__main__":
    try:
        debug_robot()
    except Exception as e:
        print(f"发生错误: {e}")