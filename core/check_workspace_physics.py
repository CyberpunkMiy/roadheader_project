import mujoco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def check_workspace_physics():
    xml_path = "../myrobot.xml" 
    print(f"Loading {xml_path}...")
    try:
        m = mujoco.MjModel.from_xml_path(xml_path)
        d = mujoco.MjData(m)
    except Exception as e:
        print(f"Error: {e}")
        return

    # === 设置 ===
    # 我们只控制这 3 个主动关节
    target_joints = [
        "gangti_self_joint_left",  # 左油缸 (决定左右)
        "huosai_front_joint",      # 升降油缸
        "jiege_shengsuo_joint"     # 伸缩臂
    ]
    
    joint_info = []
    for name in target_joints:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            print(f"❌ 找不到关节: {name}")
            return
        
        j_range = m.jnt_range[jid]
        qadr = m.jnt_qposadr[jid]
        print(f"关节 {name:25} | 范围: {j_range}")
        joint_info.append({'qadr': qadr, 'min': j_range[0], 'max': j_range[1]})

    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "cutter_tip")
    
    # 采样次数
    num_samples = 10000 
    points = []
    
    # 临时关闭重力，防止测试时大臂下垂影响范围判断
    m.opt.gravity[:] = [0, 0, 0]

    print(f"\n正在通过物理仿真计算 {num_samples} 个采样点...")

    for i in range(num_samples):
        # 1. 随机设置主动关节的位置
        for j in joint_info:
            rand_val = np.random.uniform(j['min'], j['max'])
            d.qpos[j['qadr']] = rand_val
        
        # 2. 关键步骤：重置速度，防止飞出去
        d.qvel[:] = 0
        
        # 3. 核心差异：运行物理仿真，让约束把机器“拉”到正确位置
        # 运行 100 步通常足够让约束稳定下来
        for _ in range(50):
            mujoco.mj_step(m, d)
        
        # 4. 记录稳定后的位置
        points.append(d.site_xpos[site_id].copy())
        
        if (i+1) % 500 == 0:
            print(f"已采样 {i+1} / {num_samples}")

    points = np.array(points)

    # === 统计与绘图 ===
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    range_xyz = max_xyz - min_xyz

    print("\n" + "="*40)
    print("📊 真实物理工作空间 (Physics Based)")
    print("="*40)
    print(f"X轴 (左右): {min_xyz[0]:.3f} 到 {max_xyz[0]:.3f} (宽度: {range_xyz[0]:.3f})")
    print(f"Y轴 (前后): {min_xyz[1]:.3f} 到 {max_xyz[1]:.3f} (进深: {range_xyz[1]:.3f})")
    print(f"Z轴 (上下): {min_xyz[2]:.3f} 到 {max_xyz[2]:.3f} (高度: {range_xyz[2]:.3f})")
    print("="*40)
    print("现在你可以根据这个宽度去设置 trajectory_test.py 了！")

    # 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c=points[:, 2], cmap='viridis')
    
    # 设置比例
    mid_x = (max_xyz[0] + min_xyz[0]) / 2
    mid_y = (max_xyz[1] + min_xyz[1]) / 2
    mid_z = (max_xyz[2] + min_xyz[2]) / 2
    max_range = max(range_xyz) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X (Left-Right)')
    ax.set_ylabel('Y (Front-Back)')
    ax.set_zlabel('Z (Up-Down)')
    ax.set_title('Robot Workspace (Physics Simulation)')
    plt.show()

if __name__ == "__main__":
    check_workspace_physics()