import mujoco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def check_workspace():
    xml_path = "../myrobot.xml" # 注意路径
    try:
        m = mujoco.MjModel.from_xml_path(xml_path)
        d = mujoco.MjData(m)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 1. 定义我们要采样的 3 个核心关节 (对应左右、升降、伸缩)
    #    这些名字必须和你 XML 里的一致
    target_joints = [
        "gangti_self_joint_left",  # 左右回转油缸
        "huosai_front_joint",      # 升降油缸
        "jiege_shengsuo_joint"     # 伸缩臂
    ]
    
    # 获取关节的 ID 和 地址
    joint_info = []
    for name in target_joints:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            print(f"❌ 找不到关节: {name}")
            return
        
        # 获取关节的物理限制范围 (range)
        j_range = m.jnt_range[jid]
        qadr = m.jnt_qposadr[jid]
        
        print(f"关节 {name:25} | 范围: {j_range}")
        joint_info.append({'qadr': qadr, 'min': j_range[0], 'max': j_range[1]})

    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "cutter_tip")
    
    # 2. 开始随机采样
    num_samples = 10000  # 采样点数，越多越精确
    points = []

    print(f"\n正在计算 {num_samples} 个采样点，请稍候...")

    for _ in range(num_samples):
        # 对每个关节，在它的 min 和 max 之间随机取一个值
        for j in joint_info:
            # np.random.uniform 生成均匀分布的随机数
            rand_val = np.random.uniform(j['min'], j['max'])
            d.qpos[j['qadr']] = rand_val
        
        # ⚠️ 关键：计算正向运动学 (Forward Kinematics)
        # mj_forward 会自动处理连杆闭环约束，算出大臂实际的角度
        mujoco.mj_forward(m, d)
        
        # 记录此时截割头的位置
        points.append(d.site_xpos[site_id].copy())

    points = np.array(points)

    # 3. 分析结果 (打印边界)
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    range_xyz = max_xyz - min_xyz

    print("\n" + "="*40)
    print("📊 工作空间统计结果 (单位: 米)")
    print("="*40)
    print(f"X轴 (左右): {min_xyz[0]:.3f} 到 {max_xyz[0]:.3f} (宽度: {range_xyz[0]:.3f})")
    print(f"Y轴 (前后): {min_xyz[1]:.3f} 到 {max_xyz[1]:.3f} (进深: {range_xyz[1]:.3f})")
    print(f"Z轴 (上下): {min_xyz[2]:.3f} 到 {max_xyz[2]:.3f} (高度: {range_xyz[2]:.3f})")
    print("="*40)
    print("💡 建议：在 trajectory_test.py 中设置的 width 和 height")
    print(f"   不应超过 {range_xyz[0]:.2f} 和 {range_xyz[2]:.2f}")
    print("="*40)

    # 4. 可视化 (绘制 3D 散点图)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 画点
    # X, Y 是水平面，Z 是高度
    # c=points[:, 2] 让颜色随高度变化
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c=points[:, 2], cmap='viridis')
    
    ax.set_xlabel('X (Left-Right)')
    ax.set_ylabel('Y (Front-Back)')
    ax.set_zlabel('Z (Up-Down)')
    ax.set_title('Robot Workspace (Reachable Points)')
    
    # 保持比例一致，避免视觉变形
    # (Matplotlib 3D 的 axis equal 有点麻烦，这里简单设置范围)
    max_range = np.array([points[:,0].max()-points[:,0].min(), points[:,1].max()-points[:,1].min(), points[:,2].max()-points[:,2].min()]).max() / 2.0
    mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
    mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
    mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.colorbar(sc, label='Height (Z)')
    plt.show()

if __name__ == "__main__":
    check_workspace()