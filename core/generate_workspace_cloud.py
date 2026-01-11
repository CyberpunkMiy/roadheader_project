import mujoco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

# 导入你提供的接口
try:
    from kinematics import RoadheaderIK
except ImportError:
    print("❌ 错误: 未找到 kinematics.py，请确保它在同一目录下。")
    exit()

def generate_workspace(xml_path, sample_count=2000):
    """
    使用蒙特卡洛法采样生成工作空间点云
    """
    print(f"Loading model from {xml_path}...")
    if not os.path.exists(xml_path):
        print(f"❌ 文件不存在: {xml_path}")
        return

    # 1. 加载模型
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 2. 初始化你的 IK 求解器作为接口
    # (虽然我们这次用的是正解，但利用你的类来初始化环境是很方便的)
    ik_solver = RoadheaderIK(model, data, site_name="cutter_tip")
    
    # 3. 获取驱动器范围 (Ctrl Range)
    # 假设 XML 里定义了 actuators，我们直接读取它们的物理限制
    ctrl_ranges = model.actuator_ctrlrange
    n_actuators = model.nu # 驱动器数量
    
    print(f"检测到 {n_actuators} 个驱动器。")
    print(f"开始采样 {sample_count} 个物理点 (这可能需要几秒钟)...")

    points = []
    
    # 4. 蒙特卡洛采样循环
    start_time = time.time()
    
    for i in range(sample_count):
        # --- A. 随机生成控制量 ---
        # 在每个驱动器的 [min, max] 范围内随机取值
        random_ctrl = np.random.uniform(
            ctrl_ranges[:, 0], 
            ctrl_ranges[:, 1]
        )
        
        # 将随机值应用到仿真数据中
        data.ctrl[:] = random_ctrl
        
        # --- B. 物理仿真 (关键步骤) ---
        # 对于并联机构(Parallel Mechanism)，必须运行 mj_step 
        # 让物理引擎计算约束力，把断开的关节“吸”在一起。
        # 如果只用 mj_forward，并联闭环可能会断开。
        for _ in range(20): 
            mujoco.mj_step(model, data)
            
        # --- C. 记录截割头位置 ---
        # 从你的接口中获取 site_id 并读取坐标
        tip_pos = data.site_xpos[ik_solver.site_id].copy()
        points.append(tip_pos)
        
        # 进度条
        if (i+1) % 500 == 0:
            print(f"已采样 {i+1}/{sample_count} 个点...")

    elapsed = time.time() - start_time
    print(f"采样完成！耗时: {elapsed:.2f} 秒")
    
    return np.array(points)

def plot_cloud(points):
    """
    绘制漂亮的 3D 点云图
    """
    if len(points) == 0:
        print("没有数据点。")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取 X, Y, Z
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    
    # 绘制散点，使用 Z 轴高度作为颜色映射
    sc = ax.scatter(xs, ys, zs, c=zs, cmap='viridis', s=5, alpha=0.6)
    
    # 设置坐标轴标签
    ax.set_xlabel('X (Left-Right) [m]')
    ax.set_ylabel('Y (Front-Back) [m]')
    ax.set_zlabel('Z (Up-Down) [m]')
    ax.set_title(f'Roadheader Cutter Workspace ({len(points)} points)')
    
    # 调整视角 (类似你之前的截图)
    ax.view_init(elev=30, azim=135)
    
    # 添加颜色条
    plt.colorbar(sc, label='Height (Z)')
    
    # 设置比例尺一致 (防止图形被拉伸)
    try:
        ax.set_box_aspect([np.ptp(xs), np.ptp(ys), np.ptp(zs)])
    except:
        pass # 旧版 matplotlib 可能不支持

    print("正在显示图像...")
    plt.show()

if __name__ == "__main__":
    # 请根据你的实际文件路径修改这里
    # 建议使用 myrobot_position.xml 或者 myrobot.xml
    XML_FILE = "../myrobot.xml" 
    
    # 如果文件在上一级目录，可以改成 "../myrobot.xml"
    if not os.path.exists(XML_FILE):
        # 尝试找一下常见路径
        if os.path.exists("../myrobot.xml"):
            XML_FILE = "../myrobot.xml"
    
    # 生成并绘图
    cloud_data = generate_workspace(XML_FILE, sample_count=30000)
    
    if cloud_data is not None:
        plot_cloud(cloud_data)