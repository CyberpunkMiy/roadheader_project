import mujoco
import mujoco.viewer
import numpy as np
import time
import math
from core import RoadheaderIK

def generate_s_curve_trajectory(start_pos, width=1.5, height=1.0, rows=4, points_per_row=50):
    """
    生成一个 S 型扫描轨迹 (用于模拟截割断面的工况)
    
    :param start_pos: 轨迹起始中心点 [x, y, z]
    :param width: 扫描宽度 (米)
    :param height: 扫描高度 (米)
    :param rows: 扫描行数 (Z轴方向分的层数)
    :param points_per_row: 每一行生成的轨迹点数量
    :return: 轨迹点列表 (N, 3)
    """
    waypoints = []
    
    # 定义扫描范围的边界
    z_start = start_pos[2] + height / 2
    z_end = start_pos[2] - height / 2
    x_left = start_pos[0] - width / 2
    x_right = start_pos[0] + width / 2
    
    # 保持 Y 轴恒定 (或者稍微向前进给，这里暂定恒定)
    y_pos = start_pos[1]
    
    # 生成 Z 轴的层
    z_levels = np.linspace(z_start, z_end, rows)
    
    for i, z in enumerate(z_levels):
        # 偶数行：从左到右
        if i % 2 == 0:
            x_range = np.linspace(x_left, x_right, points_per_row)
        # 奇数行：从右到左 (这样就是连续的 S 型)
        else:
            x_range = np.linspace(x_right, x_left, points_per_row)
            
        for x in x_range:
            waypoints.append([x, y_pos, z])
            
    return np.array(waypoints)

def main():
    xml_path = "myrobot.xml"
    print(f"Loading model from {xml_path}...")
    
    try:
        m = mujoco.MjModel.from_xml_path(xml_path)
        d = mujoco.MjData(m)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 1. 初始物理刷新 (非常重要!)
    mujoco.mj_forward(m, d)

    # 2. 初始化 IK 求解器
    try:
        ik_solver = RoadheaderIK(m, d, site_name="cutter_tip")
    except ValueError as e:
        print(e)
        return

    # 3. 获取当前截割头位置作为基准
    current_tip = d.site_xpos[ik_solver.site_id].copy()
    print(f"Initial Tip Position: {current_tip}")

    # 4. 生成目标轨迹 (以当前位置前方 0.2米为中心)
    #    注意：不要设置太大的 width/height，以免超出机械臂物理极限
    center_pos = current_tip + np.array([0, 0.05, 0]) 
    
    trajectory = generate_s_curve_trajectory(
        start_pos=center_pos,
        width=0.001,   # 左右只扫 0.3 米
        height=0.001,  # 上下只扫 0.2 米
        rows=3,      
        points_per_row=30
    )
    print(f"Generated trajectory with {len(trajectory)} points.")

    # 5. 启动可视化
    print("Launching viewer... (Please wait)")
    with mujoco.viewer.launch_passive(m, d) as viewer:
        
        # --- 摄像机设置 (上帝视角) ---
        viewer.cam.lookat[:] = [0, 0, 1.5]  # 看着机身
        viewer.cam.distance = 7.0           # 拉远
        viewer.cam.elevation = -25          # 俯视
        viewer.cam.azimuth = 140            # 侧角
        # ---------------------------

        # 初始停顿
        start_wait = time.time()
        while time.time() - start_wait < 1.0:
            viewer.sync()
            time.sleep(0.01)

        # 循环执行轨迹
        for i, target_pos in enumerate(trajectory):
            if not viewer.is_running():
                break

            # A. 调用 IK 求解目标油缸长度
            #    注意：IK 求解很快，但物理运动需要时间
            target_ctrl = ik_solver.solve(target_pos, tol=0.01, max_steps=20)

            if target_ctrl is not None:
                # B. 应用控制信号
                # 假设前3个是: [0]左右, [1]升降, [2]伸缩
                d.ctrl[0] = target_ctrl[0]
                d.ctrl[1] = target_ctrl[1]
                d.ctrl[2] = target_ctrl[2]
                
                # C. 物理仿真步进 (平滑移动)
                # 给 PID 控制器一点时间去执行动作 (模拟真实液压响应)
                # 步数越多，动作越慢越平滑
                steps_per_waypoint = 100 
                for _ in range(steps_per_waypoint):
                    mujoco.mj_step(m, d)
                    viewer.sync()
                
                # 在控制台打印进度
                print(f"Point {i+1}/{len(trajectory)}: Target {target_pos} -> Done")
            else:
                print(f"Point {i+1}/{len(trajectory)}: ❌ IK Failed (Out of reach?)")

        print("Trajectory finished.")
        while viewer.is_running():
            time.sleep(0.1)

if __name__ == "__main__":
    main()