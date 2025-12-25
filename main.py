# 程序入口
import time
import numpy as np
from envs import RoadheaderEnv
from policies import SineWavePolicy
from core.kinematics import RoadheaderIK

def main():
    # 假设你的 XML 文件在根目录
    xml_path = "myrobot.xml"
    
    print(f"正在初始化环境: {xml_path} ...")
    
    # 1. 创建环境
    # 注意：如果 xml 不存在会报错，请确保文件路径正确
    try:
        env = RoadheaderEnv(xml_path)
    except FileNotFoundError as e:
        print(e)
        return

    # 2. 创建策略
    policy = SineWavePolicy(freq=0.4)
    
    print("初始化完成。开始仿真循环...")
    
    # 3. 运行循环
    obs = env.reset()
    start_time = time.time()
    
    try:
        # 持续运行直到手动关闭窗口
        while env.viewer is None or env.viewer.is_running():
            current_time = time.time() - start_time
            
            # 获取策略动作
            action = policy.predict(obs, current_time)
            
            # 环境步进
            obs, reward, done, info = env.step(action)
            
            # 渲染
            env.render()
            
            # 控制帧率 (可选，让仿真不要跑得太快以至于看不清)
            time.sleep(env.dt)

    except KeyboardInterrupt:
        print("\n仿真被用户中断")
    finally:
        env.close()
        print("环境已关闭")

if __name__ == "__main__":
    main()