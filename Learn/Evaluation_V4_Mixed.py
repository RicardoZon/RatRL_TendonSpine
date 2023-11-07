from RatEnv.RL_wrapper4_Mixed import RatRL
# from RatEnv.RL_wrapper4_Mixed_OnlySpine import RatRL
# from RatEnv.RL_wrapper4_Mixed_TendonVersion import RatRL
import gym
from stable_baselines3 import PPO
# from stable_baselines3 import SAC
# from stable_baselines3 import A2C
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from Tools.DataRecorder import DATA_Recorder

RENDER = True
Par = None

if __name__ == '__main__':
    # SceneFile = "../models/dynamic_4l_t3.xml"
    # MODELPATH = "Local_Data/Env4_Mixed/S0_PPO_Env4Mixed_015"  #  frame = 15
    # Par = {'frame_skip': 15}

    SceneFile = "../models/Scenario1_Planks.xml"
    MODELPATH = "Local_Data/Env4_Mixed/S4P_PPO_Env4Mixed_018"

    # SceneFile = "../models/Scenario2_Uphill.xml"  # S2

    # SceneFile = "../models/scene_test2pro.xml"  # S2

    # SceneFile = "../models/Scenario3_Logs.xml"  # 3

    # SceneFile = "../models/Scenario4_Stairs.xml"

    # SceneFile = "../models/Scenario4_Stairs_Sparse.xml"
    # MODELPATH = "Local_Data/Env4_Mixed/S4P_PPO_Env4Mixed_017"
    Par = {'frame_skip': 15}

    Recorder = DATA_Recorder()


    env = RatRL(SceneFile, Render=RENDER, Pars=Par)
    model = PPO.load(MODELPATH, env=env)

    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    # pos_Ori = vec_env.envs[0].pos[1]
    pos_end = []
    for i in range(int(6000)):
        pos_pre = vec_env.envs[0].pos[1]

        action, _states = model.predict(obs, deterministic=True)
        # action[0][3] = 1.0
        # print(action)
        # action = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        obs, rewards, dones, info = vec_env.step(action)
        # print(info)
        # print(vec_env.envs[0].pos)
        # vec_env.render()
        # Recorder.update(vec_env.envs[0])

        if dones[0]:
            pos_end.append(pos_pre)
            print(pos_pre)
            print(t)
        else:
            t = vec_env.envs[0].env.sim.data.time

    # times = np.array(vec_env.envs[0].episode_lengths)* vec_env.envs[0].dt
    # v_global = -(np.array(pos_end) - pos_Ori) / np.array(times)
    # print(v_global.mean())

    # Recorder.savePath_Basic("S1_Pass_073")

