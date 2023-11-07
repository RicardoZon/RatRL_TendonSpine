from RatEnv.RL_wrapper4_Mixed import RatRL
# from RatEnv.RL_wrapper4_Mixed_OnlySpine import RatRL
# from RatEnv.RL_wrapper4_Mixed_TendonVersion import RatRL
import gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import A2C
import warnings
from stable_baselines3.common.callbacks import CheckpointCallback

RENDER_TRAIN = False

if __name__ == '__main__':
    Par = {'frame_skip': 15}
    # SceneFile = "../models/dynamic_4l_t3.xml"
    # NAME = "S0_PPO_Env4Mixed_014"
    # Par = {'frame_skip': 10}

    # NAME = "S0_PPO_Env4Mixed_015"
    # Par = {'frame_skip': 15}

    # SceneFile = "../models/Scenario1_Planks.xml"  # S1
    # NAME = "S4P_PPO_Env4Mixed_018"

    # SceneFile = "../models/Scenario2_Uphill.xml"  # S2  Spe
    # NAME = "S4P_PPO_Env4Mixed_019"

    # SceneFile = "../models/Scenario3_Logs.xml"  # S3
    # NAME = "S4P_PPO_Env4Mixed_020"

    # SceneFile = "../models/Scenario4_Stairs.xml"  # S4
    # NAME = "S4P_PPO_Env4Mixed_021"

    # SceneFile = "../models/Scenario4_Stairs_Sparse.xml"
    # NAME = "S4P_PPO_Env4Mixed_016"
    # Par = {'frame_skip': 10}

    # warnings.filterwarnings("ignore")
    # checkpoint_callback = CheckpointCallback(
    #     save_freq=200_000,
    #     save_path="./Local_Logs/Wrapper3Div/" + NAME,
    #     name_prefix="NAME",
    #     save_replay_buffer=True,
    #     save_vecnormalize=False,
    # )

    env = RatRL(SceneFile, Render=RENDER_TRAIN, Pars=Par)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./Local_Rat_Tensorboard/Env4_Mixed/")
    # model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./Local_Rat_Tensorboard/Wrapper3Div/")
    # model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./Local_Rat_Tensorboard/Wrapper3Div/")
    model.learn(total_timesteps=2_000_000, tb_log_name=NAME, reset_num_timesteps=True)
    model.save("./Local_Data/Env4_Mixed/" + NAME)

    # del model
