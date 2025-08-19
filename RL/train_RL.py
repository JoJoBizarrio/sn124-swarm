#!/usr/bin/env python3

import argparse
import os


from stable_baselines3 import PPO

from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task
from swarm.validator.forward import SIM_DT, HORIZON_SEC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--timesteps", type=int, default=5000)
    parser.add_argument("-s", "--seed", type=int, default=1)
    parser.add_argument("-g", "--gui", type=bool, default=False)
    args = parser.parse_args()

    #task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=1)
    task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=args.seed)

    env = make_env(task, gui=args.gui)

    model_path = "model/ppo_policy.zip"
    if os.path.exists(model_path):
        model = PPO.load("model/ppo_policy", env=env, device="cuda")
        print("Loaded existing model from", model_path)
    else:
        model = PPO("MlpPolicy", env, verbose=1)
        print("Created new model")

    print("Start position:", task.start)  
    print("Goal position:", task.goal) 
    
    info = env._computeInfo()
    print("Distance to goal:", info["distance_to_goal"])
    print("Success:", info["success"])
    print("Time to goal:", info["t_to_goal"]) 
    model.learn(args.timesteps)

    # while not done:
    #     # Для старта — случайные действия (пример)
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated

    #     env.render()

    #frame = env.render(mode='rgb_array') 

    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    model.save("model/ppo_policy")

    env.close()


if __name__ == "__main__":
    main()
