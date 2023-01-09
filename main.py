from ast import Is
from distutils.log import INFO
import numpy as np
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor
from pydantic import BaseModel
import cppimport.import_hook
from environment import Environment
from mcts import MonteCarloTreeSearch
from config import Config
from pogema.wrappers.metrics import CSRMetric, ISRMetric, EpLengthMetric
import os

def main():
    gc = GridConfig(size=5, num_agents=2, seed=63, density=0.3, obs_radius=2, max_episode_steps=128)
    gc = GridConfig(size=4, num_agents=2, seed=62, density=0.4, obs_radius=2, max_episode_steps=32)
    gc = GridConfig(size=4, num_agents=3, seed=42, density=0.3, obs_radius=2, max_episode_steps=32)
    gc = GridConfig(size=4, num_agents=3, seed=42, density=0.0, obs_radius=2, max_episode_steps=32)
    gc = GridConfig(size=6, num_agents=5, seed=62, density=0.1, max_episode_steps=32)
    #for seed in range(1000):

    gc = GridConfig(size=16, num_agents=16, seed=207, density=0.3, obs_radius=5, max_episode_steps=32)
    # gc = GridConfig(map=""".BabA""", obs_radius=2, max_episode_steps=12)
    #gc = GridConfig(size=8, num_agents=3, seed=1, density=0.2, obs_radius=5, max_episode_steps=64)
    gc.persistent = True
    gc.collision_system = 'block_both'
    mcts_config = Config()
    mcts_config.num_parallel_trees = 1
    mcts_config.heuristic_coef = 0.5
    mcts_config.render = False
    mcts = MonteCarloTreeSearch()
    env = pogema_v0(gc)
    env = CSRMetric(env)
    env = ISRMetric(env)
    env = EpLengthMetric(env)
    env = AnimationMonitor(env)
    env.reset()
    mcts_config.steps_limit = env.grid_config.max_episode_steps
    mcts.set_config(mcts_config)
    env.render()
    cpp_env = Environment()
    for i in range(env.get_num_agents()):
        cpp_env.add_agent(env.grid.positions_xy[i][0], env.grid.positions_xy[i][1],
                env.grid.finishes_xy[i][0], env.grid.finishes_xy[i][1])
        cpp_env.create_grid(len(env.grid.obstacles),len(env.grid.obstacles[0]))
    for i in range(len(env.grid.obstacles)):
        for j in range(len(env.grid.obstacles[0])):
            if env.grid.obstacles[i][j]:
                cpp_env.add_obstacle(i, j)
    mcts.set_env(cpp_env)
    done = [False]
    while not all(done):
        actions = mcts.act()
        obs, rew, done, info = env.step(actions)
        env.render()
    print(info)

if __name__ == '__main__':
    main()