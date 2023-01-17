import numpy as np
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor
from pydantic import BaseModel
import cppimport.import_hook
from environment import Environment
from mcts import MonteCarloTreeSearch
from replan import RePlan
from config import Config
from pogema.wrappers.metrics import CSRMetric, ISRMetric, EpLengthMetric
import os
from time import time
import pandas as pd
from tqdm import tqdm

def main():
    #gc = GridConfig(size=5, num_agents=2, seed=63, density=0.3, obs_radius=5, max_episode_steps=128)
    #gc = GridConfig(size=4, num_agents=2, seed=62, density=0.4, obs_radius=5, max_episode_steps=32)
    #gc = GridConfig(size=4, num_agents=3, seed=42, density=0.3, obs_radius=2, max_episode_steps=32)
    #gc = GridConfig(size=4, num_agents=3, seed=42, density=0.0, obs_radius=2, max_episode_steps=32)
    gc = GridConfig(size=6, num_agents=5, seed=62, density=0.1, max_episode_steps=32)
    results = []
    # for heuristics in tqdm([0.5, 2]):
    #     #for seed in tqdm([207,522,503,511,116,504,770,694,977,710 ,513,411,381,280,333,774,60,449,728,673,512,249,173,658,656,356,753,217]):
    #     for seed in tqdm([0,1,2,3,4,5]):
    #gc = GridConfig(size=16, num_agents=16, seed=1, density=0.3, obs_radius=5, max_episode_steps=32)
    #gc = GridConfig(map=""".BabA""", obs_radius=5, max_episode_steps=12)
    #gc = GridConfig(size=8, num_agents=3, seed=1, density=0.2, obs_radius=5, max_episode_steps=64)
    gc.persistent = True
    gc.collision_system = 'block_both'
    mcts_config = Config()
    mcts_config.num_parallel_trees = 1
    mcts_config.heuristic_coef = 0
    mcts_config.render = False
    mcts_config.simulation_type = "replan"
    mcts_config.num_expansions = 100
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
    replan = RePlan()
    replan.init(env.get_num_agents(),gc.obs_radius,True,0.2,True, 10000000, -1, False)
    cpp_env = Environment()
    for i in range(env.get_num_agents()):
        cpp_env.add_agent(env.grid.positions_xy[i][0], env.grid.positions_xy[i][1],
                env.grid.finishes_xy[i][0], env.grid.finishes_xy[i][1])
        cpp_env.create_grid(len(env.grid.obstacles),len(env.grid.obstacles[0]))
    for i in range(len(env.grid.obstacles)):
        for j in range(len(env.grid.obstacles[0])):
            if env.grid.obstacles[i][j]:
                cpp_env.add_obstacle(i, j)
    mcts.set_env(cpp_env, gc.obs_radius)
    replan.set_env(cpp_env)
    done = [False]
    start = time()
    while not all(done):
        actions = mcts.act()
        obs, rew, done, info = env.step(actions)
        env.render()
    end = time() - start
    results.append(info[0]['metrics'])
    results[-1]['FPS'] = 32/end
    results[-1]['heuristic_coef'] = 0
    # print(pd.DataFrame(results).groupby('heuristic_coef').mean().applymap(lambda x: "{:.2f}".format(x)) + \
    #       '(±' + pd.DataFrame(results).groupby('heuristic_coef').std().applymap(lambda x: "{:.4f}".format(x)) + ')')
if __name__ == '__main__':
    main()