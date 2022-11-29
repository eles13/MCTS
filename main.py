import numpy as np
from pogema import pogema_v0, GridConfig
from pogema.animation import AnimationMonitor
from pydantic import BaseModel
import cppimport.import_hook
from environment import Environment
from mcts import MonteCarloTreeSearch
from config import Config

def main():
    gc = GridConfig(size=5, num_agents=2, seed=63, density=0.3, obs_radius=2, max_episode_steps=128)
    gc = GridConfig(size=4, num_agents=2, seed=62, density=0.4, obs_radius=2, max_episode_steps=32)
    gc = GridConfig(size=4, num_agents=3, seed=42, density=0.3, obs_radius=2, max_episode_steps=32)
    gc = GridConfig(size=4, num_agents=3, seed=42, density=0.0, obs_radius=2, max_episode_steps=32)
    gc = GridConfig(size=6, num_agents=5, seed=62, density=0.1, max_episode_steps=32)
    gc = GridConfig(size=12, num_agents=4, seed=62, density=0.3, obs_radius=2)
    # gc = GridConfig(map=""".BabA""", obs_radius=2, max_episode_steps=12)
    # gc = GridConfig(size=8, num_agents=3, seed=1, density=0.2, obs_radius=5, max_episode_steps=64)
    gc.persistent = True
    gc.collision_system = 'block_both'
    mcts_config = Config()
    mcts = MonteCarloTreeSearch()
    mcts.set_config(mcts_config)
    env = pogema_v0(gc)
    env = AnimationMonitor(env)
    env.reset()
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
    print(done)

if __name__ == '__main__':
    main()