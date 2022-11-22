#ifndef MCTS_CONFIG_H
#define MCTS_CONFIG_H

struct Config
{
    double gamma = 0.99;
    int num_actions = 5;
    int num_expansions = 1000;
    double uct_c = 1.0;
    int steps_limit = 64;
    int multi_simulations = 1;
    bool use_move_limits = true;
    bool agents_as_obstacles = false;
    int batch_size = 8;
};

#endif //MCTS_CONFIG_H
