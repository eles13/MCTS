#ifdef __APPLE__
#else
    #include "omp.h"
#endif
#include "BS_thread_pool.hpp"
#include <iostream>
#include <list>
#include <vector>
#include <numeric>
#include <cmath>
#include <string>
#include <chrono>
#include "config.cpp"
#include "node.hpp"

class MonteCarloTreeSearch
{
    Node* root;
    std::list<Node> all_nodes;
    Environment env;
    Config cfg;
    BS::thread_pool pool;
public:
    explicit MonteCarloTreeSearch();

    std::vector<int> act();

    void set_env(Environment& env_);

    void set_config(const Config& config);

protected:
    double simulation(Environment& local_env);

    double uct(Node* n) const;

    double batch_uct(Node* n) const;

    int expansion(Node* n, const int agent_idx) const;

    double selection(Node* n, std::vector<int> actions, Environment& env);

    int select_action_for_batch_path(Node* n, const int agent_idx);

    std::vector<int> batch_selection(Node* n, std::vector<int> actions);

    double batch_expansion(std::vector<int> path_actions, std::vector<int> prev_actions, Environment cpenv);

    void loop(std::vector<int>& prev_actions);

    void batch_loop(std::vector<int>& prev_actions);

    void retrieve_statistics(Node* tree, Node* from_root);

    Node tree_parallelization_loop_internal(Node root, std::vector<int> prev_actions, Environment cpenv);

    void tree_parallelization_loop(std::vector<int>& prev_actions);
};