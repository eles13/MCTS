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
#include "environment.h"
#include "config.h"
#include "node.hpp"

class MonteCarloTreeSearch
{
    Node* root;
    std::list<Node> all_nodes;
    Environment env;
    Config cfg;
    BS::thread_pool pool;
public:
    explicit MonteCarloTreeSearch(const std::string& fileName, int seed = -1);

    bool act();

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