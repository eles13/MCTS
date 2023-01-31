#include "BS_thread_pool.hpp"
#include <iostream>
#include <list>
#include <vector>
#include <numeric>
#include <cmath>
#include <string>
#include <chrono>
#include <unordered_map>
#include "config.cpp"
#include "node.hpp"
#include "replan.cpp"

class DepthStatsHandler
{
    public:
    int agent_id;
    int cnt;
    double q;
    int action;
    int depth;
};

class MonteCarloTreeSearch
{
    Node* root;
    std::list<Node> all_nodes;
    std::list<Environment> all_envs;
    Config cfg;
    BS::thread_pool pool;
    std::vector<Node*> ptrees;
    std::vector<Environment> penvs;
    int num_envs;
    std::vector<std::vector<std::vector<double>>> shortest_paths;
    int obs_radius;
    bool first_move = true;

public:
    Environment env;

    explicit MonteCarloTreeSearch();

    std::vector<int> act();

    void set_env(Environment env_, const int obs_radius_);

    void set_config(const Config& config);

    std::vector<DepthStatsHandler> stats;
    std::vector<DepthStatsHandler> fmstats;

    int depth;

protected:
    Node* safe_insert_node(Node* n, const int action, const double score, const int num_actions, const int next_agent_idx);

    double single_simulation(const int process_num);

    double simulation(const int process_num);

    double uct(Node* n, const int agent_idx, const int process_num) const;

    double batch_uct(Node* n) const;

    int expansion(Node* n, const int agent_idx, const int process_num) const;

    double selection(Node* n, std::vector<int> actions, const int process_num);

    int select_action_for_batch_path(Node* n, const int agent_idx, const int process_num);

    std::vector<int> batch_selection(Node* n, std::vector<int> actions, const int process_num);

    double batch_expansion(std::vector<int> path_actions, std::vector<int> prev_actions, const int process_num);

    void loop(std::vector<int>& prev_actions);

    void batch_loop(std::vector<int>& prev_actions);

    void retrieve_statistics(Node* tree, Node* from_root);

    void tree_parallelization_loop_internal(std::vector<int> prev_actions, const int process_num);

    void tree_parallelization_loop(std::vector<int>& prev_actions);

    std::vector<std::vector<std::vector<double>>> bfs(Environment& env);

    std::vector<DepthStatsHandler> get_path(Node* n, const int process_num, int rec_depth);
};
