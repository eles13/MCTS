// cppimport
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "BS_thread_pool.hpp"
#include "mcts.hpp"
#include <mutex>
#include <deque>
#include <utility>
#include <functional>

std::mutex insert_mutex;
namespace py = pybind11;

template<typename T>
void pop_front(std::vector<T>& vec)
{
    assert(!vec.empty());
    vec.erase(vec.begin());
}


MonteCarloTreeSearch::MonteCarloTreeSearch()
{}

Node* MonteCarloTreeSearch::safe_insert_node(Node* n, const int action, const double score, const int num_actions, const int next_agent_idx)
{
    const std::lock_guard<std::mutex> lock(insert_mutex);
    all_nodes.emplace_back(n, action, score, num_actions, next_agent_idx);
    return &all_nodes.back();
}

double MonteCarloTreeSearch::single_simulation(const int process_num)
{
    penvs[process_num].reset_seed();
    double score(0);
    double g(1), reward(0);
    int num_steps(0);
    while(!penvs[process_num].all_done() && num_steps < cfg.steps_limit)
    {
        reward = penvs[process_num].step(penvs[process_num].sample_actions(cfg.num_actions, cfg.use_move_limits, cfg.agents_as_obstacles));
        num_steps++;
        score += reward*g;
        g *= cfg.gamma;
    }
    for (int i = 0; i < num_steps; i++)
    {
        penvs[process_num].step_back();
    }
    return score;
}

double MonteCarloTreeSearch::simulation(const int process_num = 0)
{
    double score(0);
    if (cfg.multi_simulations > 1)
    {
        std::vector<std::future<double>> futures;
        for(int thread = 0; thread < cfg.multi_simulations; thread++)
        {
            futures.push_back(pool.submit(&MonteCarloTreeSearch::single_simulation, this, thread));
        }
        for(int thread = 0; thread < cfg.multi_simulations; thread++)
        {
            score += futures[thread].get();
        }
    }
    else
    {
        score = single_simulation(process_num);
    }
    const auto result = score/cfg.multi_simulations;
    return result;
}

double MonteCarloTreeSearch::uct(Node* n, const int agent_idx, const int process_num) const
{
    auto uct_val = n->q + cfg.uct_c*std::sqrt(2.0*std::log(n->parent->cnt)/n->cnt);
    if (cfg.heuristic_coef > 0)
    {
        const auto position = penvs[process_num].cur_positions[agent_idx];
        const auto move = penvs[process_num].moves[n->action_id];
        const int lenpath = shortest_paths[agent_idx][position.first + move.first][position.second + move.second];
        uct_val -= cfg.heuristic_coef * lenpath / n->cnt;
    }
    return uct_val;
}

double MonteCarloTreeSearch::batch_uct(Node* n) const
{
    const int adjusted_count = n->cnt + n->cnt_sne;
    return n->w/adjusted_count + cfg.uct_c * std::sqrt(2.0 * std::log(n->parent->cnt + n->parent->cnt_sne)/adjusted_count);
}

int MonteCarloTreeSearch::expansion(Node* n, const int agent_idx, const int process_num = 0) const
{
    int best_action(0), k(0);
    double best_score(-1);
    for(auto c: n->child_nodes)
    {
        if ((cfg.use_move_limits && penvs[process_num].check_action(agent_idx, k, cfg.agents_as_obstacles)) || !cfg.use_move_limits)
        {
            if(c == nullptr)
            {
                return k;
            }
            const auto uct_val = uct(c, agent_idx, process_num);
            if (uct_val > best_score)
            {
                best_action = k;
                best_score = uct_val;
            }
        }
        k++;
    }
    return best_action;
}

double MonteCarloTreeSearch::selection(Node* n, std::vector<int> actions, const int process_num = 0)
{
    int agent_idx = int(actions.size())%penvs[process_num].get_num_agents();
    int next_agent_idx = (agent_idx + 1)%penvs[process_num].get_num_agents();
    int action(0);
    double score;
    if(!penvs[process_num].reached_goal(agent_idx))
        action = expansion(n, agent_idx, process_num);
    if(actions.size() == penvs[process_num].get_num_agents())
    {
        double reward = penvs[process_num].step(actions);
        actions.clear();
        if(penvs[process_num].all_done())
            score = reward;
        else
        {
            if(n->child_nodes[action] == nullptr)
            {
                score = reward + cfg.gamma*simulation(process_num);
                n->child_nodes[action] = safe_insert_node(n, action, score, cfg.num_actions, next_agent_idx);
            }
            else
                score = reward +cfg.gamma*selection(n->child_nodes[action], {action}, process_num);
        }
        n->update_value(score);
        penvs[process_num].step_back();
    }
    else
    {
        if(n->child_nodes[action] == nullptr)
        {
            n->child_nodes[action] = safe_insert_node(n, action, 0, cfg.num_actions, next_agent_idx);
        }
        actions.push_back(action);
        score = selection(n->child_nodes[action], actions, process_num);
        n->update_value(score);
    }
    return score*cfg.gamma;
}

int MonteCarloTreeSearch::select_action_for_batch_path(Node* n, const int agent_idx, const int process_num = 0)
{
    int best_action(0), k(0);
    double best_score(-1);
    for(auto c: n->child_nodes)
    {
        if ((cfg.use_move_limits && penvs[process_num].check_action(agent_idx, k, cfg.agents_as_obstacles)) || !cfg.use_move_limits)
        {
            if(c == nullptr && !n->mask_picked[k])
                return k;
            else if (c == nullptr)
            {
                k++;
                continue;
            }
            const auto uct_val = batch_uct(c);
            if (uct_val > best_score)
            {
                best_action = k;
                best_score = uct_val;
            }
        }
        k++;
    }
    if (best_score < 0)
    {
        best_action = -1;
    }
    return best_action;
}

std::vector<int> MonteCarloTreeSearch::batch_selection(Node* n, std::vector<int> actions, const int process_num = 0)
{
    int agent_idx = int(actions.size())%penvs[process_num].get_num_agents();
    int action(0);
    if(!penvs[process_num].reached_goal(agent_idx))
        action = select_action_for_batch_path(n, agent_idx, process_num);
    actions.push_back(action);
    if (action < 0)
    {
        return actions;
    }
    if (n->child_nodes[action] == nullptr)
    {
        n->mask_picked[action] = true;
        n->cnt_sne += 1;
        return actions;
    }
    else
    {
        auto new_actions = batch_selection(n->child_nodes[action], actions, process_num);
        n->cnt_sne += 1;
        return new_actions;
    }
}

double MonteCarloTreeSearch::batch_expansion(std::vector<int> path_actions, std::vector<int> prev_actions, const int process_num = 0)
{
    double score = 0.0;
    double g = 1.0;
    if(prev_actions.size() == penvs[process_num].get_num_agents())
    {
        double reward = penvs[process_num].step(prev_actions);
        score += g * reward;
        g *= cfg.gamma;
        prev_actions.clear();
    }
    for (const auto action: path_actions)
    {
        prev_actions.push_back(action);
        if(prev_actions.size() == penvs[process_num].get_num_agents())
        {
            double reward = penvs[process_num].step(prev_actions);
            score += g * reward;
            g *= cfg.gamma;
            prev_actions.clear();
        }
    }
    if(!penvs[process_num].all_done())
    {
        score += cfg.gamma * simulation(process_num);
    }
    return score;
}

void MonteCarloTreeSearch::loop(std::vector<int>& prev_actions)
{
    for (int i = 0; i < cfg.num_expansions; i++)
    {
        double score = selection(root, prev_actions, 0);
        root->update_value(score);
    }
}

void MonteCarloTreeSearch::batch_loop(std::vector<int>& prev_actions)
{
    for (int i = 0; i < cfg.num_expansions; i++)
    {
        root->zero_snes();
        std::vector<std::future<double>> pool_futures;
        std::vector<std::vector<int>> batch_paths;
        for(int batch = 0; batch < cfg.batch_size; batch++)
        {
            auto batch_actions = batch_selection(root, prev_actions, batch);
            if (batch_actions[batch_actions.size() - 1] >= 0)
            {
                for([[maybe_unused]] auto& _ : prev_actions)
                    pop_front(batch_actions);
                batch_paths.push_back(batch_actions);
                pool_futures.push_back(pool.submit(&MonteCarloTreeSearch::batch_expansion, this, batch_actions, prev_actions, batch));
            }
        }
        for (size_t enum_paths = 0; enum_paths < batch_paths.size(); enum_paths++)
        {
            Node* local_root = root;
            for (size_t enum_actions = 0; enum_actions < batch_paths[enum_paths].size() - 1; enum_actions++)
            {
                local_root = local_root->child_nodes[batch_paths[enum_paths][enum_actions]];
            }
            const auto score = pool_futures[enum_paths].get();
            const auto action = batch_paths[enum_paths][batch_paths[enum_paths].size() - 1];
            if(local_root->child_nodes[action] == nullptr)
            {
                local_root->child_nodes[action] = safe_insert_node(local_root, action, score, cfg.num_actions, (local_root->agent_id + 1) % penvs[0].get_num_agents());
                local_root->update_value_batch(score);
            }
            else
            {
                local_root->child_nodes[action]->update_value_batch(score);
            }
        }
    }
}

void MonteCarloTreeSearch::retrieve_statistics(Node* tree, Node* from_root)
{
    from_root->cnt += tree->cnt;
    from_root->w += tree->w;
    int action(0);
    for(auto c: tree->child_nodes)
    {
        if(c != nullptr)
        {
            if(from_root->child_nodes[action] == nullptr)
            {
                from_root->child_nodes[action] = safe_insert_node(from_root, action, 0, cfg.num_actions, c->agent_id);
            }
            retrieve_statistics(c, from_root->child_nodes[action]);
        }
        action++;
    }
}

void MonteCarloTreeSearch::tree_parallelization_loop_internal(std::vector<int> prev_actions, const int process_num)
{
    for (int i = 0; i < cfg.num_expansions; i++)
    {
        double score = selection(ptrees[process_num], prev_actions, process_num);
        ptrees[process_num]->update_value(score);
    }
}

void MonteCarloTreeSearch::tree_parallelization_loop(std::vector<int>& prev_actions)
{
    std::vector<std::future<void>> futures;
    for(int i = 0; i < cfg.num_parallel_trees; i++)
    {
        futures.push_back(pool.submit(&MonteCarloTreeSearch::tree_parallelization_loop_internal, this, prev_actions, i));
    }
    futures[0].get();
    for(int i = 1; i < cfg.num_parallel_trees; i++)
    {
        futures[i].get();
        retrieve_statistics(ptrees[i], root);
    }
    root->update_q();
}

std::vector<int> MonteCarloTreeSearch::act()
{
    std::vector<int> actions;
    if (penvs[0].all_done())
    {
        for(size_t agent_idx = 0; agent_idx < penvs[0].get_num_agents(); agent_idx++)
        {
            actions.push_back(0);
        }
        return actions;
    }
    std::vector<char> action_names = {'S','U', 'D', 'L', 'R'};
    for(size_t agent_idx = 0; agent_idx < penvs[0].get_num_agents(); agent_idx++)
    {
        try
        {
            if (!penvs[0].reached_goal(agent_idx))
            {
                if (cfg.batch_size > 1)
                {
                    batch_loop(actions);
                }
                else if (cfg.num_parallel_trees > 1)
                {
                    tree_parallelization_loop(actions);
                }
                else
                {
                    loop(actions);
                }
            }
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            std::cout<<"loop\n";
        }

        if (cfg.render)
        {
            std::cout<<agent_idx<<" "<<root->q<<std::endl;
            for(int i = 0; i < cfg.num_actions; i++) {
                int cnt = (root->child_nodes[i] == nullptr) ? 0 : root->child_nodes[i]->cnt;
                std::cout << action_names[i] << ":" << cnt << " ";
            }
            std::cout<<std::endl;
            for(int i = 0; i < cfg.num_actions; i++) {
                double c = (root->child_nodes[i] == nullptr) ? 0.0 : uct(root->child_nodes[i], agent_idx, 0);
                std::cout << action_names[i] << ":" << c << " ";
            }
            std::cout<<std::endl;
            std::cout<<"---------------------------------------------------------------------\n";
        }
        int action = root->get_action();
        root = root->child_nodes[action];
        for(int i = 0; i < cfg.num_parallel_trees; i++)
        {
            if(ptrees[i]->child_nodes[action] != nullptr)
            {
                ptrees[i] = ptrees[i]->child_nodes[action];
            }
            else
            {
                ptrees[i]->child_nodes[action] = safe_insert_node(ptrees[i], action, 0, cfg.num_actions, (agent_idx + 1) % penvs[i].get_num_agents());
            }
        }
        actions.push_back(action);
    }

    for(int i = 0; i < num_envs; i++)
    {
        penvs[i].step(actions);
    }
    if (cfg.render)
    {
        for(auto a: actions)
            std::cout<<a<<" ";
        std::cout<<" actions\n";
    }
    return actions;
}

void MonteCarloTreeSearch::set_config(const Config& config)
{
    cfg = config;
}

void MonteCarloTreeSearch::set_env(Environment env)
{
    for(int i = 0; i < cfg.num_parallel_trees; i++)
    {
        ptrees.push_back(safe_insert_node(nullptr, -1, 0, cfg.num_actions, 0));
    }
    num_envs = std::max({cfg.num_parallel_trees, cfg.batch_size, cfg.multi_simulations});
    for(int i = 0; i < num_envs; i++)
    {
        penvs.push_back(env);
    }
    root = ptrees[0];
    if (cfg.heuristic_coef > 0)
    {
        shortest_paths = bfs(env);
    }
}

std::vector<std::vector<std::vector<int>>> MonteCarloTreeSearch::bfs(Environment& env)
{
    auto obstacles = env.grid;

    std::vector<std::vector<std::vector<int>>> agents_map;
    agents_map.reserve(env.num_agents);

    for(size_t i = 0; i < env.num_agents; i++)
    {
        auto filled = obstacles;
        for(size_t j = 0; j < filled.size(); j++)
        {
            for(size_t k = 0; k < filled[0].size(); k++)
            {
                if(filled[j][k] >= 0)
                {
                    filled[j][k] = 1000000;
                }
            }
        }
        filled[env.goals[i].first][env.goals[i].second] = 0;
        std::deque<std::pair<int, int>> q;
        q.push_back(env.goals[i]);
        while (q.size() > 0)
        {
            auto pos = q.front();
            q.pop_front();
            for(const auto& move: env.moves)
            {
                if ((pos.first + move.first >= 0) && (static_cast<size_t>(pos.first + move.first) < filled.size())\
                         && (pos.second + move.second >= 0) && (static_cast<size_t>(pos.second + move.second) < filled[0].size()))
                {
                    if ((filled[pos.first + move.first][pos.second + move.second] == 1000000) && env.grid[pos.first + move.first][pos.second + move.second] != 1)
                    {
                        q.push_back(std::make_pair(pos.first + move.first, pos.second + move.second));
                        filled[pos.first + move.first][pos.second + move.second] = filled[pos.first][pos.second] + 1;
                    }
                }
            }
        }
        agents_map.push_back(filled);
    }
    return agents_map;
}

PYBIND11_MODULE(mcts, m) {
    py::class_<MonteCarloTreeSearch>(m, "MonteCarloTreeSearch")
            .def(py::init<>())
            .def("act", &MonteCarloTreeSearch::act)
            .def("set_config", &MonteCarloTreeSearch::set_config)
            .def("set_env", &MonteCarloTreeSearch::set_env)
            ;
}

/*
<%
cfg['extra_compile_args'] = ['-std=c++17']
setup_pybind11(cfg)
%>
*/