// cppimport
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "BS_thread_pool.hpp"
#include "mcts.hpp"
#include <mutex>
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

double MonteCarloTreeSearch::single_simulation(Environment local_env)
{
    local_env.reset_seed();
    double score(0);
    double g(1), reward(0);
    int num_steps(0);
    while(!local_env.all_done() && num_steps < cfg.steps_limit)
    {
        reward = local_env.step(local_env.sample_actions(cfg.num_actions, cfg.use_move_limits, cfg.agents_as_obstacles));
        num_steps++;
        score += reward*g;
        g *= cfg.gamma;
    }
    return score;
}

double MonteCarloTreeSearch::simulation(Environment& local_env)
{
    double score(0);
    if (cfg.multi_simulations > 1)
    {
        std::vector<std::future<double>> futures;
        for(int thread = 0; thread < cfg.multi_simulations; thread++)
        {
            futures.push_back(pool.submit(&MonteCarloTreeSearch::single_simulation, this, local_env));
        }
        for(int thread = 0; thread < cfg.multi_simulations; thread++)
        {
            score += futures[thread].get();
        }
    }
    else
    {
        score = single_simulation(local_env);
    }
    const auto result = score/cfg.multi_simulations;
    return result;
}

double MonteCarloTreeSearch::uct(Node* n) const
{
    return n->q + cfg.uct_c*std::sqrt(2.0*std::log(n->parent->cnt)/n->cnt);
}

double MonteCarloTreeSearch::batch_uct(Node* n) const
{
    const int adjusted_count = n->cnt + n->cnt_sne;
    return n->w/adjusted_count + cfg.uct_c * std::sqrt(2.0 * std::log(n->parent->cnt + n->parent->cnt_sne)/adjusted_count);
}

int MonteCarloTreeSearch::expansion(Node* n, const int agent_idx) const
{
    int best_action(0), k(0);
    double best_score(-1);
    for(auto c: n->child_nodes)
    {
        if ((cfg.use_move_limits && env.check_action(agent_idx, k, cfg.agents_as_obstacles)) || !cfg.use_move_limits)
        {
            if(c == nullptr)
            {
                return k;
            }
            if (uct(c) > best_score) {
                best_action = k;
                best_score = uct(c);
            }
        }
        k++;
    }
    return best_action;
}

double MonteCarloTreeSearch::selection(Node* n, std::vector<int> actions, Environment& env)
{
    int agent_idx = int(actions.size())%env.get_num_agents();
    int next_agent_idx = (agent_idx + 1)%env.get_num_agents();
    int action(0);
    double score;
    if(!env.reached_goal(agent_idx))
        action = expansion(n, agent_idx);
    if(actions.size() == env.get_num_agents())
    {
        double reward = env.step(actions);
        actions.clear();
        if(env.all_done())
            score = reward;
        else
        {
            if(n->child_nodes[action] == nullptr)
            {
                score = reward + cfg.gamma*simulation(env);
                n->child_nodes[action] = safe_insert_node(n, action, score, cfg.num_actions, next_agent_idx);
            }
            else
                score = reward +cfg.gamma*selection(n->child_nodes[action], {action}, env);
        }
        n->update_value(score);
        env.step_back();
    }
    else
    {
        if(n->child_nodes[action] == nullptr)
        {
            n->child_nodes[action] = safe_insert_node(n, action, 0, cfg.num_actions, next_agent_idx);
        }
        actions.push_back(action);
        score = selection(n->child_nodes[action], actions, env);
        n->update_value(score);
    }
    return score*cfg.gamma;
}

int MonteCarloTreeSearch::select_action_for_batch_path(Node* n, const int agent_idx)
{
    int best_action(0), k(0);
    double best_score(-1);
    for(auto c: n->child_nodes)
    {
        if ((cfg.use_move_limits && env.check_action(agent_idx, k, cfg.agents_as_obstacles)) || !cfg.use_move_limits)
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

std::vector<int> MonteCarloTreeSearch::batch_selection(Node* n, std::vector<int> actions)
{
    int agent_idx = int(actions.size())%env.get_num_agents();
    int action(0);
    if(!env.reached_goal(agent_idx))
        action = select_action_for_batch_path(n, agent_idx);
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
        auto new_actions = batch_selection(n->child_nodes[action], actions);
        n->cnt_sne += 1;
        return new_actions;
    }
}

double MonteCarloTreeSearch::batch_expansion(std::vector<int> path_actions, std::vector<int> prev_actions, Environment cpenv)
{
    double score = 0.0;
    double g = 1.0;
    if(prev_actions.size() == cpenv.get_num_agents())
    {
        double reward = cpenv.step(prev_actions);
        score += g * reward;
        g *= cfg.gamma;
        prev_actions.clear();
    }
    for (const auto action: path_actions)
    {
        prev_actions.push_back(action);
        if(prev_actions.size() == cpenv.get_num_agents())
        {
            double reward = cpenv.step(prev_actions);
            score += g * reward;
            g *= cfg.gamma;
            prev_actions.clear();
        }
    }
    if(!env.all_done())
    {
        score += cfg.gamma * simulation(cpenv);
    }
    return score;
}

void MonteCarloTreeSearch::loop(std::vector<int>& prev_actions)
{
    for (int i = 0; i < cfg.num_expansions; i++)
    {
        double score = selection(root, prev_actions, this->env);
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
            auto batch_actions = batch_selection(root, prev_actions);
            if (batch_actions[batch_actions.size() - 1] >= 0)
            {
                for([[maybe_unused]] auto& _ : prev_actions)
                    pop_front(batch_actions);
                batch_paths.push_back(batch_actions);
                pool_futures.push_back(pool.submit(&MonteCarloTreeSearch::batch_expansion, this, batch_actions, prev_actions, env));
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
                local_root->child_nodes[action] = safe_insert_node(local_root, action, score, cfg.num_actions, (local_root->agent_id + 1) % env.get_num_agents());
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

void MonteCarloTreeSearch::tree_parallelization_loop_internal(Node* root, std::vector<int> prev_actions, Environment cpenv)
{
    for (int i = 0; i < cfg.num_expansions; i++)
    {
        double score = selection(root, prev_actions, cpenv);
        root->update_value(score);
    }
}

void MonteCarloTreeSearch::tree_parallelization_loop(std::vector<int>& prev_actions)
{
    std::vector<std::future<void>> futures;
    for(int i = 0; i < cfg.num_parallel_trees; i++)
    {
        futures.push_back(pool.submit(&MonteCarloTreeSearch::tree_parallelization_loop_internal, this, ptrees[i], prev_actions, env));
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
    try
    {
        if (env.all_done())
        {
            for(size_t agent_idx = 0; agent_idx < env.get_num_agents(); agent_idx++)
            {
                actions.push_back(0);
            }
            return actions;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        std::cout << "Forplay\n";
        throw;
    }
    std::vector<char> action_names = {'S','U', 'D', 'L', 'R'};
    for(size_t agent_idx = 0; agent_idx < env.get_num_agents(); agent_idx++)
    {
        try
        {
            if (!env.reached_goal(agent_idx))
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
                double c = (root->child_nodes[i] == nullptr) ? 0.0 : uct(root->child_nodes[i]);
                std::cout << action_names[i] << ":" << c << " ";
            }
            std::cout<<std::endl;
            std::cout<<"---------------------------------------------------------------------\n";
        }
        int action = root->get_action(env);
        root = root->child_nodes[action];
        for(int i = 0; i < cfg.num_parallel_trees; i++)
        {
            if(ptrees[i]->child_nodes[action] != nullptr)
            {
                ptrees[i] = ptrees[i]->child_nodes[action];
            }
            else
            {
                ptrees[i]->child_nodes[action] = safe_insert_node(ptrees[i], action, 0, cfg.num_actions, (agent_idx + 1) % env.get_num_agents());
            }
        }

        actions.push_back(action);
    }

    env.step(actions);
    if (cfg.render)
    {
        for(auto a: actions)
            std::cout<<a<<" ";
        std::cout<<" actions\n";
    }
    return actions;
}

Node* MonteCarloTreeSearch::make_copy_node(Node* orig)
{
    auto n = safe_insert_node(orig->parent, orig->action_id, orig->w, orig->num_actions_, orig->agent_id);
    n->mask_picked = orig->mask_picked;
    n->cnt = orig->cnt;
    n->cnt_sne = orig->cnt_sne;
    n->q = orig->q;
    for(size_t i = 0; i < orig->child_nodes.size(); i++)
    {
        if(orig->child_nodes[i] != nullptr)
        {
            n->child_nodes[i] = make_copy_node(orig->child_nodes[i]);
        }
    }
    return n;
}

void MonteCarloTreeSearch::set_config(const Config& config)
{
    cfg = config;
}

void MonteCarloTreeSearch::set_env(Environment& env_)
{
    env = env_;
    for(int i = 0; i < cfg.num_parallel_trees; i++)
    {
        ptrees.push_back(safe_insert_node(nullptr, -1, 0, cfg.num_actions, 0));
    }
    root = ptrees[0];
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