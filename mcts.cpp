#ifdef __APPLE__
#else
    #include "omp.h"
#endif
#include "BS_thread_pool.hpp"
#include "mcts.hpp"

template<typename T>
void pop_front(std::vector<T>& vec)
{
    assert(!vec.empty());
    vec.erase(vec.begin());
}


MonteCarloTreeSearch::MonteCarloTreeSearch(const std::string& fileName, int seed):env(fileName, seed)
{
    all_nodes.emplace_back(nullptr, -1, 0, cfg.num_actions, 0);
    root = &all_nodes.back();
}

double MonteCarloTreeSearch::simulation(Environment& local_env)
{
    double score(0);
    #ifdef __APPLE__
    #else
    omp_set_num_threads(cfg.multi_simulations);
    #pragma omp parallel for reduction(+: score) firstprivate(local_env, cfg)
    #endif
    for(int thread = 0; thread < cfg.multi_simulations; thread++)
    {
        local_env.reset_seed();
        #ifdef __APPLE__
        #else
        const int thread_num = omp_get_thread_num();
        #endif
        double g(1), reward(0);
        int num_steps(0);
        while(!local_env.all_done() && num_steps < cfg.steps_limit)
        {
            reward = local_env.step(local_env.sample_actions(cfg.num_actions, cfg.use_move_limits, cfg.agents_as_obstacles));
            num_steps++;
            score += reward*g;
            g *= cfg.gamma;
        }
        #ifdef __APPLE__
        for(int i = 0; i < num_steps; i++)
        {
            local_env.step_back();
        }
        #endif
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
        if (cfg.use_move_limits && env.check_action(agent_idx, k, cfg.agents_as_obstacles) || !cfg.use_move_limits)
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
                all_nodes.emplace_back(n, action, score, cfg.num_actions, next_agent_idx);
                n->child_nodes[action] = &all_nodes.back();
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
            all_nodes.emplace_back(n, action, 0, cfg.num_actions, next_agent_idx);
            n->child_nodes[action] = &all_nodes.back();
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
        if ((cfg.use_move_limits && env.check_action(agent_idx, k, cfg.agents_as_obstacles) || !cfg.use_move_limits))
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
                for(auto _: prev_actions)
                    pop_front(batch_actions);
                batch_paths.push_back(batch_actions);
                pool_futures.push_back(pool.submit(&MonteCarloTreeSearch::batch_expansion, this, batch_actions, prev_actions, env));
            }
        }
        for (int enum_paths = 0; enum_paths < batch_paths.size(); enum_paths++)
        {
            Node* local_root = root;
            for (int enum_actions = 0; enum_actions < batch_paths[enum_paths].size() - 1; enum_actions++)
            {
                local_root = local_root->child_nodes[batch_paths[enum_paths][enum_actions]];
            }
            const auto score = pool_futures[enum_paths].get();
            const auto action = batch_paths[enum_paths][batch_paths[enum_paths].size() - 1];
            if(local_root->child_nodes[action] == nullptr)
            {
                all_nodes.emplace_back(local_root, action, score, cfg.num_actions, (local_root->agent_id + 1) % env.get_num_agents());
                local_root->child_nodes[action] = &all_nodes.back();
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
                all_nodes.emplace_back(from_root, action, 0, cfg.num_actions, c->agent_id);
                from_root->child_nodes[action] = &all_nodes.back();
            }
            retrieve_statistics(c, from_root->child_nodes[action]);
        }
        action++;
    }
}

Node MonteCarloTreeSearch::tree_parallelization_loop_internal(Node root, std::vector<int> prev_actions, Environment cpenv)
{
    for (int i = 0; i < cfg.num_expansions; i++)
    {
        double score = selection(&root, prev_actions, cpenv);
        root.update_value(score);
    }
    return root;
}

void MonteCarloTreeSearch::tree_parallelization_loop(std::vector<int>& prev_actions)
{
    std::vector<std::future<Node>> futures;
    for(int i = 0; i < cfg.num_parallel_trees; i++)
    {
        futures.push_back(pool.submit(&MonteCarloTreeSearch::tree_parallelization_loop_internal, this, *root, prev_actions, env));
    }
    for(int i = 0; i < cfg.num_parallel_trees; i++)
    {
        auto tree = futures[i].get();
        retrieve_statistics(&tree, root);
    }
    root->update_q();
}

bool MonteCarloTreeSearch::act()
{
    env.render();
    std::vector<int> actions;
    std::vector<char> action_names = {'S','U', 'D', 'L', 'R'};
    for(int agent_idx = 0; agent_idx < env.get_num_agents(); agent_idx++)
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
        int action = root->get_action(env);
        root = root->child_nodes[action];

        actions.push_back(action);
    }
    for(auto a: actions)
        std::cout<<a<<" ";
    std::cout<<" actions\n";
    env.step(actions);
    if(env.all_done())
        env.render();
    return env.all_done();
}