#include <iostream>
#include <list>
#include <vector>
#include <numeric>
#include <cmath>
#include <string>
#include "environment.h"
#include "config.h"
struct Node
{
    int action_id;
    Node* parent;
    int cnt;
    double w;
    double q;
    std::vector<Node*> child_nodes;
    int agent_id;
    Node(Node* _parent, int _action_id, double _w, int num_actions, int _agent_id=-1)
            :parent(_parent), action_id(_action_id), w(_w), agent_id(_agent_id)
    {
        cnt = 1;
        q = w;
        child_nodes.resize(num_actions, nullptr);
    }
    void update_value(double value)
    {
        w += value;
        cnt++;
        q = w/cnt;
    }
    int get_action(const Environment& env)
    {
        int best_action(0), best_score(-1), k(0);
        for(auto c:child_nodes) {
            if (c != nullptr && c->cnt > best_score)
            {
                best_action = k;
                best_score = c->cnt;
            }
            k++;
        }
        return best_action;
    }
};

class MonteCarloTreeSearch
{
    Node* root;
    std::list<Node> all_nodes;
    Environment env;
    Config cfg;
public:
    explicit MonteCarloTreeSearch(const std::string& fileName, int seed = -1):env(fileName, seed)
    {
        all_nodes.emplace_back(nullptr, -1, 0, cfg.num_actions, 0);
        root = &all_nodes.back();
    }
    double simulation()
    {
        double score(0), g(1), reward(0);
        int num_steps(0);
        while(!env.all_done() && num_steps < cfg.steps_limit)
        {
            reward = env.step(env.sample_actions(cfg.num_actions));
            num_steps++;
            score += reward*g;
            g *= cfg.gamma;
        }
        for(int i = 0; i < num_steps; i++)
            env.step_back();
        return score;
    }
    double uct(Node* n) const
    {
        return n->q + cfg.uct_c*std::sqrt(2.0*std::log(n->parent->cnt)/n->cnt);
    }
    int expansion(Node* n) const
    {
        int best_action(0), k(0);
        double best_score(-1);
        for(auto c:n->child_nodes) {
            if(c == nullptr)
                return k;
            if (uct(c) > best_score) {
                best_action = k;
                best_score = uct(c);
            }
            k++;
        }
        return best_action;
    }
    double selection(Node* n, std::vector<int> actions)
    {
        int agent_idx = int(actions.size())%env.get_num_agents();
        int next_agent_idx = (agent_idx + 1)%env.get_num_agents();
        int action(0);
        double score;
        if(!env.reached_goal(agent_idx))
            action = expansion(n);
        if(actions.size() == env.get_num_agents())
        {
            double reward = env.step(actions);
            if(env.all_done())
                score = reward;
            else
            {
                if(n->child_nodes[action] == nullptr)
                {
                    score = reward + cfg.gamma*simulation();
                    all_nodes.emplace_back(n, action, score, cfg.num_actions, next_agent_idx);
                    n->child_nodes[action] = &all_nodes.back();
                }
                else
                    score = reward +cfg.gamma*selection(n->child_nodes[action], {action});
            }
            n->update_value(score);
            env.step_back();
        }
        else
        {
            if(n->child_nodes[action]== nullptr)
            {
                all_nodes.emplace_back(n, action, 0, cfg.num_actions, next_agent_idx);
                n->child_nodes[action] = &all_nodes.back();
            }
            actions.push_back(action);
            score = selection(n->child_nodes[action], actions);
            n->update_value(score);
        }
        return score*cfg.gamma;
    }
    void loop()
    {
        for (int i = 0; i < cfg.num_expansions; i++) {
            double score = selection(root, {});
            root->update_value(score);
        }
        return;
        auto cur_root = root;
        for(int k = 0 ; k < env.get_num_agents(); k++) {
            if(!env.reached_goal(k))
                for (int i = 0; i < cfg.num_expansions; i++) {
                    double score = selection(cur_root, {});
                    cur_root->update_value(score);
                    /*std::cout<<i<<" "<<root->cnt<<" "<<score<<"  ";
                    for(auto a: root->child_nodes)
                        if(a != nullptr)
                            std::cout<<" "<<a->cnt;
                    std::cout<<std::endl;*/
                }
            cur_root = cur_root->child_nodes[cur_root->get_action(env)];
        }
    }
    bool act()
    {
        loop();
        env.render();
        std::vector<int> actions;
        std::vector<char> action_names = {'S','U', 'D', 'L', 'R'};
        for(int agent_idx = 0; agent_idx < env.get_num_agents(); agent_idx++)
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
            int action = root->get_action(env);
            root = root->child_nodes[action];

            actions.push_back(action);
        }
        for(auto a:actions)
            std::cout<<a<<" ";
        std::cout<<" actions\n";
        env.step(actions);
        if(env.all_done())
            env.render();
        return env.all_done();
    }
};