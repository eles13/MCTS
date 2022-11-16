#include <iostream>
#include <list>
#include <vector>
#include <numeric>
#include <cmath>
#include <string>
#include <random>
#include <chrono>
#include "tinyxml2.h"
#define OBSTACLE 1
#define TRAVERSABLE 0
using namespace tinyxml2;

struct Map
{
    int width;
    int height;
    std::vector<std::vector<int>> grid;
    explicit Map(int _width=0, int _height=0):width(_width), height(_height)
    {
        grid = std::vector<std::vector<int>>(height, std::vector<int>(width, TRAVERSABLE));
    }
    void add_obstacle(int i, int j)
    {
        grid[i][j] = OBSTACLE;
    }
};

class Loader
{
public:
    std::vector<std::pair<int, int>> starts;
    std::vector<std::pair<int, int>> goals;
    Map grid;
    bool load_instance(const char* fileName)
    {
        XMLDocument doc;
        if(doc.LoadFile(fileName) != XMLError::XML_SUCCESS)
        {
            std::cout << "Error openning input XML file."<<std::endl;
            return false;
        }
        XMLElement* root;
        root = doc.FirstChildElement("root");
        for(auto elem = root->FirstChildElement("agent"); elem; elem = elem->NextSiblingElement("agent"))
        {
            starts.emplace_back(elem->IntAttribute("start_i"), elem->IntAttribute("start_j"));
            goals.emplace_back(elem->IntAttribute("goal_i"), elem->IntAttribute("goal_j"));
        }
        XMLElement* map = root->FirstChildElement("map");
        grid = Map(map->IntAttribute("width"), map->IntAttribute("height"));
        int curi(0), curj(0);
        for(auto row = map->FirstChildElement("row"); row; row = row->NextSiblingElement("row"))
        {
            std::string values = row->GetText();
            curj = 0;
            for(char value : values)
            {
                if(value == ' ')
                    continue;
                if(value == '1')
                    grid.add_obstacle(curi, curj);
                curj++;
            }
            curi++;
        }
        return true;
    }
};

struct Config
{
    double gamma = 0.95;
    int num_actions = 5;
    int num_expansions = 1000;
    double uct_c = 1.0;
    int steps_limit = 128;
};

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

struct Environment
{
    int num_agents;
    std::vector<std::pair<int, int>> moves = {{0,0}, {-1, 0}, {1,0},{0,-1},{0,1}};
    std::vector<std::vector<int>> grid;
    std::vector<std::pair<int, int>> goals;
    std::vector<std::pair<int, int>> cur_positions;
    std::vector<std::vector<int>> made_actions;
    std::vector<bool> reached_goal;
    std::default_random_engine engine;
    explicit Environment(const std::string& fileName)
    {
        Loader loader;
        loader.load_instance(fileName.c_str());
        grid = loader.grid.grid;
        cur_positions = loader.starts;
        goals = loader.goals;
        num_agents = goals.size();
        reached_goal.resize(num_agents, false);
        engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }
    double step(std::vector<int> actions)
    {
        std::vector<std::pair<int, int>> executed_pos;
        for(int i = 0; i < num_agents; i++) {
            if (reached_goal[i])
            {
                executed_pos.push_back(cur_positions[i]);
                actions[i] = 0;
            }
            else
                executed_pos.emplace_back(cur_positions[i].first + moves[actions[i]].first,
                                    cur_positions[i].second + moves[actions[i]].second);
        }
        for(int i = 0; i < num_agents; i++)
            for(int j = i+1; j < num_agents; j++) {
                if (reached_goal[i] || reached_goal[j])
                    continue;
                if ((executed_pos[i].first == executed_pos[j].first &&
                     executed_pos[i].second == executed_pos[j].second) ||
                    (executed_pos[i].first == cur_positions[j].first &&
                     executed_pos[i].second == cur_positions[j].second
                     && executed_pos[j].first == cur_positions[i].first &&
                     executed_pos[j].second == cur_positions[i].second))
                {
                    executed_pos[i] = cur_positions[i];
                    executed_pos[j] = cur_positions[j];
                    actions[i] = 0;
                    actions[j] = 0;
                }
            }
        double reward(0);
        for(int i = 0; i < num_agents; i++)
            if(executed_pos[i].first < 0 || executed_pos[i].first >= grid.size() ||
            executed_pos[i].second < 0 || executed_pos[i].second >= grid[0].size()
            || grid[executed_pos[i].first][executed_pos[i].second])
            {
                executed_pos[i] = cur_positions[i];
                actions[i] = 0;
                //reward -= 0.1;
            }
        for(int i = 0; i < num_agents; i++) {
            //if (reached_goal[i])
            //    continue;
            if(executed_pos[i].first == goals[i].first && executed_pos[i].second == goals[i].second)
            {
                reward += 1;
                reached_goal[i] = true;
            }
            //int old_h = abs(cur_positions[i].first - goals[i].first) + abs(cur_positions[i].second - goals[i].second);
            //int new_h = abs(executed_pos[i].first - goals[i].first) + abs(executed_pos[i].second - goals[i].second);
            //reward += old_h - new_h;
        }
        made_actions.push_back(actions);
        cur_positions = executed_pos;
        return reward;
    }
    void step_back()
    {
        for(int i = 0; i < num_agents; i++)
        {
            cur_positions[i].first = cur_positions[i].first - moves[made_actions.back()[i]].first;
            cur_positions[i].second = cur_positions[i].second - moves[made_actions.back()[i]].second;
            if(cur_positions[i].first != goals[i].first || cur_positions[i].second != goals[i].second)
                reached_goal[i] = false;
        }
        made_actions.pop_back();
    }
    std::vector<int> sample_actions(int num_actions)
    {
        std::vector<int> actions;
        for(int i = 0; i < num_agents; i++) {
            actions.emplace_back(engine() % num_actions);
            /*if (engine() % 2 == 1) {
                std::pair<int, int> right_move = {sgn(goals[i].first - cur_positions[i].first),
                                                  sgn(goals[i].second - cur_positions[i].second)};
                for(int i = 0; i < moves.size(); i++)
                    if(moves[i].first == right_move.first && moves[i].second == right_move.second)
                    {
                        actions.back() = i;
                        break;
                    }
            }*/
        }
        return actions;
    };
    bool all_done()
    {
        return num_agents == std::accumulate(reached_goal.begin(), reached_goal.end(), 0);
    }
    void render()
    {
        engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
        for(int i = 0; i < num_agents; i++) {
            auto c = cur_positions[i];
            grid[c.first][c.second] = i + 2;
            c = goals[i];
            grid[c.first][c.second] = i + 2 + num_agents;
        }
        for(int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == 0)
                    std::cout << " . ";
                else if (grid[i][j] == 1)
                    std::cout << " # ";
                else {
                    if (grid[i][j] > num_agents + 1)
                        std::cout << "|" << grid[i][j] - 2 - num_agents << "|";
                    else
                        std::cout << " " << grid[i][j] - 2 << " ";
                    grid[i][j] = 0;
                }
            }
            std::cout<<std::endl;
        }
    };

};

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
                //if(env.grid[env.cur_positions[c->agent_id].first + env.moves[c->action_id].first]
                //[env.cur_positions[c->agent_id].second + env.moves[c->action_id].second] == 0)
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
    explicit MonteCarloTreeSearch(const std::string& fileName):env(fileName)
    {
        all_nodes.emplace_back(nullptr, -1, 0, cfg.num_actions, 0);
        root = &all_nodes.back();
    }
    double simulation()
    {
        double score(0), g(1), reward(0);
        int num_steps(0);
        //std::cout<<env.cur_positions[0].first<<" "<<env.cur_positions[0].second<<" vs ";
        while(!env.all_done() && num_steps < cfg.steps_limit)
        {
            reward = env.step(env.sample_actions(cfg.num_actions));
            num_steps++;
            score += reward*g;
            g *= cfg.gamma;
        }
        //std::cout<<"score numsteps "<<score<<" "<<num_steps<<"\n";
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
        int agent_idx = int(actions.size())%env.num_agents;
        int next_agent_idx = (agent_idx + 1)%env.num_agents;
        int action(0);
        double score;
        if(!env.reached_goal[agent_idx])
            action = expansion(n);
        if(actions.size() == env.num_agents)
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
        for(int k = 0 ; k < env.num_agents; k++) {
            if(!env.reached_goal[k])
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
        for(int agent_idx = 0; agent_idx < env.num_agents; agent_idx++)
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