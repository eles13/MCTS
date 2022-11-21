#ifndef MCTS_ENVIRONMENT_H
#define MCTS_ENVIRONMENT_H
#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include "tinyxml2.h"
#define OBSTACLE 1
#define TRAVERSABLE 0
using namespace tinyxml2;

class Environment
{
    int num_agents;
    std::vector<std::pair<int, int>> moves = {{0,0}, {-1, 0}, {1,0},{0,-1},{0,1}};
    std::vector<std::vector<int>> grid;
    std::vector<std::pair<int, int>> goals;
    std::vector<std::pair<int, int>> cur_positions;
    std::vector<std::vector<int>> made_actions;
    std::vector<bool> reached;
    std::default_random_engine engine;
public:
    explicit Environment(const std::string& fileName, int seed)
    {
        load_instance(fileName.c_str());
        num_agents = goals.size();
        reached.resize(num_agents, false);
        set_seed(seed);
    }

    void set_seed(const int seed)
    {
        if(seed < 0)
            engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
        else
            engine.seed(seed);
    }

    void reset_seed()
    {
        engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }

    int get_num_agents()
    {
        return num_agents;
    }

    bool reached_goal(int i)
    {
        if(i >= 0 && i < num_agents)
            return reached[i];
        else
            return false;
    }

    void load_instance(const char* fileName)
    {
        XMLDocument doc;
        if(doc.LoadFile(fileName) != XMLError::XML_SUCCESS)
        {
            std::cout << "Error openning input XML file."<<std::endl;
            return;
        }
        XMLElement* root;
        root = doc.FirstChildElement("root");
        for(auto elem = root->FirstChildElement("agent"); elem; elem = elem->NextSiblingElement("agent"))
        {
            cur_positions.emplace_back(elem->IntAttribute("start_i"), elem->IntAttribute("start_j"));
            goals.emplace_back(elem->IntAttribute("goal_i"), elem->IntAttribute("goal_j"));
        }
        XMLElement* map = root->FirstChildElement("map");
        grid = std::vector<std::vector<int>>(map->IntAttribute("width"), std::vector<int>(map->IntAttribute("height"),TRAVERSABLE));
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
                    grid[curi][curj] = OBSTACLE;
                curj++;
            }
            curi++;
        }
    }

    double step(std::vector<int> actions)
    {
        std::vector<std::pair<int, int>> executed_pos;
        for(int i = 0; i < num_agents; i++) {
            if (reached[i])
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
                if (reached[i] || reached[j])
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
            }
        for(int i = 0; i < num_agents; i++) {
            if (reached[i])
                continue;
            if(executed_pos[i].first == goals[i].first && executed_pos[i].second == goals[i].second)
            {
                reward += 1;
                reached[i] = true;
            }
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
                reached[i] = false;
        }
        made_actions.pop_back();
    }

    std::vector<int> sample_actions(int num_actions, const bool use_move_limits=false, const bool agents_as_obstackles=false)
    {
        std::vector<int> actions;
        for(int i = 0; i < num_agents; i++)
        {
            auto action = engine() % num_actions;
            if (use_move_limits)
            {
                while (!check_action(i, action, agents_as_obstackles))
                    action = engine() % num_actions;
            }
            actions.emplace_back(action);
        }
        return actions;
    }

    bool all_done()
    {
        return num_agents == std::accumulate(reached.begin(), reached.end(), 0);
    }

    void render()
    {
        engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
        for(int i = 0; i < num_agents; i++) {
            auto c1 = cur_positions[i], c2 = goals[i];
            if(c1.first != c2.first || c1.second != c2.second)
            {
                grid[c1.first][c1.second] = i + 2;
                grid[c2.first][c2.second] = i + 2 + num_agents;
            }
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
    }

    const bool check_action(const int agent_idx, const int action, const bool agents_as_obstacles) const
    {
        const std::pair<int, int> future_position = {cur_positions[agent_idx].first + moves[action].first, cur_positions[agent_idx].second + moves[action].second};
        if (future_position.first < 0 || future_position.second < 0 || future_position.first >= grid.size() || future_position.second >= grid.size())
            return false;
        if (grid[future_position.first][future_position.second] == 1)
            return false;
        if (agents_as_obstacles)
        {
            for (int i = 0; i < num_agents; i++)
            {
                if (i != agent_idx)
                {
                    if((cur_positions[i].first == future_position.first) && (cur_positions[i].second == future_position.second))
                        return false;
                }
            }
        }
        return true;
    }

    Environment(const Environment& orig)
    {
        num_agents = orig.num_agents;
        moves = orig.moves;
        grid = orig.grid;
        goals = orig.goals;
        cur_positions = orig.cur_positions;
        made_actions = orig.made_actions;
        reached = orig.reached;
        engine = orig.engine;
        reset_seed();
    }
};

#endif //MCTS_ENVIRONMENT_H
