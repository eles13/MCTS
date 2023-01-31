// cppimport
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#define OBSTACLE 1
#define TRAVERSABLE 0
namespace py = pybind11;

class Environment
{
    std::vector<std::vector<int>> made_actions;
    std::vector<bool> reached;
    std::default_random_engine engine;
public:
    size_t num_agents;
    std::vector<std::pair<int, int>> moves = {{0,0}, {-1, 0}, {1,0},{0,-1},{0,1}};
    std::vector<std::pair<int, int>> goals;
    std::vector<std::pair<int, int>> cur_positions;
    std::vector<std::vector<int>> grid;
    explicit Environment()
    {
        num_agents = 0;
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

    size_t get_num_agents()
    {
        return num_agents;
    }

    void add_agent(int si, int sj, int gi, int gj)
    {
        cur_positions.push_back({si, sj});
        goals.push_back({gi, gj});
        num_agents++;
        reached.push_back(false);
    }

    void create_grid(int height, int width)
    {
        grid = std::vector<std::vector<int>>(height, std::vector<int>(width,TRAVERSABLE));
    }

    void add_obstacle(int i, int j)
    {
        grid[i][j] = OBSTACLE;
    }

    bool reached_goal(size_t i) const
    {
        if(i >= 0 && i < num_agents)
            return reached[i];
        else
            return false;
    }

    int get_num_done()
    {
        return std::accumulate(reached.begin(), reached.end(), 0);
    }

    double step(std::vector<int> actions)
    {
        std::vector<std::pair<int, int>> executed_pos;
        for(size_t i = 0; i < num_agents; i++) {
            if (reached[i])
            {
                executed_pos.push_back(cur_positions[i]);
                actions[i] = 0;
            }
            else
                executed_pos.emplace_back(cur_positions[i].first + moves[actions[i]].first,
                                          cur_positions[i].second + moves[actions[i]].second);
        }
        for(size_t i = 0; i < num_agents; i++)
            for(size_t j = i+1; j < num_agents; j++) {
                if (reached[i] || reached[j])
                    continue;
                if (executed_pos[i].first == executed_pos[j].first && executed_pos[i].second == executed_pos[j].second)
                {
                    executed_pos[i] = cur_positions[i];
                    executed_pos[j] = cur_positions[j];
                    actions[i] = 0;
                    actions[j] = 0;
                }
                if(executed_pos[i].first == cur_positions[j].first && executed_pos[i].second == cur_positions[j].second)
                {
                    executed_pos[i] = cur_positions[i];
                    actions[i] = 0;
                }
                if(executed_pos[j].first == cur_positions[i].first && executed_pos[j].second == cur_positions[i].second)
                {
                    executed_pos[j] = cur_positions[j];
                    actions[j] = 0;
                }
            }
        double reward(0);
        for(size_t i = 0; i < num_agents; i++)
            if(executed_pos[i].first < 0 || executed_pos[i].first >= static_cast<int>(grid.size()) ||
               executed_pos[i].second < 0 || executed_pos[i].second >= static_cast<int>(grid[0].size())
               || grid[executed_pos[i].first][executed_pos[i].second])
            {
                executed_pos[i] = cur_positions[i];
                actions[i] = 0;
            }
        for(size_t i = 0; i < num_agents; i++) {
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
        for(size_t i = 0; i < num_agents; i++)
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
        for(size_t i = 0; i < num_agents; i++)
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
        return static_cast<int>(num_agents) == std::accumulate(reached.begin(), reached.end(), 0);
    }

    void render()
    {
        for(size_t i = 0; i < num_agents; i++) {
            auto c1 = cur_positions[i], c2 = goals[i];
            if(c1.first != c2.first || c1.second != c2.second)
            {
                grid[c1.first][c1.second] = i + 2;
                grid[c2.first][c2.second] = i + 2 + num_agents;
            }
        }
        for(size_t i = 0; i < grid.size(); i++) {
            for (size_t j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == 0)
                    std::cout << " . ";
                else if (grid[i][j] == 1)
                    std::cout << " # ";
                else {
                    if (grid[i][j] > static_cast<int>(num_agents) + 1)
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
        //std::cout<<"ag_idx "<<agent_idx<<"\n";
        const std::pair<int, int> future_position = {cur_positions[agent_idx].first + moves[action].first, cur_positions[agent_idx].second + moves[action].second};
        //std::cout<<"gs "<<grid.size()<<" pos "<<future_position.first<<" "<<future_position.second<<"\n";
        if (future_position.first < 0 || future_position.second < 0 || future_position.first >= static_cast<int>(grid.size()) || future_position.second >= static_cast<int>(grid.size()))
            return false;
        if (grid[future_position.first][future_position.second] == 1)
            return false;
        if (agents_as_obstacles)
        {
            for (size_t i = 0; i < num_agents; i++)
            {
                if (static_cast<int>(i) != agent_idx)
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

PYBIND11_MODULE(environment, m) {
    py::class_<Environment>(m, "Environment")
            .def(py::init<>())
            .def("all_done", &Environment::all_done)
            .def("sample_actions", &Environment::sample_actions)
            .def("step", &Environment::step)
            .def("step_back", &Environment::step_back)
            .def("set_seed", &Environment::set_seed)
            .def("reset_seed", &Environment::reset_seed)
            .def("create_grid", &Environment::create_grid)
            .def("add_obstacle", &Environment::add_obstacle)
            .def("add_agent", &Environment::add_agent)
            .def("render", &Environment::render)
            .def("get_num_agents", &Environment::get_num_agents)
            .def("reached_goal", &Environment::reached_goal)
            ;
}

/*
<%
setup_pybind11(cfg)
%>
*/
