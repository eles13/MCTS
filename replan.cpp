// cppimport
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "planner.cpp"
#include "environment.cpp"
#include <mutex>
#include <deque>
#include <utility>
#include <functional>
#include <algorithm>
#include <list>
namespace py = pybind11;

class RePlan
{
    int num_agents;
    int obs_radius;
    bool fix_loops = false;
    double stay_if_loop_prob = 0.5;
    bool use_best_move = true;
    std::vector<std::pair<int, int>> moves = {{0,0}, {-1, 0}, {1,0},{0,-1},{0,1}};
    int steps = 0;
    int max_steps = 0;
    int seed;
    bool ignore_other_agents = false;
    std::vector<planner> planners;
    std::vector<std::vector<std::pair<int, int>>> previous_positions;
    std::default_random_engine engine;
    Environment env;

public:

    RePlan() {}
    void init(const int num_agents_, const int obs_radius_, const bool fix_loops_ = false, const double stay_if_loop_prob_ = 0.5,
           const bool use_best_move_ = true, const int max_steps_ = INT_MAX,
           const int seed_ = -1, const bool ignore_other_agents_ = false
    )
    {
        num_agents = num_agents_;
        obs_radius = obs_radius_;
        fix_loops = fix_loops_;
        stay_if_loop_prob = stay_if_loop_prob_;
        use_best_move = use_best_move_;
        max_steps = max_steps_;
        seed = seed_;
        ignore_other_agents = ignore_other_agents_;
        if(seed < 0)
            engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
        else
            engine.seed(seed);
        for(int i = 0; i < num_agents; i++)
        {
            planners.push_back(planner(max_steps));
        }
        previous_positions.reserve(num_agents);
        for(int i = 0; i < num_agents; i++)
        {
            previous_positions.push_back({});
        }
    }

    int _get_random_move(const int agent_idx, const Environment& env)
    {
        if(seed < 0)
            engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
        std::vector<int> to_shuffle = {1, 2, 3, 4};
        std::shuffle(std::begin(to_shuffle), std::end(to_shuffle), engine);
        return to_shuffle[0];
    }

    std::vector<int> act()
    {
        std::vector<int> actions;
        for(int i = 0; i < num_agents; i++)
        {
            if (previous_positions.size() == 0)
            {
                std::vector<std::pair<int, int>> positions;
                positions.push_back(env.cur_positions[i]);
                previous_positions.push_back(positions);
            }
            else
            {
                previous_positions[i].push_back(env.cur_positions[i]);
            }
            if (env.reached_goal(i))
            {
                actions.push_back(0);
            }
            else
            {
                std::list<std::pair<int, int>> visible_obstacles;
                for(int m = env.cur_positions[i].first - obs_radius; m <= env.cur_positions[i].first + obs_radius; m++) // absence of oob guaranteed by pogema
                {
                    for(int n = env.cur_positions[i].second - obs_radius; n <= env.cur_positions[i].second + obs_radius; n++)
                    {
                        if (env.grid[m][n] == OBSTACLE)
                        {
                            visible_obstacles.push_back(std::make_pair(m - env.cur_positions[i].first + obs_radius,n - env.cur_positions[i].second + obs_radius));
                        }
                    }
                }
                std::list<std::pair<int, int>> visible_agents;
                if (!ignore_other_agents)
                {
                    for (int j = 0; j < num_agents; j++)
                    {
                        if (i != j)
                        {
                            if ((std::abs(env.cur_positions[i].first - env.cur_positions[j].first) <= obs_radius) && (std::abs(env.cur_positions[i].second - env.cur_positions[j].second) <= obs_radius))
                            {
                                visible_agents.push_back(std::make_pair(env.cur_positions[j].first - env.cur_positions[i].first + obs_radius, env.cur_positions[j].second - env.cur_positions[i].second + obs_radius));
                            }
                        }
                    }
                }
                planners[i].update_obstacles(visible_obstacles, visible_agents, std::make_pair(env.cur_positions[i].first - obs_radius, env.cur_positions[i].second - obs_radius));
                // if (skip_agents.size() > 0)
                // {
                //     if(std::find(skip_agents.begin(), skip_agents.end(), i) != skip_agents.end())
                //     {
                //         actions.push_back(0);
                //         continue;
                //     }
                // }
                planners[i].update_path(env.cur_positions[i], env.goals[i]);
                auto path = planners[i].get_next_node(use_best_move);
                if (path.second.first < INF)
                {
                    const auto action = std::make_pair(path.second.first - path.first.first, path.second.second - path.first.second);
                    const auto action_index = std::find(moves.begin(), moves.end(), action) - moves.begin();
                    actions.push_back(action_index);
                }
                else
                {
                    actions.push_back(0);
                }
            }
        }
        steps++;
        if (fix_loops)
        {
            for(int i = 0; i < num_agents; i++)
            {
                auto path = previous_positions[i];
                if (path.size() > 1)
                {
                    auto cur_pos = env.cur_positions[i];
                    auto move = moves[actions[i]];
                    std::pair next_pos = std::make_pair(cur_pos.first + move.first, cur_pos.second + move.second);
                    if ((path[path.size() - 1] == next_pos) || (path[path.size() - 2] == next_pos))
                    {
                        if (cur_pos == next_pos)
                        {
                            actions[i] = _get_random_move(i, env);
                        }
                        else
                        {
                            if(seed < 0)
                                engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
                            if ((engine() % 10) < (stay_if_loop_prob * 10))
                            {
                                actions[i] = 0;
                            }
                        }
                    }
                }
            }
        }
        env.step(actions);
        return actions;
    }

    void set_env(const Environment& env_)
    {
        env = env_;
    }
};

PYBIND11_MODULE(replan, m) {
    py::class_<RePlan>(m, "RePlan")
            .def(py::init<>())
            .def("act", &RePlan::act)
            .def("init", &RePlan::init)
            .def("set_env", &RePlan::set_env)
            ;
}

/*
<%
cfg['extra_compile_args'] = ['-std=c++17']
setup_pybind11(cfg)
%>
*/