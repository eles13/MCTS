// cppimport
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
namespace py = pybind11;
struct Config
{
    double gamma = 0.99;
    int num_actions = 5;
    int num_expansions = 1000;
    double uct_c = 1.0;
    int steps_limit = 64;
    int multi_simulations = 1;
    bool use_move_limits = true;
    bool agents_as_obstacles = false;
    int batch_size = 1;
    int num_parallel_trees = 1;
    bool render = false;
};

PYBIND11_MODULE(config, m) {
    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("gamma", &Config::gamma)
        .def_readwrite("num_actions", &Config::num_actions)
        .def_readwrite("num_expansions", &Config::num_expansions)
        .def_readwrite("uct_c", &Config::uct_c)
        .def_readwrite("steps_limit", &Config::steps_limit)
        .def_readwrite("multi_simulations", &Config::multi_simulations)
        .def_readwrite("use_move_limits", &Config::use_move_limits)
        .def_readwrite("agents_as_obstacles", &Config::agents_as_obstacles)
        .def_readwrite("batch_size", &Config::batch_size)
        .def_readwrite("num_parallel_trees", &Config::num_parallel_trees)
        ;
}

/*
<%
cfg['extra_compile_args'] = ['-std=c++17']
setup_pybind11(cfg)
%>
*/