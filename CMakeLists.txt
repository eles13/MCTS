cmake_minimum_required(VERSION 3.19)
project(MCTS)

set(CMAKE_CXX_STANDARD 17)

add_executable(MCTS main.cpp mcts.cpp environment.cpp config.cpp)
target_compile_features(MCTS PRIVATE cxx_std_17)