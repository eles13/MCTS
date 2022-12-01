cmake_minimum_required(VERSION 3.23)
project(MCTS)

set(CMAKE_CXX_STANDARD 14)

add_executable(MCTS main.cpp)
target_compile_features(MCTS PRIVATE cxx_std_17)