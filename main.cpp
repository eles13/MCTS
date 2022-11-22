#include "mcts.h"

int main(int argc, char* argv[])
{
    std::string task("../test_task.xml");
    if(argc > 1)
        task = argv[1];
    int seed(-1);
    if(argc > 2)
        seed = std::stoi(argv[2]);
    auto mcts = MonteCarloTreeSearch(task, seed);
    while(true) {
        bool done = mcts.act();
        if(done)
            break;
    }
}
