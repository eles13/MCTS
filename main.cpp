#include <iostream>
#include "mcts.h"
int main() {
    std::string task_name = "test_task.xml";
    MonteCarloTreeSearch mcts = MonteCarloTreeSearch(task_name);
    while(true)
    {
        bool done = mcts.act();
        if(done)
            break;
    }
    return 0;
}
