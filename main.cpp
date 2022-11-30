#include "mcts.cpp"

int main(int argc, char* argv[])
{
    auto mcts = MonteCarloTreeSearch();
    auto env = Environment();
    env.add_agent(0, 0, 1, 0);
    env.create_grid(2, 2);
    env.add_obstacle(0, 1);
    env.render();
    auto config = Config();
    mcts.set_config(config);
    mcts.set_env(env);
    while(true) {
        std::cout<<"in\n";
        mcts.act();
        std::cout<<"step\n";
        if(mcts.env.all_done())
            break;
    }
}