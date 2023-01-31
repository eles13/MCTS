#include <list>
#include <vector>

class Node
{
public:
    int action_id;
    Node* parent;
    uint64_t cnt;
    double w;
    double q;
    std::vector<Node*> child_nodes;
    int agent_id;
    uint64_t cnt_sne;
    std::vector<bool> mask_picked;
    int num_actions_;
    size_t num_succeeded;

    Node(Node* _parent, int _action_id, double _w, int num_actions, int _agent_id=-1)
            : action_id(_action_id), parent(_parent), w(_w), agent_id(_agent_id), num_succeeded(0)
    {
        cnt = 1;
        q = w;
        num_actions_ = num_actions;
        child_nodes.resize(num_actions, nullptr);
        zero_snes();
    }

    void update_value(double value)
    {
        w += value;
        cnt++;
        q = w/cnt;
    }

    void update_value_batch(double value)
    {
        w += value;
        cnt++;
        q = w/cnt;
        if (parent != nullptr)
        {
            parent->update_value_batch(value);
        }
    }

    int get_action()
    {
        int best_action(0), k(0);
        uint64_t best_score = 0;
        for(auto c: child_nodes)
        {
            if (c != nullptr)
            {
                if (c->cnt > best_score)
                {
                    best_action = k;
                    best_score = c->cnt;
                }
            }
            k++;
        }
        while((child_nodes[best_action] == nullptr) && (best_action < num_actions_))
        {
            best_action++;
        }
        if(best_action >= num_actions_)
        {
            return -1;
        }
        else
        {
            return best_action;
        }
    }

    void zero_snes()
    {
        cnt_sne = 0;
        mask_picked.clear();
        mask_picked.resize(num_actions_, false);
        for (auto child : child_nodes)
        {
            if (child)
            {
                child->zero_snes();
            }
        }
    }

    void update_q()
    {
        q = w/cnt;
        for(auto child: child_nodes)
        {
            if(child)
            {
                child->update_q();
            }
        }
    }
};
