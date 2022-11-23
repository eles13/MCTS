#include <list>
#include <vector>
#include "environment.h"

struct Node
{
    int action_id;
    Node* parent;
    int cnt;
    double w;
    double q;
    std::vector<Node*> child_nodes;
    int agent_id;
    int cnt_sne;
    std::vector<bool> mask_picked;
    int num_actions_;

    Node(Node* _parent, int _action_id, double _w, int num_actions, int _agent_id=-1)
            :parent(_parent), action_id(_action_id), w(_w), agent_id(_agent_id)
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

    int get_action(const Environment& env)
    {
        int best_action(0), best_score(-1), k(0);
        for(auto c:child_nodes) {
            if (c != nullptr && c->cnt > best_score)
            {
                best_action = k;
                best_score = c->cnt;
            }
            k++;
        }
        return best_action;
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

    Node(const Node& orig)
    {
        action_id = orig.action_id;
        parent = orig.parent;
        cnt = orig.cnt;
        w = orig.w;
        q = orig.q;
        child_nodes = orig.child_nodes;
        agent_id = orig.agent_id;
        cnt_sne = orig.cnt_sne;
        mask_picked = orig.mask_picked;
        num_actions_ = orig.num_actions_;
    }
};
