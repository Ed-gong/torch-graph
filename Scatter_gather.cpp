//
// Created by yidong on 8/31/20.
//

#include "Scatter_gather.h"
using namespace torch::autograd;

// Inherit from Function
class Scatter_gather :: Scatter_gather {
public:
    // Note that both forward and backward are static functions

    // bias is an optional argument
    static torch::Tensor forward(
            AutogradContext *ctx, torch::Tensor input, snap_t<dst_id_t>* snaph, string gather_operator) {
        ctx->save_for_backward({input});
        auto output = scatter_gather1(snap_t<dst_id_t>* snaph, const torch::Tensor & input, string gather_operator);

        return output;
    }

    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto grad_output = grad_outputs[0];
        auto grad_input = grad_output.sum(0);
        auto grad_graph_snaph = torch::Tensor();
        auto grad_gather_operator = torch::Tensor();

        return {grad_input, grad_graph_snaph, grad_gather_operator};
    }
};
