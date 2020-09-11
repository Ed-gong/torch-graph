//
// Created by yidong on 8/31/20.
//

#ifndef UNTITLED1_SCATTER_GATHER_H
#define UNTITLED1_SCATTER_GATHER_H

#include <torch/torch.h>
# include "ManagerWrap.h"
using namespace torch::autograd;

class Scatter_gather: public Function<Scatter_gather> {

public:
    static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input, snap_t<dst_id_t>* snaph, string gather_operator);
    static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs);

};


#endif //UNTITLED1_SCATTER_GATHER_H

