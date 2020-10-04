// Created by yidong on 9/24/20.
//
#pragma once

#include <iostream>
#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <iostream>
// This header is what defines the custom class registration
// behavior specifically. script.h already includes this, but
// we include it here so you know it exists in case you want
// to look at the API or implementation.
//#include <torch/custom_class.h>
#include <string>
#include <vector>
//#include <cstdint>
using namespace std;
using torch::Tensor;
using torch::autograd::Node;
using torch::autograd::deleteNode;
using torch::autograd::SavedVariable;
using torch::autograd::variable_list;
using torch::autograd::tensor_list;
#include "SnapWrap.h"
#include "ManagerWrap.h"

torch::Tensor gat_result1(const torch::Tensor & input_feature, snap_t<dst_id_t>* snaph, int64_t reverse);

struct GATlayer : torch::nn::Module {
        GATlayer(int64_t in_dim, int64_t out_dim);
        torch::Tensor forward(torch::Tensor input, c10::intrusive_ptr<SnapWrap> snaph);
        torch::nn::Linear linear1;
        torch::nn::Linear linear2;
        //torch::nn::Linear linear;
        int64_t in_dim;
        int64_t out_dim;
};


struct GAT : torch::nn::Module {
    GAT(int64_t in_dim, int64_t hidden_dim, int64_t out_dim);
    torch::Tensor forward(torch::Tensor input, c10::intrusive_ptr<SnapWrap> snaph);
    vector<torch::Tensor> parameters();
    GATlayer gatlayer1, gatlayer2;
};



