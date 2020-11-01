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

torch::Tensor gat_result1(const torch::Tensor & input_feature, snap_t<dst_id_t>* snaph, string gather_operator, int64_t reverse);
//torch::Tensor edge_softmax_1(snap_t<dst_id_t>* snaph,  const torch::Tensor & efficient_score);
torch::Tensor add_by_edge(snap_t<dst_id_t>* snaph, const torch::Tensor & input_left, const torch::Tensor & input_right);
torch::Tensor gat_update_by_edge1(const torch::Tensor & input_left, snap_t<dst_id_t>* snaph, const torch::Tensor & input_right, string oper, int64_t reverse);
torch::Tensor gat_update_all_vertix1(const torch::Tensor & input_feature, snap_t<dst_id_t>* snaph, const torch::Tensor &  edge_score_by_softmax, string gather_operator,  int64_t reverse);

struct GATlayerImpl : torch::nn::Module {
    GATlayerImpl(int64_t in_dim, int64_t out_dim);
    torch::Tensor forward(torch::Tensor input, c10::intrusive_ptr<SnapWrapW> snaph);
    torch::nn::Linear linear1;
    torch::Tensor W_left;
    torch::Tensor W_right;
    //torch::nn::Linear linear;
    int64_t in_dim;
    int64_t out_dim;
};
TORCH_MODULE(GATlayer);

struct GAT : torch::nn::Module {
    GAT(int64_t in_dim, int64_t hidden_dim, int64_t out_dim);
    torch::Tensor forward(torch::Tensor input, c10::intrusive_ptr<SnapWrapW> snaph);
    //vector<torch::Tensor> parameters();
    GATlayer gatlayer1, gatlayer2;
};

struct GATmultiheadImpl : torch::nn::Module {
    GATmultiheadImpl(int64_t in_dim, int64_t out_dim, int64_t num_heads);
    torch::Tensor forward(torch::Tensor input, c10::intrusive_ptr<SnapWrap> snaph, string merge = "cat");
    torch::nn::ModuleList mlist;
};
TORCH_MODULE(GATmultihead);

struct GAT1 : torch::nn::Module {
    GAT1(int64_t in_dim, int64_t hidden_dim, int64_t out_dim, int64_t num_heads);
    torch::Tensor forward(torch::Tensor input, c10::intrusive_ptr<SnapWrap> snaph);
    GATmultihead gatmultihead1, gatmultihead2;
};
