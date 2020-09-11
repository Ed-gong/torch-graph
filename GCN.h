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


//Gcn layer
struct GraphConv : torch::nn::Module {
    GraphConv(int64_t N, int64_t M); 
    torch::Tensor forward(torch::Tensor input, c10::intrusive_ptr<SnapWrap> snaph, torch::Tensor weight);
    torch::Tensor W;
    torch::Tensor b;
    int64_t M;
    int64_t N;
};

//Gcn
struct GCN : torch::nn::Module {
    GCN(int64_t in_features, int64_t hidden_size, int64_t num_class); 
    torch::Tensor forward(torch::Tensor input, c10::intrusive_ptr<SnapWrap> snaphi, torch::Tensor weight2, torch::Tensor weight3);
    vector<torch::Tensor> parameters();
    //snap_t<dst_id_t>* get_current_graph(plaingraph_manager_t<dst_id_t>* manager, snap_t<dst_id_t>* snaph);
    //c10::intrusive_ptr<SnapWrap>  get_current_graph(plaingraph_manager_t<dst_id_t>* manager, snap_t<dst_id_t>* snaph);
    GraphConv conv1, conv2;
};
