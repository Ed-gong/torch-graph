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


torch::Tensor find_in_degree(snap_t<dst_id_t>* snaph);


//Gcn layer
struct GraphConvImpl : torch::nn::Module {
    GraphConvImpl(int64_t N, int64_t M); 
    torch::Tensor forward(torch::Tensor input, c10::intrusive_ptr<SnapWrap> snaph);
    torch::Tensor W;
    torch::Tensor b;
    int64_t M;
    int64_t N;
};
TORCH_MODULE(GraphConv);

//Gcn
struct GCN : torch::nn::Module {
    GCN(int64_t in_features, int64_t hidden_size, int64_t num_class); 
    torch::Tensor forward(torch::Tensor input, c10::intrusive_ptr<SnapWrap> snaphi);
    GraphConv conv1, conv2;
};
