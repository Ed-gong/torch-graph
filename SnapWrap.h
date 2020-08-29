//
// Created by yidong on 8/18/20.
//

# pragma once
// This header is all you need to do the C++ portions of this
// tutorial
#include <torch/script.h>
#include <iostream>
// This header is what defines the custom class registration
// behavior specifically. script.h already includes this, but
// we include it here so you know it exists in case you want
// to look at the API or implementation.
#include <torch/custom_class.h>

#include <string>
#include <vector>
//#include <cstdint>

#include "type.h"
#include "graph.h"
#include "plain_to_edge.h"
#include "graph_view.h"
#include "static_view.h"



using namespace std;
using torch::Tensor;
using torch::autograd::Node;
using torch::autograd::deleteNode;
using torch::autograd::SavedVariable;
using torch::autograd::variable_list;
using torch::autograd::tensor_list;


struct SnapWrap : torch::CustomClassHolder {
    snap_t<dst_id_t>* snaph;

    SnapWrap();
    //torch::Tensor scatter_gather(const torch::Tensor & input_feature, string gather_operator);
};

