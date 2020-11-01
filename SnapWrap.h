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

template <class T>
struct SnapWrap_t : torch::CustomClassHolder {
    snap_t<T>* snaph;

    SnapWrap_t() {
        snaph = 0;
    };
    //torch::Tensor scatter_gather(const torch::Tensor & input_feature, string gather_operator);
};

typedef SnapWrap_t<dst_id_t> SnapWrap;
typedef SnapWrap_t<weight_sid_t> SnapWrapW;
