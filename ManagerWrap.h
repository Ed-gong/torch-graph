//
// Created by yidong on 8/18/20.
//

#ifndef UNTITLED1_MANAGERWRAP_H
#define UNTITLED1_MANAGERWRAP_H

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
#include "SnapWrap.h"

using namespace std;
using torch::Tensor;
using torch::autograd::Node;
using torch::autograd::deleteNode;
using torch::autograd::SavedVariable;
using torch::autograd::variable_list;
using torch::autograd::tensor_list;

torch::Tensor scatter_gather1(snap_t<dst_id_t>* snaph, 
                             const torch::Tensor & input_feature, 
                             string gather_operator, int64_t reverse);




snap_t<dst_id_t>* check_current_graph(plaingraph_manager_t<dst_id_t>* manager, snap_t<dst_id_t>* snaph);



struct ManagerWrap : torch::CustomClassHolder {
    plaingraph_manager_t<dst_id_t>* manager;

    ManagerWrap(int64_t flags, int64_t node_number, string path);
    torch::Tensor scatter_gather(const torch::Tensor & input_feature, string gather_operator, 
                            c10::intrusive_ptr<SnapWrap> snaph, int64_t reverse);     
    void create_static_view(c10::intrusive_ptr<SnapWrap> snaph);
};

#endif //UNTITLED1_MANAGERWRAP_H

