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

#include "GCN.h"
#include "ManagerWrap.h"

#include <torch/torch.h>
using namespace torch::autograd;

// Inherit from Function
class Scatter_gather : public Function<Scatter_gather> {
public:
    // Note that both forward and backward are static functions

    // bias is an optional argument
    static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input, 
                        c10::intrusive_ptr<SnapWrap> snaph,
                        //snap_t<dst_id_t>* snaph, 
                        string gather_operator, int64_t reverse) {
        ctx->save_for_backward({input});
        ctx->saved_data["snaph"] = snaph;
        ctx->saved_data["gather_operator"] = gather_operator;

        //ctx->save_for_backward({input, snaph, gather_operator, reverse});
        auto output = scatter_gather1(snaph->snaph, input, gather_operator, reverse);

        return output;
    }

    static tensor_list backward(AutogradContext *ctx, auto grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto snaph = ctx->saved_data["snaph"].toCustomClass<SnapWrap>();
        string gather_operator = ctx->saved_data["gather_operator"].toStringRef();
        auto grad_output = grad_outputs[0];
        int64_t reverse = 1;
        //auto grad_input = grad_output.sum(0);
        auto grad_graph_snaph = torch::Tensor();
        auto grad_gather_operator = torch::Tensor();
        auto grad_reverse = torch::Tensor();
        auto grad_input = scatter_gather1(snaph->snaph, grad_output, gather_operator, reverse);
        //auto grad_input = scatter_gather_reverse(input, snaph, gather_operator,  grad_reverse);

        return {grad_input, grad_graph_snaph, grad_gather_operator, grad_reverse};
    }
};


//Gcn layer
GraphConv::GraphConv(int64_t N, int64_t M) {
    W = register_parameter("W", torch::randn({N, M}));
    std::cout << W << std::endl;
    //b = register_parameter("b", torch::randn(M));
}

//torch::Tensor forward(torch::Tensor input, ManagerWrap manager) 
torch::Tensor GraphConv::forward(torch::Tensor input, c10::intrusive_ptr<SnapWrap> snaph) 
{    
    torch::Tensor dst_data;

    /*
    if (N > M) {
    //mult W first to reduce the feature size for aggregation

        //input = torch::matmul(input, W);
        dst_data = torch::matmul(input, W);
        //dst_data = scatter_gather1(snaph, input, "sum");
    }
    else {
        //aggregate first then mult W
        //dst_data = scatter_gather1(snaph, input, "sum");
        dst_data = torch::matmul(dst_data, W);

    }*/

     input = torch::matmul(input, W);
     dst_data = Scatter_gather::apply(input, snaph, "sum", 0);// "0" represent there is no reverse here.
     //dst_data = Scatter_gather::apply(input, snaph, "sum", W);

    return dst_data;
}

//Gcn
GCN::GCN(int64_t in_features, int64_t hidden_size, int64_t num_class) 
        : conv1(in_features, hidden_size), conv2(hidden_size, num_class) 
{   
    //register_module("conv1", conv1);
    //register_module("conv2", conv2);
    std::cout<< "initilize the GCN" << std::endl;
}

torch::Tensor GCN::forward(torch::Tensor input, 
                           //snap_t<dst_id_t>* snaph
                           c10::intrusive_ptr<SnapWrap> snaph
                           ) 
{
    torch:: Tensor h = conv1.forward(input, snaph);
    h = torch::relu(h);
    h = conv2.forward(h, snaph);
    return h;
}

vector<torch::Tensor> GCN::parameters(){
    std::vector<torch::Tensor> result;
    torch::Tensor para1 = conv1.W;
    torch::Tensor para2 = conv2.W;
    result.push_back(para1);
    result.push_back (para2);
    return result;
}

/*
tensor_list apply_scatter_gather(torch::Tensor input, snap_t<dst_id_t>* snaph, string gather_operator, torch::Tensor W){
     output = Scatter_gather::apply(input, snaph, "sum", W);
     return output;
}
*/

/*
auto scatter_gather_reverse(torch::Tensor input, snap_t<dst_id_t>* snaph, string gather_operator, int64_t reverse){
    torch::Tensor t2;
    return t2;
    //auto result = Scatter_gather::apply(input, snaph,gather_operator, reverse);
}*/

