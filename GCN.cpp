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



torch::Tensor find_out_degree(snap_t<dst_id_t>* snaph){
    degree_t nebr_count = 0;
    nebr_reader_t<dst_id_t> header;
    vid_t v_count = snaph->get_vcount();
    torch::Tensor degree_list = torch::ones({v_count,1});


    for (vid_t v = 0; v < v_count; v++) {
        nebr_count = snaph -> get_nebrs_out(v, header);
        degree_list[v] = degree_list[v] * nebr_count;
    }

    return degree_list;
}


// Inherit from Function
class Scatter_gather : public Function<Scatter_gather> {
public:
    // Note that both forward and backward are static functions

    // bias is an optional argument
    static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input, 
                        c10::intrusive_ptr<SnapWrap> snaph,
                        //snap_t<dst_id_t>* snaph, 
                        string gather_operator) {
        ctx->save_for_backward({input});
        ctx->saved_data["snaph"] = snaph;
        ctx->saved_data["gather_operator"] = gather_operator;

        //ctx->save_for_backward({input, snaph, gather_operator, reverse});
        auto output = scatter_gather1(snaph->snaph, input, gather_operator, 0);

        return output;
    }

    static tensor_list backward(AutogradContext *ctx, auto grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto snaph = ctx->saved_data["snaph"].toCustomClass<SnapWrap>();
        string gather_operator = ctx->saved_data["gather_operator"].toStringRef();
        //auto grad_output = grad_outputs[0];
        int64_t reverse = 1;
        std::cout<<"grad_output"<<std::endl;
        std::cout<<grad_outputs[0]<<std::endl;
        //auto grad_input = grad_output.sum(0);
        auto grad_graph_snaph = torch::Tensor();
        auto grad_gather_operator = torch::Tensor();
        auto grad_input = scatter_gather1(snaph->snaph, grad_outputs[0], gather_operator, reverse);
        cout << "grad output1" << endl;
        cout << grad_input << endl;

        return {grad_input, grad_graph_snaph, grad_gather_operator};
    }
};


//Gcn layer
GraphConv::GraphConv(int64_t N, int64_t M) {
    W = register_parameter("W", torch::randn({N, M}));
    //std::cout << W << std::endl;
    b = register_parameter("b", torch::zeros(M));
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
    //cout << "input to layer matmul" << endl;
    //cout << input << endl;
    torch::Tensor degree_list = find_out_degree(snaph->snaph);
    int num_dim = degree_list.size(0);
    degree_list = degree_list.reshape({num_dim, 1});
    torch::Tensor norm = torch::pow(degree_list, -0.5);
    input = input * norm;

    torch::Tensor input1 = torch::matmul(input, W);
    //cout << "output of layer matmul" << endl;
    //cout << input1 << endl;
    dst_data = Scatter_gather::apply(input1, snaph, "sum");// "0" represent there is no reverse here.
    dst_data = dst_data * norm; // we apply 'both'as the norm, so we need to apply it twice

    dst_data += b;
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
    std::cout<<"output of first layer"<<std::endl;
    std::cout<<h<<std::endl;
    h = torch::relu(h);
    std::cout<<"---> output of relu"<<std::endl;
    std::cout<<h<<std::endl;

    h = conv2.forward(h, snaph);
    std::cout<<"---> output of second layer"<<std::endl;
    std::cout<<h<<std::endl;
    return h;
}


vector<torch::Tensor> GCN::parameters(){
    std::vector<torch::Tensor> result;
    torch::Tensor para1 = conv1.W;
    torch::Tensor para2 = conv1.b;

    torch::Tensor para3 = conv2.W;
    torch::Tensor para4 = conv2.b;

    result.push_back(para1);
    result.push_back (para2);
    result.push_back (para3);
    result.push_back (para4);

    return result;
}
