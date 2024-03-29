#include "GCN.h"

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
                        string gather_operator) {
        //ctx->save_for_backward({input});
        ctx->saved_data["snaph"] = snaph;
        ctx->saved_data["gather_operator"] = gather_operator;

        //ctx->save_for_backward({input, snaph, gather_operator, reverse});
        
        int dim = input.size(0);
        int output_dim = input.size(1);
        array2d_t<float> input_array(input.data_ptr<float>(), dim, output_dim);
        
        torch::Tensor result = torch::zeros({dim, output_dim});
        array2d_t<float> output_array(result.data_ptr<float>(), dim, output_dim);
        
        _gspmv(snaph->snaph, input_array, output_array, gather_operator, 0);

        return result;
    }

    static tensor_list backward(AutogradContext *ctx, auto grad_outputs) {
        //auto saved = ctx->get_saved_variables();
        //auto input = saved[0];
        auto snaph = ctx->saved_data["snaph"].toCustomClass<SnapWrap>();
        string gather_operator = ctx->saved_data["gather_operator"].toStringRef();
        
        ctx->saved_data.clear();//required for memory leak fix 
        
        //auto grad_output = grad_outputs[0];
        int64_t reverse = 1;
        //std::cout<<"grad_output"<<std::endl;
        //std::cout<<grad_outputs[0]<<std::endl;
        //auto grad_input = grad_output.sum(0);
        auto grad_graph_snaph = torch::Tensor();
        auto grad_gather_operator = torch::Tensor();
       
        torch::Tensor input = grad_outputs[0];
        int dim = input.size(0);
        int output_dim = input.size(1);
        array2d_t<float> input_array(input.data_ptr<float>(), dim, output_dim);
        
        torch::Tensor result = torch::zeros({dim, output_dim});
        array2d_t<float> output_array(result.data_ptr<float>(), dim, output_dim);
        
        
        _gspmv(snaph->snaph, input_array, output_array, gather_operator, reverse);
        //cout << "grad output1" << endl;
        //cout << grad_input << endl;

        return {result, grad_graph_snaph, grad_gather_operator};
    }
};


//Gcn layer
GraphConvImpl::GraphConvImpl(int64_t N, int64_t M) {
    W = register_parameter("W", torch::randn({N, M}));
    //std::cout << W << std::endl;
    b = register_parameter("b", torch::zeros(M));
}

//torch::Tensor forward(torch::Tensor input, ManagerWrap manager) 
torch::Tensor GraphConvImpl::forward(torch::Tensor input, c10::intrusive_ptr<SnapWrap> snaph) 
{    
    torch::Tensor dst_data;

    /*
    if (N > M) {
    //mult W first to reduce the feature size for aggregation
        //input = torch::matmul(input, W);
        dst_data = torch::matmul(input, W);
        //dst_data = _gspmv(snaph, input, "sum");
    } else {
        //aggregate first then mult W
        //dst_data = _gspmv(snaph, input, "sum");
        dst_data = torch::matmul(dst_data, W);
    }*/
    
    //cout << "input to layer matmul" << endl;
    //cout << input << endl;


    //Noramlization has moved to spmm
    /*torch::Tensor degree_list = find_out_degree(snaph->snaph);
    int num_dim = degree_list.size(0);
    degree_list = degree_list.reshape({num_dim, 1});
    torch::Tensor norm = torch::pow(degree_list, -0.5);
    input = input * norm;*/


    torch::Tensor input1 = torch::matmul(input, W);
    //cout << "output of layer matmul" << endl;
    //cout << input1 << endl;
    dst_data = Scatter_gather::apply(input1, snaph, "sum");
    //dst_data = dst_data * norm; // we apply 'both'as the norm, so we need to apply it twice

    dst_data += b;
    return dst_data;
}

//Gcn
GCN::GCN(int64_t in_features, int64_t hidden_size, int64_t num_class) 
        : conv1(in_features, hidden_size), conv2(hidden_size, num_class) 
{   
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    std::cout<< "initilize the GCN" << std::endl;
}

torch::Tensor GCN::forward(torch::Tensor input, //snap_t<dst_id_t>* snaph
                           c10::intrusive_ptr<SnapWrap> snaph) 
{
    torch:: Tensor h = conv1->forward(input, snaph);
    //std::cout<<"output of first layer"<<std::endl;
    //std::cout<<h<<std::endl;
    h = torch::relu(h);
    //std::cout<<"---> output of relu"<<std::endl;
    //std::cout<<h<<std::endl;

    h = conv2->forward(h, snaph);
    //std::cout<<"---> output of second layer"<<std::endl;
    //std::cout<<h<<std::endl;
    return h;
}
