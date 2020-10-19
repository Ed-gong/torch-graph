//
// Created by yidong on 9/24/20.
//
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

#include "ManagerWrap.h"
#include <torch/torch.h>
using namespace torch::autograd;
#include "GAT.h"

torch::Tensor apply_edges(snap_t<dst_id_t>* snaph, const torch::Tensor & input_left, const torch::Tensor & input_right)
{
    degree_t nebr_count = 0;
    vid_t sid;
    nebr_reader_t<dst_id_t> header;

    vid_t v_count = snaph->get_vcount();

    //return the value of each node after gather procedure
    //int output_dim = input_feature.size(1);

    int edge_count = 0;
    if (snaph->is_udir()) {
        edge_count = _edge_count << 1;
    } else {
        edge_count = _edge_count;
    }

    torch::Tensor result = torch::zeros({edge_count,1});
    int count = 0;


    for (vid_t v = 0; v < v_count; v++) {
        nebr_count = snaph->get_nebrs_out(v, header);
        
        for (degree_t i = 0; i < nebr_count; ++i) {
            sid = TO_SID(get_sid(header[i]));
            //std::cout << "the sid" <<std::endl;
            torch::Tensor input_feature_left = input_left[v];
            torch::Tensor input_feature_right = input_right[sid];
            torch::Tensor temp = torch::add(input_feature_left , input_feature_right);
            //temp = temp.reshape({1, 1});
            result[count] = temp;
            count = count + 1;
        }
    }

    assert(count == edge_count);
    return result;
}


// calcuate the feature by edge, left is vertex, right is edge
torch::Tensor gsddmm(const torch::Tensor & input_left, snap_t<dst_id_t>* snaph, const torch::Tensor & input_right, string oper, int64_t reverse)
{
    degree_t nebr_count = 0;
    vid_t sid;
    nebr_reader_t<dst_id_t> header;
    vid_t v_count = snaph->get_vcount();
    
    //torch::Tensor out = torch::zeros({input_right.size(0), input_right.size(1)});
    int64_t count = 0;

    std::map<int, torch::Tensor> mailbox;
    for (vid_t v = 0; v < v_count; v++) {
        if (reverse == 1){
            nebr_count = snaph->get_nebrs_in(v, header);
        } else {
            nebr_count = snaph->get_nebrs_out(v, header);
        }

        if (nebr_count == 0) {
            continue;
        }
        // the node j scatter it's message to all neighors
        for (degree_t i = 0; i < nebr_count; ++i) {
            sid = TO_SID(get_sid(header[i]));
            torch::Tensor result_score = torch::zeros({1,1});

            if (oper == "div") {
                result_score = input_right[count] / input_left[sid];
            } else if (oper == "sub") {
                result_score = input_right[count] - input_left[sid];
            } else if (oper == "add") {
                result_score = input_right[count] + input_left[sid];
            } else if (oper == "mul"){
                result_score = input_right[count] * input_left[sid];
            } else {
                assert(0);
                result_score = input_right[count];
            }

            input_right[count] = result_score;// update the efficient score
            ++count;
        }
    }

    return input_right;
}

torch::Tensor spmmw(const torch::Tensor & input_feature, snap_t<dst_id_t>* snaph, const torch::Tensor &  edge_score_by_softmax, string gather_operator,  int64_t reverse) 
{
    degree_t nebr_count = 0;
    vid_t sid;
    nebr_reader_t<dst_id_t> header;
    int output_dim = 1; //input_feature.size(1);
    if (input_feature.defined() != false) {
        output_dim = input_feature.size(1);
    }

    //build the mailbox
    std::map<int, torch::Tensor> mailbox;
    vid_t v_count = snaph->get_vcount();
    //std::cout << "-> begin the scatter!!" << std::endl;
    int count = 0;
    for (vid_t v = 0; v < v_count; v++) {
        if (reverse == 1){
            nebr_count = snaph->get_nebrs_in(v, header);
        } else {
            nebr_count = snaph->get_nebrs_out(v, header);
        }
        // if one node do not have any neighbor, we do not scatter it's message
        if (nebr_count == 0) { continue; }

        // the node j scatter it's message to all neighors
        if (input_feature.defined() == false) {
            for (degree_t i = 0; i < nebr_count; ++i) {
                sid = TO_SID(get_sid(header[i]));
                torch::Tensor message = edge_score_by_softmax[count];
                message = message.reshape({1, output_dim});
                //If mailbox is empty, we initilize the mailbox
                if (mailbox.count(sid) == 0) {
                    //std::cout<<"nani?"<<std::endl;
                    //std::cout<<message<<std::endl;
                    mailbox[sid] = message;
                    count = count + 1;
                    continue;
                }

                // If the mailbox is not emptys, we concatenate the received message with the new message
                torch::Tensor temp = torch::cat({mailbox[sid], message}, 0);
                count = count + 1;
                //std::cout<<"kkk"<<std::endl;
                //std::cout << temp<< std::endl;
                mailbox[sid] = temp;
            }
        } else { 
            for (degree_t i = 0; i < nebr_count; ++i) {
                sid = TO_SID(get_sid(header[i]));
                //std::cout << "the sid" <<std::endl;
                torch::Tensor message = input_feature[v];
                torch::Tensor scope = edge_score_by_softmax[count];
                message = message * scope;
                message = message.reshape({1, output_dim});
                //If mailbox is empty, we initilize the mailbox
                if (mailbox.count(sid) == 0){
                    //std::cout<<"nani?"<<std::endl;
                    //std::cout<<message<<std::endl;
                    mailbox[sid] = message;
                    count = count + 1;
                    continue;

                }
                // If the mailbox is not emptys, we concatenate the received message with the new message
                torch::Tensor temp = torch::cat({mailbox[sid], message}, 0);
                count = count + 1;
                //std::cout<<"kkk"<<std::endl;
                //std::cout << temp<< std::endl;
                mailbox[sid] = temp;
            }
        }
    }

    //gather procedure, each node gather it's received message by the method defined by 'gather_operator'
    //std::cout << "-> gather procedure begins" << (int) reverse << std::endl;

    torch::Tensor temp;
    for (std::map<int,torch::Tensor>::iterator it = mailbox.begin(); it!= mailbox.end(); ++it){

        //std::cout<<"message:"<<std::endl;
        //std::cout<<it -> second <<std::endl;

        if (gather_operator == "max"){
            temp = torch::max(it -> second);
        }
        if (gather_operator == "min"){
            temp = torch::min(it -> second);
        }
        if (gather_operator == "sum"){
            //std::cout<<"lll"<<std::endl;
            //std::cout<<it ->second<< std::endl;
            temp = torch::sum(it -> second, 0);
        }

        it->second = temp;
    }

    //return the value of each node after gather procedure
    //int output_dim = input_feature.size(1);
    torch::Tensor result = torch::zeros({v_count,output_dim});//{v_count, 1} means the tensor is 1 dimension,otherwise, we cannot concatenated tensors
    for (vid_t v = 0; v < v_count; v++) {
        //loop the mailbox by key:node_id, value:tensor,
        //if the node did not reveive any message, the value of that node is 0
        map<int,torch::Tensor>::iterator it = mailbox.find(v);
        if (it == mailbox.end()) continue;
        result[v] = mailbox[v];
    }

    return result;
}

class GAT_update_by_edge : public Function<GAT_update_by_edge> {
public:
    // Note that both forward and backward are static functions
    // bias is an optional argument
    static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input_left,
                                 c10::intrusive_ptr<SnapWrap> snaph, torch::Tensor input_right) {
        ctx->save_for_backward({input_left, input_right});
        ctx->saved_data["snaph"] = snaph;
        //ctx->save_for_backward({input, snaph, gather_operator, reverse});
        auto output = gsddmm(input_left, snaph -> snaph, input_right, "mul", 0);
        return output;
    }

    static tensor_list backward(AutogradContext *ctx, auto grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input_left = saved[0];
        auto input_right = saved[1];
        auto snaph = ctx->saved_data["snaph"].toCustomClass<SnapWrap>();
        //auto grad_output = grad_outputs[0];
        int64_t reverse = 1;
        auto grad_graph_snaph = torch::Tensor();
        auto grad_input_left = spmmw(grad_outputs[0] * input_left, snaph->snaph, {}, "sum" ,reverse);// {} represnts the None
        auto grad_input_right = gsddmm(input_left, snaph -> snaph, input_right, "mul", 0);//not sure
        //cout << "grad output1" << endl;

        return {grad_input_left, grad_graph_snaph, grad_input_right };
    }
};


class SPMMW : public Function<SPMMW> {
public:
    // Note that both forward and backward are static functions

    // bias is an optional argument
    static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input,
                                 c10::intrusive_ptr<SnapWrap> snaph, torch::Tensor edge_score_by_softmax) {
        ctx->save_for_backward({input, edge_score_by_softmax});
        ctx->saved_data["snaph"] = snaph;
        //ctx->save_for_backward({input, snaph, gather_operator, reverse});
        auto output = spmmw(input, snaph -> snaph, edge_score_by_softmax,"sum", 0);
        return output;
    }

    static tensor_list backward(AutogradContext *ctx, auto grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto edge_score_by_softmax = saved[1];
        auto snaph = ctx->saved_data["snaph"].toCustomClass<SnapWrap>();
        //auto grad_output = grad_outputs[0];
        int64_t reverse = 1;
        auto grad_graph_snaph = torch::Tensor();
        auto grad_edge_score_by_softmax1 = torch::Tensor();  //grad_outputs[0] * edge_score_by_softmax - edge_score_by_softmax;// not sure
        auto grad_input = spmmw(grad_outputs[0], snaph->snaph, edge_score_by_softmax, "sum" ,reverse);
        //cout << "grad output1" << endl;

        return {grad_input, grad_graph_snaph, grad_edge_score_by_softmax1};
    }
};


class EdgeSoftmax : public Function<EdgeSoftmax> {
public:
    // Note that both forward and backward are static functions

    // bias is an optional argument
    static torch::Tensor forward(AutogradContext *ctx, c10::intrusive_ptr<SnapWrap> snaph,  const torch::Tensor & efficient_score) {
        ctx->saved_data["snaph"] = snaph;
        //score_max will be |V| of dst vertices, efficient_score is |E|
        auto score_max = spmmw({}, snaph->snaph, efficient_score, "max" , 0);

        //Score is |E|
        auto score = gsddmm(score_max, snaph->snaph, efficient_score, "sub", 0);
        score = torch::exp(score);
        
        //Sum edge score to dst. Score_sum will be |V|
        auto score_sum = spmmw({}, snaph->snaph, score, "sum" , 0);

        //score%score_sum. out is |E|
        auto out = gsddmm(score_sum, snaph -> snaph, score, "div", 0);
        ctx->save_for_backward({out});
        return out;
    }

    static tensor_list backward(AutogradContext *ctx, auto grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto out = saved[0];
        auto sds = out * grad_outputs[0];
        auto snaph = ctx->saved_data["snaph"].toCustomClass<SnapWrap>();
        
        auto accum = spmmw({}, snaph->snaph, sds, "sum", 0);
        auto grad_score = sds - gsddmm(accum, snaph->snaph, out, "mul", 0);
        
        auto grad_graph_snaph = torch::Tensor();
        return {grad_graph_snaph, grad_score};
    }
};


GATlayerImpl::GATlayerImpl(int64_t in_dim, int64_t out_dim)
        : linear1(register_module("linear1", torch::nn::Linear(in_dim, out_dim))) {
    W_left = register_parameter("W_left", torch::randn({out_dim, 1}));
    W_right = register_parameter("W_right", torch::randn({out_dim, 1}));
}

torch::Tensor GATlayerImpl::forward(torch::Tensor input, c10::intrusive_ptr<SnapWrap> snaph){

    torch::Tensor map_input = linear1(input); // equation1
    //std::cout<<"nani?"<< std::endl;
    torch::Tensor input_left = torch::matmul(map_input, W_left);
    //std::cout<<"nani2"<<std::endl;
    torch::Tensor input_right = torch::matmul(map_input, W_right);
    //std::cout<<"nani3"<<std::endl;
    torch::Tensor edge_score = apply_edges(snaph -> snaph, input_left, input_right);//equation 2 by edge
    //std::cout<<"nani4"<<std::endl;
    torch::Tensor efficient_score = torch::leaky_relu(edge_score, 0.2); // double check
    //std::cout<<"nani5"<<std::endl;
    torch::Tensor edge_score_by_softmax = EdgeSoftmax::apply(snaph, efficient_score);//get final significiant for each edge
    //std::cout<<"nani6"<<std::endl;
    torch::Tensor h = SPMMW::apply(map_input, snaph, edge_score_by_softmax);
    //std::cout<<"nani7"<<std::endl;

//    torch::Tensor h = GAT_result::apply(temp, snaph, this);
    return h;

}


GAT::GAT(int64_t in_dim, int64_t hidden_dim, int64_t out_dim)
        : gatlayer1(in_dim, hidden_dim), gatlayer2(hidden_dim, out_dim){
    register_module("gatlayer1", gatlayer1);
    register_module("gatlayer2", gatlayer2);
}

torch::Tensor GAT::forward(torch::Tensor input,
                           c10::intrusive_ptr<SnapWrap> snaph)
{
    torch:: Tensor h = gatlayer1 -> forward(input, snaph);
    h = torch::relu(h);
    h = gatlayer2 -> forward(h, snaph);

    return h;
}
