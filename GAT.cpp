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

torch::Tensor edge_softmax(snap_t<dst_id_t>* snaph,  const torch::Tensor & efficient_score){
    degree_t nebr_count = 0;
    vid_t sid;
    nebr_reader_t<dst_id_t> header;
    vid_t v_count = snaph->get_vcount();
    //return the value of each node after gather procedure
    //int output_dim = input_feature.size(1);
    int64_t count = 0;
    std::map<int, torch::Tensor> mailbox;
    for (vid_t v = 0; v < v_count; v++) {
        nebr_count = snaph->get_nebrs_out(v, header);
        if (nebr_count == 0) {
            continue;
        }
        // the node j scatter it's message to all neighors
        for (degree_t i = 0; i < nebr_count; ++i) {
            sid = TO_SID(get_sid(header[i]));
            //std::cout << "the sid" <<std::endl;
            torch::Tensor message = torch::zeros({1,1});
            message[0] = count;
            std::cout<<"nani1111"<<std::endl;
            //If mailbox is empty, we initilize the mailbox
            if (mailbox.count(sid) == 0) {
                //std::cout<<message<<std::endl;
                mailbox[sid] = message;
                count = count + 1;
                continue;

            }
             std::cout<<"nani2222"<<std::endl;
            // If the mailbox is not emptys, we concatenate the received message with the new message
            torch::Tensor temp = torch::cat({mailbox[sid], message}, 0);
            //std::cout<<"kkk"<<std::endl;
            //std::cout << temp<< std::endl;
            std::cout<<"nani3333"<<std::endl;
            mailbox[sid] = temp;
            std::cout<<"nani4444"<<std::endl;
            count = count + 1;
        }
    }

        //apply softmax for each node
        for (std::map<int,torch::Tensor>::iterator it = mailbox.begin(); it!= mailbox.end(); ++it){
                torch::Tensor edge_id_list = it -> second;
                int length = edge_id_list.size(0);
                torch::Tensor original_score = torch::zeros({length,1});
            for (int i = 0; i < length; i++) {
                int64_t edge_id = edge_id_list[i].item().to<int64_t>();
                std::cout<<"nani5555"<<std::endl;
                original_score[i] = efficient_score[edge_id];
                }

            at::Tensor alpha = torch::softmax(original_score, 1);// apply softmax for specific node
            for (int i = 0; i < length; i++) {
                int64_t edge_id = edge_id_list[i].item().to<int64_t>();

                std::cout<<"nani6666"<<std::endl;
                std::cout<<edge_id<<std::endl;
                efficient_score[edge_id] = alpha[i];// update the efficient score
                 std::cout<<"nani7777"<<std::endl;
            }
            //std::cout<<"aaaa"<<std::endl;
            //std::cout<<temp<<std::endl;
        }




    return efficient_score;



}


torch::Tensor add_by_edge(snap_t<dst_id_t>* snaph, const torch::Tensor & input_left, const torch::Tensor & input_right){
    //snap_t<dst_id_t>* snaph = 0;
    degree_t nebr_count = 0;
    vid_t sid;
    nebr_reader_t<dst_id_t> header;

    vid_t v_count = snaph->get_vcount();

    //return the value of each node after gather procedure
    //int output_dim = input_feature.size(1);
     
    int edge_count = 0;
    for (vid_t v = 0; v < v_count; v++) {
        nebr_count = snaph->get_nebrs_out(v, header);
        if (nebr_count == 0) {
            continue;
        }
        // the node j scatter it's message to all neighors
        for (degree_t i = 0; i < nebr_count; ++i) {
            edge_count = edge_count + 1;
        }
    }


    torch::Tensor result = torch::zeros({edge_count,1});
    int count = 0;
    

    for (vid_t v = 0; v < v_count; v++) {
        nebr_count = snaph->get_nebrs_out(v, header);
        if (nebr_count == 0) {
            continue;
        }
        // the node j scatter it's message to all neighors
        for (degree_t i = 0; i < nebr_count; ++i) {
            sid = TO_SID(get_sid(header[i]));
            //std::cout << "the sid" <<std::endl;
            torch::Tensor input_feature_left = input_left[v];
            torch::Tensor input_feature_right = input_right[sid];
            torch::Tensor temp = torch::add(input_feature_left , input_feature_right);
            //temp = temp.reshape({1, 1});
            result[count] = temp;
            count = count + 1;
            //If mailbox is empty, we initilize the mailbox

            //torch::Tensor message = input_feature[sid] * (float)1 / (float) nebr_count;


        }
    }
    return result;

}


torch::Tensor gat_update_all_1(const torch::Tensor & input_feature, snap_t<dst_id_t>* snaph, const torch::Tensor &  edge_score_by_softmax, string gather_operator,  int64_t reverse) {
    //snap_t<dst_id_t>* snaph = 0;
    degree_t nebr_count = 0;
    vid_t sid;
    nebr_reader_t<dst_id_t> header;
    int output_dim = input_feature.size(1);


    //build the mailbox
    std::map<int, torch::Tensor> mailbox;
    vid_t v_count = snaph->get_vcount();
    std::cout << "-> begin the scatter!!" << std::endl;
    int count = 0;
    for (vid_t v = 0; v < v_count; v++) {

        //if ( 0 == (nebr_count = snaph->get_nebrs_out(v, header))) continue;
        if (reverse == 1){
            nebr_count = snaph->get_nebrs_in(v, header);
        } else {
            nebr_count = snaph->get_nebrs_out(v, header);
        }
        //std::cout << nebr_count  << std::endl;
        // if one node do not have any neighbor, we do not scatter it's message
        if (nebr_count == 0){
            continue;
        }
        // the node j scatter it's message to all neighors
        for (degree_t i = 0; i < nebr_count; ++i) {
            sid = TO_SID(get_sid(header[i]));
            //std::cout << "the sid" <<std::endl;
            torch::Tensor message = input_feature[sid];
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
    //gather procedure, each node gather it's received message by the method defined by 'gather_operator'
    std::cout << "-> gather procedure begins" << (int) reverse << std::endl;

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
        //std::cout<<"aaaa"<<std::endl;
        //std::cout<<temp<<std::endl;
        std::cout<< "ceshi" << std::endl;

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
    //std::cout<< "print the mailbox output" << std::endl;
    //std::cout<< result<< std::endl;

    return result;



}


/*
torch::Tensor gat_result1(const torch::Tensor & input_feature, snap_t<dst_id_t>* snaph, int64_t reverse, GATlayer * gatlayer2) {
    //snap_t<dst_id_t>* snaph = 0;
    degree_t nebr_count = 0;
    vid_t sid;
    nebr_reader_t<dst_id_t> header;
    int output_dim = input_feature.size(1);

    vid_t v_count = snaph->get_vcount();

    //return the value of each node after gather procedure
    //int output_dim = input_feature.size(1);
    torch::Tensor result = torch::zeros({v_count,output_dim});
    std::map<int, torch::Tensor> mailbox;

    for (vid_t v = 0; v < v_count; v++) {

        if (reverse == 1) {
            nebr_count = snaph->get_nebrs_in(v, header);
        } else {
            nebr_count = snaph->get_nebrs_out(v, header);
        }

        if (nebr_count == 0){
            continue;
        }
        // the node j scatter it's message to all neighors
        for (degree_t i = 0; i < nebr_count; ++i) {
            sid = TO_SID(get_sid(header[i]));
            //std::cout << "the sid" <<std::endl;

            //If mailbox is empty, we initilize the mailbox
            if (mailbox.count(sid) == 0){
                //torch::Tensor message = input_feature[sid] * (float)1 / (float) nebr_count;
                torch::Tensor message = input_feature[sid];
                
                torch::Tensor temp_message = torch::zeros({1,output_dim});
                for (int a = 0; a < output_dim; a++){
                    temp_message[1][a] = message[a];
                }
            
                //std::cout<<"haaaaa?"<<std::endl;
                //std::cout<< temp_message << std::endl;
                message = message.reshape({1,output_dim});
                //std::cout<<"nani?"<<std::endl;
                //std::cout<<message<<std::endl;
                mailbox[sid] = message;
                continue;

            }
            // If the mailbox is not emptys, we concatenate the received message with the new message
            //torch::Tensor message = input_feature[sid] * (float)1 / (float) nebr_count;
            torch::Tensor message = input_feature[sid];

            message = message.reshape({1, output_dim});

            torch::Tensor temp = torch::cat({mailbox[sid], message}, 0);
            //std::cout<<"kkk"<<std::endl;
            //std::cout << temp<< std::endl;
            mailbox[sid] = temp;
        }
    }

    torch::Tensor temp;
    for (std::map<int,torch::Tensor>::iterator it = mailbox.begin(); it!= mailbox.end(); ++it){


        vid_t node_id = it -> first;
        torch::Tensor message = it -> second;
        int num_nebr = input_feature.size(0);
        torch::Tensor node_input = input_feature[node_id];

        node_input = node_input.reshape({1,output_dim});
        torch::Tensor temp_append = node_input;
        for (degree_t i = 1; i < num_nebr; ++i) {
            temp_append = torch::cat((temp_append, node_input), 0);
        }


        // concatenate the node feature with message feature
        temp = torch::cat((message, temp_append), 1);
        torch::Tensor temp = gatlayer2 ->linear2(temp);// equation 2 here
        torch::Tensor efficient_score = torch::leaky_relu(temp, 0.2);
        at::Tensor alpha = torch::softmax(efficient_score, 1);//equation3 here
        //check here.
        torch::Tensor h = torch::sum(alpha * message, 0); // equation (4) weighted sum for that node

        it->second = h;
    }

    //return the value of each node after gather procedure
    //int output_dim = input_feature.size(1);
    result = torch::zeros({v_count,output_dim});//{v_count, 1} means the tensor is 1 dimension,otherwise, we cannot concatenated tensors
    for (vid_t v = 0; v < v_count; v++) {
        //loop the mailbox by key:node_id, value:tensor,
        //if the node did not reveive any message, the value of that node is 0
        map<int,torch::Tensor>::iterator it = mailbox.find(v);
        if (it == mailbox.end()) continue;
        result[v] = mailbox[v];
    }

    return result;

}


*/
/*

class GAT_result : public Function<GAT_result> {
public:
    // Note that both forward and backward are static functions

    // bias is an optional argument
    static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input,
                                 c10::intrusive_ptr<SnapWrap> snaph, GATlayer * gatlayer2) {
        ctx->save_for_backward({input});
        ctx->saved_data["snaph"] = snaph;
        //ctx->save_for_backward({input, snaph, gather_operator, reverse});
        auto output = gat_result1(input, snaph -> snaph, 0, gatlayer2);
        return output;
    }

    static tensor_list backward(AutogradContext *ctx, auto grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto snaph = ctx->saved_data["snaph"].toCustomClass<SnapWrap>();
        //auto grad_output = grad_outputs[0];
        int64_t reverse = 1;
        auto grad_graph_snaph = torch::Tensor();
        auto grad_input = gat_result1(grad_outputs[0], snaph->snaph, reverse);//TODO  check here XXX
        cout << "grad output1" << endl;

        return {grad_input, grad_graph_snaph};
    }
};
*/

class GAT_update_all : public Function<GAT_update_all> {
public:
    // Note that both forward and backward are static functions

    // bias is an optional argument
    static torch::Tensor forward(AutogradContext *ctx, torch::Tensor input,
                                 c10::intrusive_ptr<SnapWrap> snaph, torch::Tensor edge_score_by_softmax) {
        ctx->save_for_backward({input, edge_score_by_softmax});
        ctx->saved_data["snaph"] = snaph;
        //ctx->save_for_backward({input, snaph, gather_operator, reverse});
        auto output = gat_update_all_1(input, snaph -> snaph, edge_score_by_softmax,"max", 0);
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
        auto grad_edge_score_by_softmax1 = grad_outputs[2] * edge_score_by_softmax - edge_score_by_softmax;// not sure
        auto grad_input = gat_update_all_1(grad_outputs[0], snaph->snaph, edge_score_by_softmax,"max" ,reverse);
        cout << "grad output1" << endl;

        return {grad_input, grad_graph_snaph, grad_edge_score_by_softmax1};
    }
};


GATlayerImpl::GATlayerImpl(int64_t in_dim, int64_t out_dim)
        : linear1(register_module("linear1", torch::nn::Linear(in_dim, out_dim))) {
    W_left = register_parameter("W_left", torch::randn({out_dim, 1}));
    W_right = register_parameter("W_right", torch::randn({out_dim, 1}));

    //register_module("linear1", linear1);

}

torch::Tensor GATlayerImpl::forward(torch::Tensor input, c10::intrusive_ptr<SnapWrap> snaph){

    torch::Tensor map_input = linear1(input); // equation1
    std::cout<<"nani?"<< std::endl;
    torch::Tensor input_left = torch::matmul(map_input, W_left);
    std::cout<<"nani2"<<std::endl;
    torch::Tensor input_right = torch::matmul(map_input, W_right);
    std::cout<<"nani3"<<std::endl;
    torch::Tensor edge_score = add_by_edge(snaph -> snaph, input_left, input_right);//equation 2 by edge
    std::cout<<"nani4"<<std::endl;
    torch::Tensor efficient_score = torch::leaky_relu(edge_score, 0.2); // double check
    std::cout<<"nani5"<<std::endl;
    torch::Tensor edge_score_by_softmax = edge_softmax(snaph -> snaph, efficient_score);//get final significiant for each edge
    std::cout<<"nani6"<<std::endl;
    torch::Tensor h = GAT_update_all::apply(map_input, snaph, edge_score_by_softmax);
    std::cout<<"nani7"<<std::endl;



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


/*
vector<torch::Tensor> GAT::parameters(){
    
    std::vector<torch::Tensor> result;
    torch::Tensor para1 = gatlayer1.linear1.parameters() ;
    torch::Tensor para2 = gatlayer2.linear1.parameters() ;
    torch::Tensor para3 = gatlayer1.W_left;
    torch::Tensor para4 = gatlayer1.W_right;

    torch::Tensor para5 = gatlayer2.W_left;
    torch::Tensor para6 = gatlayer2.W_right;

    result.push_back(para1);
    result.push_back(para2);
    result.push_back(para3);
    result.push_back(para4);
    result.push_back(para5);
    result.push_back(para6);
    return result; 
}
*/
