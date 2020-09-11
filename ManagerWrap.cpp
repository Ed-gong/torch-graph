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
#include "ManagerWrap.h"

using namespace std;
using torch::Tensor;
using torch::autograd::Node;
using torch::autograd::deleteNode;
using torch::autograd::SavedVariable;
using torch::autograd::variable_list;
using torch::autograd::tensor_list;

//using torch::autograd::as_variable;
//using torch::autograd::as_variable_ref;
//using at::Scalar;
//using torch::autograd::compute_requires_grad;
//using torch::autograd::collect_next_edges;
//using torch::autograd::flatten_tensor_args;

graph*g = new graph();

snap_t<dst_id_t>* check_current_graph(plaingraph_manager_t<dst_id_t>* manager){
    pgraph_t<dst_id_t>* pgraph_1 = manager->pgraph;
   //the input_message is the scatter messge
    snap_t<dst_id_t>* snaph = create_static_view(pgraph_1, 1);//1 means it is a direct graph
    return snaph;
}

torch::Tensor scatter_gather1(snap_t<dst_id_t>* snaph, 
                             const torch::Tensor & input_feature, 
                             string gather_operator, int64_t reverse)
{
    
    //snap_t<dst_id_t>* snaph = 0;
    degree_t nebr_count = 0;
    vid_t sid;
    nebr_reader_t<dst_id_t> header;
    int output_dim = input_feature.size(1);
    
    vid_t v_count = snaph->get_vcount();

    //return the value of each node after gather procedure
    //int output_dim = input_feature.size(1);
    torch::Tensor result = torch::zeros({v_count,output_dim});
    //{v_count, 1} means the tensor is 1 dimension,otherwise, we cannot concatenated tensors

    std::cout << "-> gather procedure begins" << (int) reverse << std::endl;

    for (vid_t v = 0; v < v_count; v++) {
        if (reverse == 1) {
            nebr_count = snaph->get_nebrs_in(v, header);
        } else {
            nebr_count = snaph->get_nebrs_out(v, header);
        }
        // if one node do not have any neighbor, we do not scatter it's message
        if (nebr_count == 0){
                continue;
        }
        // the node j scatter it's message to all neighors
        torch::Tensor message = input_feature[v];
        for (degree_t i = 0; i < nebr_count; ++i) {
            sid = TO_SID(get_sid(header[i]));
            message += input_feature[sid];
        }
        result[v] = message/nebr_count;
    }

    return result;
}

torch::Tensor scatter_gather2(snap_t<dst_id_t>* snaph, 
                             const torch::Tensor & input_feature, 
                             string gather_operator, int64_t reverse)
{
    
    //snap_t<dst_id_t>* snaph = 0;
    degree_t nebr_count = 0;
    vid_t sid;
    nebr_reader_t<dst_id_t> header;
    int output_dim = input_feature.size(1);


    //build the mailbox
    std::map<int, torch::Tensor> mailbox;
    vid_t v_count = snaph->get_vcount();
    std::cout << "-> begin the scatter!!" << std::endl;

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
            
            //If mailbox is empty, we initilize the mailbox
            if (mailbox.count(sid) == 0){
                //torch::Tensor message = input_feature[sid] * (float)1 / (float) nebr_count;
                torch::Tensor message = input_feature[sid];
                /*
                torch::Tensor temp_message = torch::zeros({1,output_dim});
                for (int a = 0; a < output_dim; a++){

                    temp_message[1][a] = message[a];
                }
                */
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



ManagerWrap::ManagerWrap(int64_t flags, int64_t node_number, string path)  {
    manager = new plaingraph_manager_t<dst_id_t>();

    std::cout << "-> initilize the DAG!!" << std::endl;
    manager -> schema(0);
    //manager -> setup_graph(100);
    manager -> setup_graph(node_number);
    //manager -> prep_graph("/home/datalab/data/test1","");
    manager -> prep_graph(path,"");
}
  
torch::Tensor ManagerWrap::scatter_gather(const torch::Tensor & input_feature, string gather_operator,
                                          c10::intrusive_ptr<SnapWrap> snaph, int64_t reverse)
{
    torch::Tensor result = scatter_gather1(snaph->snaph, input_feature, gather_operator, reverse);
    return result;
}

void ManagerWrap::create_static_view(c10::intrusive_ptr<SnapWrap> snaph)
{
    //create view here
    snaph->snaph = check_current_graph(manager);
    std::cout<<"print the graph info"<<std::endl;
    snap_t<dst_id_t>* h = snaph->snaph;
    degree_t nebr_count = 0;
    vid_t sid;
    nebr_reader_t<dst_id_t> header;
    vid_t v_count = h->get_vcount();

    for (vid_t v = 0; v < v_count ; ++v) {
        nebr_count = h->get_nebrs_out(v, header);
        std::cout << "vid = " << v << ":"; 
        for (degree_t i = 0; i < nebr_count; i++) {
            std::cout << TO_SID(get_sid(header[i])) << ",";
        }
        std::cout << std::endl;
    }
}
