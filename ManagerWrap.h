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
#include "Scatter_gather.h"

template <class T>
torch::Tensor scatter_gather2(snap_t<T>* snaph,
                             const torch::Tensor & input_feature,
                             string gather_operator, int64_t reverse);

template <class T>
torch::Tensor check_adjacency_matrix1(snap_t<T>* snaph);

template<class T>
struct ManagerWrap_t : torch::CustomClassHolder {
    pgraph_manager_t<T>* manager;

    ManagerWrap_t(int64_t flags, int64_t node_number, string path);

    torch::Tensor scatter_gather(const torch::Tensor & input_feature, string gather_operator, 
                            c10::intrusive_ptr<SnapWrap_t<T>> snaph, int64_t reverse);  
    
    torch::Tensor check_adjacency_matrix(c10::intrusive_ptr<SnapWrap_t<T>> snaph);   
    
    void reg_static_view(c10::intrusive_ptr<SnapWrap_t<T>> snaph);
};

template <class T>
torch::Tensor check_adjacency_matrix1(snap_t<T>* snaph)
{
    degree_t nebr_count = 0;
    nebr_reader_t<T> header;
    vid_t sid;
    vid_t v_count = snaph->get_vcount();
    torch::Tensor adj_matrix = torch::zeros({v_count,v_count});


    for (vid_t v = 0; v < v_count; v++) {
        nebr_count = snaph -> get_nebrs_out(v, header);
        //cout << "Vertex ID: " << v << endl;
        for (degree_t i = 0; i < nebr_count; ++i) {
            sid = TO_SID(get_sid(header[i]));
            //std::cout << sid << ",";
            adj_matrix[v][sid] = 1;
        }
        //cout << endl;
    }

    return adj_matrix;
}
    

template <class T>
torch::Tensor scatter_gather2(snap_t<T>* snaph, 
                             const torch::Tensor & input_feature, 
                             string gather_operator, int64_t reverse)
{
    degree_t nebr_count = 0;
    vid_t sid;
    nebr_reader_t<T> header;
    int output_dim = input_feature.size(1);


    //build the mailbox
    std::map<int, torch::Tensor> mailbox;
    vid_t v_count = snaph->get_vcount();
    //std::cout << "-> begin the scatter!!" << std::endl;

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

template <class T>
ManagerWrap_t<T>::ManagerWrap_t(int64_t flags, int64_t node_number, string path)  {
    manager = new pgraph_manager_t<T>();

    std::cout << "-> Create in-Memory Graph !!" << std::endl;
    manager -> schema(flags);
    manager -> setup_graph(node_number);
    manager -> prep_graph(path,"");
}

template <class T>
torch::Tensor ManagerWrap_t<T>::scatter_gather(const torch::Tensor & input, string gather_operator,
                                          c10::intrusive_ptr<SnapWrap_t<T>> snaph, int64_t reverse)
{
    int dim = input.size(0);
    int output_dim = input.size(1);
    array2d_t<float> input_array(input.data_ptr<float>(), dim, output_dim);
    
    torch::Tensor result = torch::zeros({dim, output_dim});
    array2d_t<float> output_array(result.data_ptr<float>(), dim, output_dim);
    
    _gspmv(snaph->snaph, input_array, output_array, gather_operator, reverse);
    return result;
}

template <class T>
torch::Tensor ManagerWrap_t<T>::check_adjacency_matrix(c10::intrusive_ptr<SnapWrap_t<T>> snaph)
{
    torch::Tensor result = check_adjacency_matrix1(snaph->snaph);
    return result;
}

template <class T>
void ManagerWrap_t<T>::reg_static_view(c10::intrusive_ptr<SnapWrap_t<T>> snaph)
{
    pgraph_t<T>* pgraph_1 = manager->pgraph;
    snaph->snaph = create_static_view(pgraph_1, 1);//1 means it is a direct graph
    
    //TODO: Comment it out later.
    /*std::cout<<"print the graph info"<<std::endl;
    snap_t<T>* h = snaph->snaph;
    degree_t nebr_count = 0;
    vid_t sid;
    nebr_reader_t<T> header;
    vid_t v_count = h->get_vcount();

    for (vid_t v = 0; v < v_count ; ++v) {
        nebr_count = h->get_nebrs_out(v, header);
        std::cout << "vid = " << v << ":"; 
        for (degree_t i = 0; i < nebr_count; i++) {
            std::cout << TO_SID(get_sid(header[i])) << ",";
        }
        std::cout << std::endl;
    }*/
}

typedef ManagerWrap_t<dst_id_t> ManagerWrap;

#endif //UNTITLED1_MANAGERWRAP_H

