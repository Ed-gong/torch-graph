//
// Created by yidong on 10/16/20.
//
#pragma once
#include <omp.h>
#include <algorithm>
#include <iostream>
#include <list>

#include "graph_view.h"
#include "onesnb.h"
using std::min;
using namespace std;

template<class T>
torch::Tensor mem_page_rank_snb(gview_t<T>* viewh, const torch::Tensor & input_feature)
{
    snap_t<T>* snaph = (snap_t<T>*)viewh;
    int		   top_down   = 1;
    sid_t	   frontier   = 0;
    sid_t      tile_count = snaph->get_vcount();
    sid_t      v_count    = _global_vcount;
    vid_t      p = (v_count >> bit_shift1)
                   + (0 != (v_count & part_mask1_2));

//    double start1 = mywtime();
    memset(status, 255, v_count*sizeof(level_t));

    map<int, torch::Tensor>* mailbox = new map<int, torch::Tensor>();
    degree_t nebr_count = 0;
    header_t<T> header;
    T dst;
    vid_t index = 0, m, n, offset;
    int64_t reverse = 0;
    int output_dim = input_feature.size(1);
//    #pragma omp for nowait
    for (vid_t i = 0; i < p; ++i) {
        for (vid_t j = 0; j < p; ++j) {
            offset = ((i*p + j) << bit_shift2);
            for (vid_t s_i = 0; s_i < p_p; s_i++) {
                for (vid_t s_j = 0; s_j < p_p; s_j++) {
                    index = offset + ((s_i << bit_shift3) + s_j);
                    m = ((i << bit_shift3) + s_i) << bit_shift2;
                    n = ((j << bit_shift3) + s_j) << bit_shift2;
                    scatter_by_vertex(input_feature, <int, torch::Tensor>* mailbox, snaph, reverse, output_dim);// apply scatter in each tile
                }
            }
        }
    }
   result = gather_by_mailbox(mailbox, snaph,  reverse, output_dim, v_count);//apply gather after we scatter in each tile

return result;

}



torch::Tensor gather_by_mailbox(<int, torch::Tensor>* mailbox, snap_t<dst_id_t>* snaph,  int64_t reverse, int output_dim, sid_t v_count) {
    //gather procedure, each node gather it's received message by the method defined by 'gather_operator'
    std::cout << "-> gather procedure begins" << (int) reverse << std::endl;

    torch::Tensor temp;
    for (std::map<int,torch::Tensor>::iterator it = mailbox -> begin(); it!= mailbox -> end(); ++it){

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
        map<int,torch::Tensor>::iterator it = mailbox -> find(v);
        if (it == mailbox -> end()) continue;
        result[v] = mailbox->operator[](v);
    }
    //std::cout<< "print the mailbox output" << std::endl;
    //std::cout<< result<< std::endl;

    return result;




}

void scatter_by_vertex(const torch::Tensor & input_feature, <int, torch::Tensor>* mailbox, snap_t<dst_id_t>* snaph,  int64_t reverse, int output_dim) {
    //snap_t<dst_id_t>* snaph = 0;
    header_t<T> header;
    degree_t nebr_count = snaph->start_out(index, header);
    if (0 == nebr_count) return 0;
    T dst;
    snb_t snb;

    //build the mailbox
    std::map<int, torch::Tensor> mailbox;
    vid_t v_count = snaph->get_vcount();
    std::cout << "-> begin the scatter!!" << std::endl;
    int count = 0;

    for (degree_t e = 0; e < nebr_count; ++e) {

        snaph->next(header, dst);
        snb = get_snb(dst);

        //if ( 0 == (nebr_count = snaph->get_nebrs_out(v, header))) continue;
        if (reverse == 0){
            snb_t dst = snb.dst;
            snb_t src = snb.src;
        } else {
            snb_t dst = snb.src;
            snb_t src = snb.dst;
        }

        // the node j scatter it's message to all neighors
        //std::cout << "the sid" <<std::endl;
        torch::Tensor message = input_feature[src];// no edge scope here
        message = message.reshape({1, output_dim});
        //If mailbox is empty, we initilize the mailbox
        if (mailbox -> count(dst) == 0){
            //std::cout<<"nani?"<<std::endl;
            //std::cout<<message<<std::endl;
            mailbox->operator[](dst) = message;
            continue;

        }
        // If the mailbox is not emptys, we concatenate the received message with the new message
        torch::Tensor temp = torch::cat({mailbox->operator[](dst), message}, 0);
        //std::cout<<"kkk"<<std::endl;
        //std::cout << temp<< std::endl;
        mailbox->operator[](dst) = temp;
    }
}
