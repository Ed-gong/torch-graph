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
    sid_t      tile_count = snaph->get_vcount();
    sid_t      v_count    = _global_vcount;
    vid_t      p = (v_count >> bit_shift1)
                   + (0 != (v_count & part_mask1_2));

//    double start1 = mywtime();
    memset(status, 255, v_count*sizeof(level_t));

    degree_t nebr_count = 0;
    header_t<T> header;
    T dst;
    vid_t index = 0, m, n, offset;
    int64_t reverse = 0;
    int output_dim = input_feature.size(1);
    torch::Tensor output = torch::zeros({v_count, 1});
//    #pragma omp for nowait
    for (vid_t i = 0; i < p; ++i) {
        for (vid_t j = 0; j < p; ++j) {
            offset = ((i*p + j) << bit_shift2);
            for (vid_t s_i = 0; s_i < p_p; s_i++) {
                for (vid_t s_j = 0; s_j < p_p; s_j++) {
                    index = offset + ((s_i << bit_shift3) + s_j);
                    m = ((i << bit_shift3) + s_i) << bit_shift2;
                    n = ((j << bit_shift3) + s_j) << bit_shift2;
                    update_all_vertex(input_feature, output, snaph, reverse, output_dim);// apply scatter in each tile
                }
            }
        }
    }

    return output;

}



void update_all_vertex(const torch::Tensor & input_feature, const torch::Tensor & output, snap_t<dst_id_t>* snaph,  int64_t reverse, int output_dim) {
    //snap_t<dst_id_t>* snaph = 0;
    header_t<T> header;
    degree_t nebr_count = snaph->start_out(index, header);
    if (0 == nebr_count) return 0;
    T dst;
    snb_t snb;
    vid_t v_count = snaph->get_vcount();
    std::cout << "-> begin the scatter!!" << std::endl;

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
        output[dst] = output[dst] + message;

    }
}
