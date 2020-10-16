#pragma once
#include <omp.h>
#include <algorithm>
#include <iostream>
#include <list>

#include "graph_view.h"
#include "onesnb.h"
using std::min;
using namespace std;

//typedef uint8_t level_t;
typedef vid_t level_t;
#ifdef _MPI
extern MPI_Datatype bfs_status_type; 
#endif
static level_t MAX_LEVEL = -1L;

void print_bfs_summary(level_t* status, vid_t v_count);



//try_bfs
template<class T>
void bfs_try(gview_t<T>* snaph, sid_t root){

    sid_t numVertices = snaph->get_vcount();
    degree_t nebr_count = 0;
    vid_t v = 0;
    bool *visited = new bool[numVertices];

    for (int i = 0; i < numVertices; i++)
        visited[i] = false;

    list<sid_t> queue;
    nebr_reader_t<T> local_adjlist;
    visited[root] = true;
    queue.push_back(root);

    while (!queue.empty()) {
        sid_t currVertex = queue.front();
        cout << "Visited " << currVertex << ", ";
        queue.pop_front();
        nebr_count = snaph->get_nebrs_out(v, local_adjlist);
        for (degree_t i = 0; i < nebr_count; ++i) {
            sid_t adjVertex = get_sid(local_adjlist[i]);
            if (!visited[adjVertex]) {
                visited[adjVertex] = true;
                queue.push_back(adjVertex);
                    }
                }
    }
}

template<class T>
void bfs_try1(gview_t<T>* snaph, sid_t root) {
	int		level      = 0;
	int		top_down   = 1;
	sid_t	frontier   = 0;
    sid_t   v_count    = snaph->get_vcount();
    level_t* status = (level_t*)malloc(sizeof(level_t)*v_count);
    memset(status, 255, v_count*sizeof(level_t));
    
    status[root] = level;
    sid_t sid;
    degree_t nebr_count = 0;
    nebr_reader_t<T> local_adjlist;

	do {
		frontier = 0;
        for (vid_t v = 0; v < v_count; v++) {
            if (status[v] != level) continue;

            nebr_count = snaph->get_nebrs_out(v, local_adjlist);
            for (degree_t i = 0; i < nebr_count; ++i) {
                sid = get_sid(local_adjlist[i]);
                if (status[sid] == MAX_LEVEL) {
                    status[sid] = level + 1;
                    ++frontier;
                    //cout << " " << sid << endl;
                }
            }
        }
		++level;
	} while (frontier);
		
    double end1 = mywtime();
    cout << "root = " << root << endl; 
    print_bfs_summary(status, v_count);
}





template<class T>
void mem_bfs_simple(gview_t<T>* snaph, level_t* status, sid_t root)
{
	int		level      = 0;
	int		top_down   = 1;
	sid_t	frontier   = 0;
    sid_t   v_count    = snaph->get_vcount();
    memset(status, 255, v_count*sizeof(level_t));
    
    status[root] = level;

    double start1 = mywtime();
    
	do {
		frontier = 0;
		//double start = mywtime();
		#pragma omp parallel num_threads(THD_COUNT) reduction(+:frontier)
		{
            sid_t sid;
            degree_t nebr_count = 0;
            nebr_reader_t<T> local_adjlist;
		    
            if (top_down) {
                #pragma omp for nowait
				for (vid_t v = 0; v < v_count; v++) {
					if (status[v] != level) continue;

                    nebr_count = snaph->get_nebrs_out(v, local_adjlist);
                    for (degree_t i = 0; i < nebr_count; ++i) {
                        sid = get_sid(local_adjlist[i]);
                        if (status[sid] == MAX_LEVEL) {
                            status[sid] = level + 1;
                            ++frontier;
                            //cout << " " << sid << endl;
                        }
                    }
				}
			} else {//bottom up
				#pragma omp for nowait
				for (vid_t v = 0; v < v_count; v++) {
					if (status[v] != MAX_LEVEL) continue;
                    
                    nebr_count = snaph->get_nebrs_in(v, local_adjlist);
                    for (degree_t i = 0; i < nebr_count; ++i) {
                        sid = get_sid(local_adjlist[i]);
                        if (status[sid] == level) {
                            status[v] = level + 1;
                            ++frontier;
                            break;
                        }
                    }
				}
		    }
        }

		//double end = mywtime();
	
		//cout << "Top down = " << top_down
		//     << " Level = " << level
        //     << " Frontier Count = " << frontier
		//     << " Time = " << end - start
		//     << endl;
	
        //Point is to simulate bottom up bfs, and measure the trade-off    
		if ((!snaph->is_unidir()) && ((frontier >= 0.002*v_count) /*|| level == 2*/)) {
			top_down = false;
		} else {
            top_down = true;
        }
		++level;
	} while (frontier);
		
    double end1 = mywtime();
    cout << "BFS Time = " << end1 - start1 << endl;
    cout << "root = " << root << endl; 
    print_bfs_summary(status, v_count);
}

template <class T>
index_t bfs_tile(snap_t<T>* snaph, vid_t index, level_t* lstatus, level_t* rstatus, level_t level)
{
    header_t<T> header; 
    degree_t nebr_count = snaph->start_out(index, header);
    if (0 == nebr_count) return 0;
    
    index_t frontier = 0;
    T dst;
    snb_t snb;
    for (degree_t e = 0; e < nebr_count; ++e) {
        snaph->next(header, dst);
        snb = get_snb(dst);
        if (lstatus[snb.src] == level && 
            rstatus[snb.dst] == MAX_LEVEL) {
            rstatus[snb.dst] = level + 1;
            ++frontier;
            //cout << " " << snb.dst + dst_offset << endl;
        }
        if (rstatus[snb.dst] == level && 
            lstatus[snb.src] == MAX_LEVEL) {
            lstatus[snb.src] = level + 1;
            ++frontier;
            //cout << " " << snb.src + src_offset << endl;
        }
    }
    return frontier;
}

template <class T>
index_t bfs_async_tile(snap_t<T>* snaph, vid_t index, level_t* lstatus, level_t* rstatus, level_t level)
{
    header_t<T> header; 
    degree_t nebr_count = snaph->start_out(index, header);
    if (0 == nebr_count) return 0;
    
    index_t frontier = 0;
    T dst;
    snb_t snb;
    level_t llevel = 0, rlevel = 0;
    for (degree_t e = 0; e < nebr_count; ++e) {
        snaph->next(header, dst);
        snb = get_snb(dst);
        if (lstatus[snb.src] == level && 
            rstatus[snb.dst] > level+1) {
            rstatus[snb.dst] = level + 1;
            ++frontier;
            //cout << " " << snb.dst + dst_offset << endl;
        }
        if (rstatus[snb.dst] == level && 
            lstatus[snb.src] > level + 1) {
            lstatus[snb.src] = level + 1;
            ++frontier;
            //cout << " " << snb.src + src_offset << endl;
        }
    }
    return frontier;
}

template<class T>
void mem_bfs_snb(gview_t<T>* viewh,
        level_t* status, sid_t root)
{
    snap_t<T>* snaph = (snap_t<T>*)viewh;
	int		   top_down   = 1;
	sid_t	   frontier   = 0;
    sid_t      tile_count = snaph->get_vcount();
    sid_t      v_count    = _global_vcount;
    vid_t      p = (v_count >> bit_shift1) 
                 + (0 != (v_count & part_mask1_2));
    
	double start1 = mywtime();
    memset(status, 255, v_count*sizeof(level_t));
    int	  level  = 0;
	status[root] = level;
    
	do {
		frontier = 0;
		double start = mywtime();
        #pragma omp parallel num_threads (THD_COUNT) reduction(+:frontier)
		{
            degree_t nebr_count = 0;
            header_t<T> header; 
            T dst;
            vid_t index = 0, m, n, offset;

            #pragma omp for nowait
            for (vid_t i = 0; i < p; ++i) {
                for (vid_t j = 0; j < p; ++j) {
                    offset = ((i*p + j) << bit_shift2); 
                    for (vid_t s_i = 0; s_i < p_p; s_i++) {
                        for (vid_t s_j = 0; s_j < p_p; s_j++) {
                            index = offset + ((s_i << bit_shift3) + s_j);
                            m = ((i << bit_shift3) + s_i) << bit_shift2;
                            n = ((j << bit_shift3) + s_j) << bit_shift2; 
                            frontier += bfs_tile(snaph, index, status+m, status+n, level); 
                        }
                    }
                }
            }
        }
        
		double end = mywtime();
	
		++level;
	} while (frontier);
		
    double end1 = mywtime();
    cout << "BFS Time = " << end1 - start1 << endl;
    print_bfs_summary(status, v_count);
}

inline void print_bfs_summary(level_t* status, vid_t v_count)
{
    vid_t vid_count = 0;
    int l = 0;
    do {
        vid_count = 0;
        //#pragma omp parallel for reduction (+:vid_count) 
        for (vid_t v = 0; v < v_count; ++v) {
            if (status[v] == l) {
                ++vid_count;
                /*if (l == 0) {
                    cerr << v << endl;
                }*/
            }
        }
        cout << " Level = " << l << " count = " << vid_count << endl;
        ++l;
    } while (vid_count !=0);
}


/*
template<class T>
void mem_bfs(vert_table_t<T>* graph_out, degree_t* degree_out, 
        vert_table_t<T>* graph_in, degree_t* degree_in,
        snapshot_t* snapshot, index_t marker, edgeT_t<T>* edges,
        vid_t v_count, level_t* status, sid_t root)
{
	int				level      = 1;
	int				top_down   = 1;
	sid_t			frontier   = 0;
    index_t         old_marker = 0;

    if (snapshot) { 
        old_marker = snapshot->marker;
    }
    
	double start1 = mywtime();
    //if (degree_out[root] == 0) { root = 0;}
	status[root] = level;
    
	do {
		frontier = 0;
		//double start = mywtime();
		#pragma omp parallel reduction(+:frontier)
		{
            sid_t sid;
            degree_t nebr_count = 0;
            degree_t local_degree = 0;
            degree_t delta_degree = 0;

            vert_table_t<T>* graph  = 0;
            delta_adjlist_t<T>* delta_adjlist;;
            vunit_t<T>* v_unit = 0;
            T* local_adjlist = 0;
		    
            if (top_down) {
                graph  = graph_out;
				
                #pragma omp for nowait
				for (vid_t v = 0; v < v_count; v++) {
					if (status[v] != level) continue;
					v_unit = graph[v].get_vunit();
                    if (0 == v_unit) continue;

					nebr_count     = degree_out[v];
                    delta_degree   = nebr_count;
                    delta_adjlist  = v_unit->delta_adjlist;
				    //cout << "delta adjlist " << delta_degree << endl;	
				    //cout << "Nebr list of " << v <<" degree = " << nebr_count << endl;	
                    
                    //traverse the delta adj list
                    while (delta_adjlist != 0 && delta_degree > 0) {
                        local_adjlist = delta_adjlist->get_adjlist();
                        local_degree = delta_adjlist->get_nebrcount();
                        degree_t i_count = min(local_degree, delta_degree);
                        for (degree_t i = 0; i < i_count; ++i) {
                            sid = get_nebr(local_adjlist, i);
                            if (status[sid] == 0) {
                                status[sid] = level + 1;
                                ++frontier;
                                //cout << " " << sid << endl;
                            }
                        }
                        delta_adjlist = delta_adjlist->get_next();
                        delta_degree -= local_degree;
                    }
				}
			} else {//bottom up
				graph = graph_in;
                int done = 0;
				
				#pragma omp for nowait
				for (vid_t v = 0; v < v_count; v++) {
					if (status[v] != 0 ) continue;
					v_unit = graph[v].get_vunit();
                    if (0 == v_unit) continue;

                    delta_adjlist = v_unit->delta_adjlist;
					
					nebr_count = degree_in[v];
                    done = 0;

                    //traverse the delta adj list
                    delta_degree = nebr_count;
                    while (delta_adjlist != 0 && delta_degree > 0) {
                        local_adjlist = delta_adjlist->get_adjlist();
                        local_degree = delta_adjlist->get_nebrcount();
                        degree_t i_count = min(local_degree, delta_degree);
                        for (degree_t i = 0; i < i_count; ++i) {
                            sid = get_nebr(local_adjlist, i);
                            if (status[sid] == level) {
                                status[v] = level + 1;
                                ++frontier;
                                done = 1;
                                break;
                            }
                        }
                        if (done == 1) break;
                        delta_adjlist = delta_adjlist->get_next();
                        delta_degree -= local_degree;
                    }
				}
		    }

            //on-the-fly snapshots should process this
            //cout << "On the Fly" << endl;
            vid_t src, dst;
            #pragma omp for schedule (static)
            for (index_t i = old_marker; i < marker; ++i) {
                src = edges[i].src_id;
                dst = get_dst(edges+i);
                if (status[src] == 0 && status[dst] == level) {
                    status[src] = level + 1;
                    ++frontier;
                    //cout << " " << src << endl;
                } 
                if (status[src] == level && status[dst] == 0) {
                    status[dst] = level + 1;
                    ++frontier;
                    //cout << " " << dst << endl;
                }
            }
        }

		//double end = mywtime();
	
		//cout << "Top down = " << top_down
		//     << " Level = " << level
        //     << " Frontier Count = " << frontier
		//     << " Time = " << end - start
		//     << endl;
	
        //Point is to simulate bottom up bfs, and measure the trade-off    
        if ((frontier >= 0.002*v_count) || level == 2) {
			top_down = false;
		} else {
            top_down = true;
        }
		++level;
	} while (frontier);
		
    double end1 = mywtime();
    cout << "BFS Time = " << end1 - start1 << endl;
    print_bfs_summary(status, level, v_count);
}
*/
