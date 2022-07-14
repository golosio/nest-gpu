/*
 *  This file is part of NESTGPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NESTGPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NESTGPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NESTGPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */


#ifndef GETSPIKE_H
#define GETSPIKE_H
#include "utilities.h"
#include "send_spike.h"
#include "new_connect.h"
#include "node_group.h"

extern __constant__ NodeGroupStruct NodeGroupArray[];
extern __device__ signed char *NodeGroupMap;

template<int i_func>
__device__  __forceinline__ void NestedLoopFunction(int i_spike, int i_syn);

//////////////////////////////////////////////////////////////////////
// This is the function called by the nested loop
// that collects the spikes
template<>
__device__  __forceinline__ void NestedLoopFunction<0>(int i_spike, int i_syn)
{
  int i_source = SpikeSourceIdx[i_spike];
  int i_source_conn_group = SpikeConnIdx[i_spike];
  float height = SpikeHeight[i_spike];
  int ig = ConnGroupIdx0[i_source] + i_source_conn_group;

  // USARE BLOCK_SIZE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // trovare i_block e i_block_conn
  int64_t i_conn = ConnGroupIConn0[ig] + i_syn;
  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  connection_struct conn = ConnectionArray[i_block][i_block_conn];
  uint target_port = conn.target_port;
  int i_target = target_port >> MaxPortNBits;
  uint port = target_port & PortMask;
  unsigned char syn_group = 0; //ValueSubarray[i_conn].syn_group;
  float weight = conn.weight;
  //printf("handles spike %d src %d conn %d syn %d target %d"
  //" port %d weight %f\n",
  //i_spike, i_source, i_conn, i_syn, i_target,
  //port, weight);
  
  /////////////////////////////////////////////////////////////////
  int i_group=NodeGroupMap[i_target];
  int i = port*NodeGroupArray[i_group].n_node_ + i_target
    - NodeGroupArray[i_group].i_node_0_;
  double d_val = (double)(height*weight);

  atomicAddDouble(&NodeGroupArray[i_group].get_spike_array_[i], d_val);
  /* RIMOSSO TEMPORANEAMENTE, RIPRISTINARE CON DOVUTE MODIFICHE
  if (syn_group>0) {
    ConnectionGroupTargetSpikeTime[i_conn*NSpikeBuffer+i_source][i_syn]
      = (unsigned short)(NESTGPUTimeIdx & 0xffff);
    
    long long Dt_int = NESTGPUTimeIdx - LastRevSpikeTimeIdx[i_target];
     if (Dt_int>0 && Dt_int<MAX_SYN_DT) {
       SynapseUpdate(syn_group, &ConnectionGroupTargetWeight
		    [i_conn*NSpikeBuffer+i_source][i_syn],
		     -NESTGPUTimeResolution*Dt_int);
    }
  }
  */
  ////////////////////////////////////////////////////////////////
}
///////////////

//////////////////////////////////////////////////////////////////////
// This is the function called by the nested loop
// that collects the spikes from poisson generators
template<>
__device__  __forceinline__ void NestedLoopFunction<2>(int i_spike, int i_syn)
{
  int i_source = PoissonSpikeSourceIdx[i_spike];
  int i_source_conn_group = PoissonSpikeConnIdx[i_spike];
  float height = PoissonSpikeHeight[i_spike];
  int ig = ConnGroupIdx0[i_source] + i_source_conn_group;

  // USARE BLOCK_SIZE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // trovare i_block e i_block_conn
  int64_t i_conn = ConnGroupIConn0[ig] + i_syn;
  uint i_block = (uint)(i_conn / ConnBlockSize);
  int64_t i_block_conn = i_conn % ConnBlockSize;
  connection_struct conn = ConnectionArray[i_block][i_block_conn];
  uint target_port = conn.target_port;
  int i_target = target_port >> MaxPortNBits;
  uint port = target_port & PortMask;
  unsigned char syn_group = 0; //ValueSubarray[i_conn].syn_group;
  float weight = conn.weight;
  //printf("handles spike %d src %d conn %d syn %d target %d"
  //" port %d weight %f\n",
  //i_spike, i_source, i_conn, i_syn, i_target,
  //port, weight);
  
  /////////////////////////////////////////////////////////////////
  int i_group=NodeGroupMap[i_target];
  int i = port*NodeGroupArray[i_group].n_node_ + i_target
    - NodeGroupArray[i_group].i_node_0_;
  double d_val = (double)(height*weight);

  atomicAddDouble(&NodeGroupArray[i_group].get_spike_array_[i], d_val);
  ////////////////////////////////////////////////////////////////
}
///////////////


__global__ void GetSpikes(double *spike_array, int array_size, int n_port,
			  int n_var,
			  float *port_weight_arr,
			  int port_weight_arr_step,
			  int port_weight_port_step, //float *y_arr);
			  float *port_input_arr,
			  int port_input_arr_step,
			  int port_input_port_step);

#endif
