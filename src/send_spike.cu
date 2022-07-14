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





#include <config.h>
#include <stdio.h>
#include "send_spike.h"
#include "cuda_error.h"

int *d_SpikeNum;
int *d_SpikeSourceIdx;
int *d_SpikeConnIdx;
float *d_SpikeHeight;
int *d_SpikeTargetNum;

int *d_PoissonSpikeNum;
int *d_PoissonSpikeSourceIdx;
int *d_PoissonSpikeConnIdx;
float *d_PoissonSpikeHeight;
int *d_PoissonSpikeTargetNum;

__device__ int MaxSpikeNum;
__device__ int *SpikeNum;
__device__ int *SpikeSourceIdx;
__device__ int *SpikeConnIdx;
__device__ float *SpikeHeight;
__device__ int *SpikeTargetNum;

__device__ int *PoissonSpikeNum;
__device__ int *PoissonSpikeSourceIdx;
__device__ int *PoissonSpikeConnIdx;
__device__ float *PoissonSpikeHeight;
__device__ int *PoissonSpikeTargetNum;

__device__ void SendSpike(int i_source, int i_conn, float height,
			  int target_num)
{
  int pos = atomicAdd(SpikeNum, 1);
  if (pos>=MaxSpikeNum) {
    printf("Number of spikes larger than MaxSpikeNum: %d\n", MaxSpikeNum);
    *SpikeNum = MaxSpikeNum;
    return;
  }
  SpikeSourceIdx[pos] = i_source;
  SpikeConnIdx[pos] = i_conn;
  SpikeHeight[pos] = height;
  SpikeTargetNum[pos] = target_num;
}

__device__ void PoissonSendSpike(int i_source, int i_conn, float height,
				 int target_num)
{
  int pos = atomicAdd(PoissonSpikeNum, 1);
  if (pos>=MaxSpikeNum) {
    printf("Number of spikes larger than MaxSpikeNum: %d\n", MaxSpikeNum);
    *PoissonSpikeNum = MaxSpikeNum;
    return;
  }
  PoissonSpikeSourceIdx[pos] = i_source;
  PoissonSpikeConnIdx[pos] = i_conn;
  PoissonSpikeHeight[pos] = height;
  PoissonSpikeTargetNum[pos] = target_num;
}

__global__ void DeviceSpikeInit(int *spike_num, int *spike_source_idx,
				int *spike_conn_idx, float *spike_height,
				int *spike_target_num, int *poiss_spike_num,
				int *poiss_spike_source_idx,
				int *poiss_spike_conn_idx,
				float *poiss_spike_height,
				int *poiss_spike_target_num,
				int max_spike_num)
{
  SpikeNum = spike_num;
  SpikeSourceIdx = spike_source_idx;
  SpikeConnIdx = spike_conn_idx;
  SpikeHeight = spike_height;
  SpikeTargetNum = spike_target_num;

  PoissonSpikeNum = poiss_spike_num;
  PoissonSpikeSourceIdx = poiss_spike_source_idx;
  PoissonSpikeConnIdx = poiss_spike_conn_idx;
  PoissonSpikeHeight = poiss_spike_height;
  PoissonSpikeTargetNum = poiss_spike_target_num;

  MaxSpikeNum = max_spike_num;

  *SpikeNum = 0;
  *PoissonSpikeNum = 0;
}


void SpikeInit(int max_spike_num)
{
  //h_SpikeTargetNum = new int[PrefixScan::AllocSize];

  gpuErrchk(cudaMalloc(&d_SpikeNum, sizeof(int)));
  gpuErrchk(cudaMalloc(&d_SpikeSourceIdx, max_spike_num*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_SpikeConnIdx, max_spike_num*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_SpikeHeight, max_spike_num*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_SpikeTargetNum, max_spike_num*sizeof(int)));
  
  gpuErrchk(cudaMalloc(&d_PoissonSpikeNum, sizeof(int)));
  gpuErrchk(cudaMalloc(&d_PoissonSpikeSourceIdx, max_spike_num*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_PoissonSpikeConnIdx, max_spike_num*sizeof(int)));
  gpuErrchk(cudaMalloc(&d_PoissonSpikeHeight, max_spike_num*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_PoissonSpikeTargetNum, max_spike_num*sizeof(int)));
  
  //printf("here: SpikeTargetNum size: %d", max_spike_num);
  DeviceSpikeInit<<<1,1>>>(d_SpikeNum, d_SpikeSourceIdx, d_SpikeConnIdx,
			   d_SpikeHeight, d_SpikeTargetNum, d_PoissonSpikeNum,
			   d_PoissonSpikeSourceIdx, d_PoissonSpikeConnIdx,
			   d_PoissonSpikeHeight, d_PoissonSpikeTargetNum,
			   max_spike_num);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void SpikeReset()
{
  *SpikeNum = 0;
  *PoissonSpikeNum = 0;
}
