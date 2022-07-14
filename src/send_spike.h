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





#ifndef SENDSPIKE_H
#define SENDSPIKE_H

extern int *d_SpikeNum;
extern int *d_SpikeSourceIdx;
extern int *d_SpikeConnIdx;
extern float *d_SpikeHeight;
extern int *d_SpikeTargetNum;

extern int *d_PoissonSpikeNum;
extern int *d_PoissonSpikeSourceIdx;
extern int *d_PoissonSpikeConnIdx;
extern float *d_PoissonSpikeHeight;
extern int *d_PoissonSpikeTargetNum;

extern __device__ int MaxSpikeNum;
extern __device__ int *SpikeNum;
extern __device__ int *SpikeSourceIdx;
extern __device__ int *SpikeConnIdx;
extern __device__ float *SpikeHeight;
extern __device__ int *SpikeTargetNum;

extern __device__ int *PoissonSpikeNum;
extern __device__ int *PoissonSpikeSourceIdx;
extern __device__ int *PoissonSpikeConnIdx;
extern __device__ float *PoissonSpikeHeight;
extern __device__ int *PoissonSpikeTargetNum;

__global__ void DeviceSpikeInit(int *spike_num, int *spike_source_idx,
				int *spike_conn_idx, float *spike_height,
				int *spike_target_num, int *poiss_spike_num,
				int *poiss_spike_source_idx,
				int *poiss_spike_conn_idx,
				float *poiss_spike_height,
				int *poiss_spike_target_num, int max_spike_num);

__device__ void SendSpike(int i_source, int i_conn, float height,
			  int target_num);

__device__ void PoissonSendSpike(int i_source, int i_conn, float height,
				 int target_num);

void SpikeInit(int max_spike_num);

__global__ void SpikeReset();

#endif
