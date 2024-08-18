/*
 *  mpi_comm.cu
 *
 *  This file is part of NEST GPU.
 *
 *  Copyright (C) 2021 The NEST Initiative
 *
 *  NEST GPU is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST GPU is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST GPU.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <config.h>

#include <list>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_error.h"
#include "getRealTime.h"
#include "nestgpu.h"

#include "mpi_comm.h"
#include "remote_connect.h"
#include "remote_spike.h"

#ifdef HAVE_MPI
#include <mpi.h>
MPI_Request* recv_mpi_request;
#endif

// Send spikes to remote MPI processes
int
NESTGPU::SendSpikeToRemote( int n_ext_spikes )
{
#ifdef HAVE_MPI
  uint *h_ExternalTargetSpikeNodeId = &h_ExternalTargetSpikeIdx0[ n_hosts_ + 1 ];
  int mpi_id, tag = 1; // id is already in the class, can be removed
  MPI_Comm_rank( MPI_COMM_WORLD, &mpi_id );

  double time_mark = getRealTime();

  // get point-to-point MPI communication activation matrix
  std::vector< std::vector < bool > > &p2p_host_conn_matrix = conn_->getP2PHostConnMatrix();
  
  gpuErrchk( cudaMemcpy(
    &h_ExternalTargetSpikeNum[0], d_ExternalTargetSpikeNum, n_hosts_ * sizeof( int ), cudaMemcpyDeviceToHost ) );
  SendSpikeToRemote_CUDAcp_time_ += ( getRealTime() - time_mark );

  time_mark = getRealTime();
  int n_spike_tot = 0;
  // copy spikes from GPU to CPU memory
  if ( n_ext_spikes > 0 )
  {
    gpuErrchk(
      cudaMemcpy( &n_spike_tot, d_ExternalTargetSpikeIdx0 + n_hosts_, sizeof( int ), cudaMemcpyDeviceToHost ) );
    if ( n_spike_tot >= max_remote_spike_num_ )
    {
      throw ngpu_exception( std::string( "Number of spikes to be sent remotely " ) + std::to_string( n_spike_tot )
        + " larger than limit " + std::to_string( max_remote_spike_num_ ) );
    }
    
    gpuErrchk( cudaMemcpy(
      h_ExternalTargetSpikeNodeId, d_ExternalTargetSpikeNodeId, n_spike_tot * sizeof( int ), cudaMemcpyDeviceToHost ) );
    gpuErrchk( cudaMemcpy( &h_ExternalTargetSpikeIdx0[0],
      d_ExternalTargetSpikeIdx0,
      ( n_hosts_ + 1 ) * sizeof( int ),
      cudaMemcpyDeviceToHost ) );
  }
  else
  {
    for ( int i = 0; i < n_hosts_ + 1; i++ )
    {
      h_ExternalTargetSpikeIdx0[ i ] = 0;
    }
  }
  
  // prepare array for sending spikes to host groups through MPI communicators
  int n_hg_spike_tot = 0;
  // copy spikes from GPU to CPU memory
  if ( n_ext_spikes > 0 ) {
    gpuErrchk( cudaMemcpy( &h_ExternalHostGroupSpikeIdx0[0], d_ExternalHostGroupSpikeIdx0, (conn_->getHostGroup().size() + 1)*sizeof(uint),
			   cudaMemcpyDeviceToHost));
    n_hg_spike_tot = h_ExternalHostGroupSpikeIdx0[conn_->getHostGroup().size()];

    if (n_hg_spike_tot > 0) {
      if ( n_hg_spike_tot >= max_remote_spike_num_ ) {
	throw ngpu_exception( std::string( "Number of spikes to be sent remotely to host groups " ) + std::to_string( n_hg_spike_tot )
			      + " larger than limit " + std::to_string( max_remote_spike_num_ ) );
      }
      gpuErrchk( cudaMemcpy(&h_ExternalHostGroupSpikeNodeId[0], d_ExternalHostGroupSpikeNodeId, n_hg_spike_tot*sizeof(int), cudaMemcpyDeviceToHost));
    }
  }
  else {
    for ( uint i=0; i<conn_->getHostGroup().size()+1; i++ ) {
      h_ExternalHostGroupSpikeIdx0[i] = 0;
    }
  }

  SendSpikeToRemote_CUDAcp_time_ += ( getRealTime() - time_mark );
  time_mark = getRealTime();

  if (true) {
    if (mpi_id !=0) { // send spikes to master
      MPI_Isend( &h_ExternalTargetSpikeIdx0[0], n_hosts_ + 1 + n_spike_tot, MPI_UNSIGNED, 0, tag, MPI_COMM_WORLD,
		 &recv_mpi_request[ 1 ] ); //&recv_mpi_request[ n_hosts_ + ih ] );
      // start nonblocking MPI receive from master
      MPI_Irecv( &h_ExternalSourceSpikeIdx0[0], n_hosts_ + 1 + max_spike_per_host_, MPI_UNSIGNED, 0, tag, MPI_COMM_WORLD,
		 &recv_mpi_request[ 0 ] );
      MPI_Status statuses[ 2 ];
      MPI_Waitall( 2, recv_mpi_request, statuses );
    }
    else { // master gathers all spikes
      // loop on remote MPI proc
      for ( int ish = 1; ish < n_hosts_; ish++ ) {
	// start nonblocking MPI receive from MPI proc ish
	MPI_Irecv( &h_ExternalTargetSpikeIdx0[ish*(n_hosts_ + 1 + max_spike_per_host_)], n_hosts_ + 1 + max_spike_per_host_,
		   MPI_UNSIGNED, ish, tag, MPI_COMM_WORLD, &recv_mpi_request[ ish - 1 ] );
      }
      MPI_Status statuses[ n_hosts_ - 1 ];
      MPI_Waitall( n_hosts_ - 1, recv_mpi_request, statuses );
      
      //// Master MPI process rearranges spikes
      /////////////////////
      // loop on remote MPI proc
      for ( int ith = 0; ith < n_hosts_; ith++ ) {
	uint *ext_src_spike_idx0 = &h_ExternalSourceSpikeIdx0[ith*(n_hosts_ + 1 + max_spike_per_host_)];
	uint *ext_src_spike_node_id = ext_src_spike_idx0 + n_hosts_ + 1;
	ext_src_spike_idx0[0] = 0;
	uint n_spike_tot = 0;
	for ( int ish = 0; ish < n_hosts_; ish++ ) {
	  uint *ext_tgt_spike_idx0 = &h_ExternalTargetSpikeIdx0[ish*(n_hosts_ + 1 + max_spike_per_host_)];
	  uint *ext_tgt_spike_node_id = ext_tgt_spike_idx0 + n_hosts_ + 1;
	  uint th_i0 = ext_tgt_spike_idx0[ith];
	  uint n_spikes = ext_tgt_spike_idx0[ith + 1] - th_i0; 
	  uint sh_i0 = n_spike_tot;
	  n_spike_tot += n_spikes;
	  ext_src_spike_idx0[ish + 1] = n_spike_tot;
	  for (uint i_spike = 0; i_spike<n_spikes; i_spike++) {
	    ext_src_spike_node_id[sh_i0 + i_spike] = ext_tgt_spike_node_id[th_i0 + i_spike]; 
	  }
	}
      }
      //////////////////////
      
      // master scatter all spikes
      // loop on remote MPI proc
      for ( int ith = 1; ith < n_hosts_; ith++ ) {
	uint *ext_src_spike_idx0 = &h_ExternalSourceSpikeIdx0[ith*(n_hosts_ + 1 + max_spike_per_host_)];
	uint n_spike_tot = ext_src_spike_idx0[n_hosts_];
	// start nonblocking MPI send to MPI proc ih
	MPI_Isend( &h_ExternalSourceSpikeIdx0[ith*(n_hosts_ + 1 + max_spike_per_host_)], n_hosts_ + 1 + n_spike_tot,
		   MPI_UNSIGNED, ith, tag, MPI_COMM_WORLD, &recv_mpi_request[ ith - 1 ] );
      }
      MPI_Waitall( n_hosts_ - 1, recv_mpi_request, statuses );

    }
    for ( int ih = 0; ih < n_hosts_; ih++ ) {
      int count = h_ExternalSourceSpikeIdx0[ih + 1] - h_ExternalSourceSpikeIdx0[ih];
      if (count < 0 || count > max_spike_per_host_) {
	throw ngpu_exception( std::string("Error in host ") + std::to_string(this_host_) +
				"\nNumber of spikes received remotely larger than limit " +
				std::to_string( max_spike_per_host_ ) +
				"\nYou can try to increase the kernel parameter \"max_spike_per_host_fact_\"." );
      }
      h_ExternalSourceSpikeNum[0][ ih ] = count;
    }

  }
  else {
    // loop on remote MPI proc
    for ( int ih = 0; ih < n_hosts_; ih++ ) {
      if (ih == mpi_id || p2p_host_conn_matrix[this_host_][ih]==false)
	{ // skip self MPI proc and unused point-to-point MPI communications
	  recv_mpi_request[ n_hosts_ + ih ] = MPI_REQUEST_NULL;
	  continue;
	}
      // get index and size of spike packet that must be sent to MPI proc ih
      // array_idx is the first index of the packet for host ih
      int array_idx = h_ExternalTargetSpikeIdx0[ ih ];
      int n_spikes = h_ExternalTargetSpikeIdx0[ ih + 1 ] - array_idx;
      // nonblocking sent of spike packet to MPI proc ih
      if (n_spikes >= max_spike_per_host_) {
	throw ngpu_exception( std::string("MPI_Isend error from host ") + std::to_string(this_host_) +
			      " to host " + std::to_string(ih) +
			      "\nNumber of spikes to be sent remotely " + std::to_string( n_spikes ) +
			      " larger than limit " + std::to_string( max_spike_per_host_ ) +
			      "\nYou can try to increase the kernel parameter \"max_spike_per_host_fact_\"." );
      }

      MPI_Isend( &h_ExternalTargetSpikeNodeId[ array_idx ], n_spikes, MPI_UNSIGNED, ih, tag, MPI_COMM_WORLD,
		 &recv_mpi_request[ n_hosts_ + ih ] );

      // printf("MPI_Send nspikes (src,tgt,nspike): "
      //	   "%d %d %d\n", mpi_id, ih, n_spikes);
      // printf("MPI_Send 1st-neuron-idx (src,tgt,idx): "
      //	   "%d %d %d\n", mpi_id, ih,
      //	   h_ExternalTargetSpikeNodeId[array_idx]);
    }
  }
  SendSpikeToRemote_comm_time_ += ( getRealTime() - time_mark );

  return 0;
#else
  throw ngpu_exception( "MPI is not available in your build" );
#endif
}

// Receive spikes from remote MPI processes
int
NESTGPU::RecvSpikeFromRemote()

{
#ifdef HAVE_MPI
  int mpi_id, tag = 1; // id is already in the class, can be removed
  MPI_Comm_rank( MPI_COMM_WORLD, &mpi_id );
  // get point-to-point MPI communication activation matrix
  std::vector< std::vector < bool > > &p2p_host_conn_matrix = conn_->getP2PHostConnMatrix();
  MPI_Status statuses[ 2*n_hosts_ ];
  
  double time_mark = getRealTime();

  if (!true) {
    // loop on remote MPI proc
    for ( int i_host = 0; i_host < n_hosts_; i_host++ )
      {
	if (i_host == mpi_id || p2p_host_conn_matrix[i_host][this_host_]==false)
	  {
	    recv_mpi_request[ i_host ] = MPI_REQUEST_NULL;
	    continue;
	  }
	// start nonblocking MPI receive from MPI proc i_host
	MPI_Irecv( &h_ExternalSourceSpikeNodeId[0][ i_host * max_spike_per_host_ ],
		   max_spike_per_host_,
		   MPI_UNSIGNED,
		   i_host,
		   tag,
		   MPI_COMM_WORLD,
		   &recv_mpi_request[ i_host ] );
      }
    
    //recv_mpi_request[ mpi_id ] = MPI_REQUEST_NULL;
    //MPI_Waitall( n_hosts_ + nhg - 1, recv_mpi_request, statuses );
    MPI_Waitall( 2*n_hosts_, recv_mpi_request, statuses );
  }
  
  std::vector< std::vector< int > > &host_group = conn_->getHostGroup();
  std::vector<MPI_Comm> &mpi_comm_vect = conn_->getMPIComm();
  uint nhg = host_group.size();
  std::vector<int> &host_group_local_id = conn_->getHostGroupLocalId();
  
  for (uint abs_ihg=0; abs_ihg<host_group_local_id.size(); abs_ihg++) {
    int ihg = host_group_local_id[abs_ihg];
    if (ihg < 0) {
      continue;
    }
    int idx0 = h_ExternalHostGroupSpikeIdx0[ihg]; // position of subarray of spikes that must be sent to host group ihg
    uint* sendbuf = &h_ExternalHostGroupSpikeNodeId[idx0]; // send address
    int sendcount = h_ExternalHostGroupSpikeNum[ihg]; // send count
    uint *recvbuf = &h_ExternalSourceSpikeNodeId[ihg][0]; //[ i_host * max_spike_per_host_ ] // receiving buffers
    int *recvcounts = &h_ExternalSourceSpikeNum[ihg][0];
    int *displs = &h_ExternalSourceSpikeDispl[0]; // displacememnts of receiving buffers, all equal to max_spike_per_host_

    MPI_Allgather(&sendcount, 1, MPI_INT, recvcounts, 1, MPI_INT, mpi_comm_vect[ihg-1]);
    
    MPI_Allgatherv(sendbuf, sendcount, MPI_INT, recvbuf, recvcounts, displs, MPI_INT, mpi_comm_vect[ihg-1]);

  }

  /* 
  MPI_Status statuses[ n_hosts_ + nhg - 1];
  recv_mpi_request[ mpi_id ] = MPI_REQUEST_NULL;
  //MPI_Waitall( n_hosts_ + nhg - 1, recv_mpi_request, statuses );
  MPI_Waitall( n_hosts_, recv_mpi_request, statuses );
  */
  if (!true) {
    for ( int i_host = 0; i_host < n_hosts_; i_host++ )
      {
	if ( ( int ) i_host == mpi_id )
	  {
	    h_ExternalSourceSpikeNum[0][ i_host ] = 0;
	    continue;
	  }
	int count = 0;
	if (p2p_host_conn_matrix[i_host][this_host_]==true) {
	  MPI_Get_count( &statuses[ i_host ], MPI_UNSIGNED, &count );
	}
	if (count < 0 || count > max_spike_per_host_) {
	  throw ngpu_exception( std::string("MPI_Irecv error in host ") + std::to_string(this_host_) +
				" from host " + std::to_string(i_host) +
				"\nNumber of spikes sent remotely larger than limit " +
				std::to_string( max_spike_per_host_ ) +
				"\nYou can try to increase the kernel parameter \"max_spike_per_host_fact_\"." );
	}
	h_ExternalSourceSpikeNum[0][ i_host ] = count;
      }
  }  
  // Maybe the barrier is not necessary?
  //MPI_Barrier( MPI_COMM_WORLD );
  RecvSpikeFromRemote_comm_time_ += ( getRealTime() - time_mark );
  
  return 0;
#else
  throw ngpu_exception( "MPI is not available in your build" );
#endif
}

int
NESTGPU::ConnectMpiInit( int argc, char* argv[] )
{
#ifdef HAVE_MPI
  CheckUncalibrated( "MPI connections cannot be initialized after calibration" );
  int initialized;
  MPI_Initialized( &initialized );
  if ( !initialized )
  {
    MPI_Init( &argc, &argv );
  }
  int n_hosts;
  int this_host;
  MPI_Comm_size( MPI_COMM_WORLD, &n_hosts );
  MPI_Comm_rank( MPI_COMM_WORLD, &this_host );
  mpi_flag_ = true;
  setNHosts( n_hosts );
  setThisHost( this_host );
  //conn_->remoteConnectionMapInit();
  recv_mpi_request = new MPI_Request[ 2*n_hosts_ ];

  return 0;
#else
  throw ngpu_exception( "MPI is not available in your build" );
#endif
}

int
NESTGPU::FakeConnectMpiInit(int n_hosts, int this_host)
{
  setNHosts( n_hosts );
  setThisHost( this_host );

  return 0;
}

int
NESTGPU::MpiFinalize()
{
#ifdef HAVE_MPI
  if ( mpi_flag_ )
  {
    int finalized;
    MPI_Finalized( &finalized );
    if ( !finalized )
    {
      MPI_Finalize();
    }
  }

  return 0;
#else
  throw ngpu_exception( "MPI is not available in your build" );
#endif
}
