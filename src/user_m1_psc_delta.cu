/*
 *  user_m1_psc_delta.cu
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


#include "rk5.h"
#include "user_m1.h"
#include "user_m1_kernel.h"
#include <cmath>
#include <config.h>
#include <iostream>

namespace user_m1_ns
{

__device__ void
NodeInit( int n_var, int n_param, double x, float* y, float* param, user_m1_rk5 data_struct )
{
  // int array_idx = threadIdx.x + blockIdx.x * blockDim.x;

  V_th = -50.4;
  Delta_T = 2.0;
  g_L = 30.0;
  E_L = -70.6;
  C_m = 281.0;
  a = 4.0;
  b = 80.5;
  tau_w = 144.0;
  I_e = 0.0;
  V_peak = 0.0;
  V_reset = -60.0;
  t_ref = 0.0;
  den_delay = 0.0;

  V_m = E_L;
  w = 0;
  refractory_step = 0;
}

__device__ void
NodeCalibrate( int n_var, int n_param, double x, float* y, float* param, user_m1_rk5 data_struct )
{
  // int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  // int n_port = (n_var-N_SCAL_VAR)/N_PORT_VAR;

  refractory_step = 0;
  // set the right threshold depending on Delta_T
  if ( Delta_T <= 0.0 )
  {
    V_peak = V_th; // same as IAF dynamics for spikes if Delta_T == 0.
  }
}

}

__device__ void
NodeInit( int n_var, int n_param, double x, float* y, float* param, user_m1_rk5 data_struct )
{
  user_m1_ns::NodeInit( n_var, n_param, x, y, param, data_struct );
}

__device__ void
NodeCalibrate( int n_var, int n_param, double x, float* y, float* param, user_m1_rk5 data_struct )

{
  user_m1_ns::NodeCalibrate( n_var, n_param, x, y, param, data_struct );
}

using namespace user_m1_ns;

int
user_m1::Init( int i_node_0, int n_node, int n_port, int i_group, unsigned long long* seed )
{
  BaseNeuron::Init( i_node_0, n_node, n_port, i_group, seed );
  node_type_ = i_user_m1_model;
  n_scal_var_ = N_SCAL_VAR;
  n_scal_param_ = N_SCAL_PARAM;
  n_group_param_ = N_GROUP_PARAM;

  n_var_ = n_scal_var_ + n_port_var_ * n_port;
  n_param_ = n_scal_param_ + n_port_param_ * n_port;

  group_param_ = new float[ N_GROUP_PARAM ];

  scal_var_name_ = user_m1_scal_var_name;
  scal_param_name_ = user_m1_scal_param_name;
  group_param_name_ = user_m1_group_param_name;
  // rk5_data_struct_.node_type_ = i_user_m1_model;
  rk5_data_struct_.i_node_0_ = i_node_0_;

  SetGroupParam( "h_min_rel", 1.0e-3 );
  SetGroupParam( "h0_rel", 1.0e-2 );
  h_ = h0_rel_ * 0.1;

  rk5_.Init( n_node, n_var_, n_param_, 0.0, h_, rk5_data_struct_ );
  var_arr_ = rk5_.GetYArr();
  param_arr_ = rk5_.GetParamArr();

  // multiplication factor of input signal is always 1 for all nodes
  float input_weight = 1.0;
  gpuErrchk( cudaMalloc( &port_weight_arr_, sizeof( float ) ) );
  gpuErrchk( cudaMemcpy( port_weight_arr_, &input_weight, sizeof( float ), cudaMemcpyHostToDevice ) );
  port_weight_arr_step_ = 0;
  port_weight_port_step_ = 0;

  port_input_arr_ = GetVarArr() + GetScalVarIdx( "V_m" );
  port_input_arr_step_ = n_var_;
  port_input_port_step_ = n_port_var_;
  den_delay_arr_ = GetParamArr() + GetScalParamIdx( "den_delay" );

  return 0;
}

int
user_m1::Calibrate( double time_min, float time_resolution )
{
  h_min_ = h_min_rel_ * time_resolution;
  h_ = h0_rel_ * time_resolution;
  rk5_.Calibrate( time_min, h_, rk5_data_struct_ );

  return 0;
}

int
user_m1::Update( long long it, double t1 )
{
  rk5_.Update< N_SCAL_VAR, N_SCAL_PARAM >( t1, h_min_, rk5_data_struct_ );

  return 0;
}
