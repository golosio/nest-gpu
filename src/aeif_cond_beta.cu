/*
Copyright (C) 2020 Bruno Golosio
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <math.h>
#include <iostream>
#include "rk5.h"
#include "aeif_cond_beta.h"
#include "aeif_cond_beta_variables.h"

int aeif_cond_beta::Init(int i_node_0, int n_nodes, int n_ports,
	       int i_group) {
  node_type_ = i_aeif_cond_beta_model;
  i_node_0_ = i_node_0;
  n_nodes_ = n_nodes;
  n_ports_ = n_ports;
  i_group_ = i_group;
  h_min_=1.0e-4;
  h_ = 1.0e-2;
  n_scal_var_ = N_SCAL_VAR;
  n_vect_var_ = N_VECT_VAR;
  n_scal_params_ = N_SCAL_PARAMS;
  n_vect_params_ = N_VECT_PARAMS;

  n_var_ = n_scal_var_ + n_vect_var_*n_ports;
  n_params_ = n_scal_params_ + n_vect_params_*n_ports;

  scal_var_name_ = aeif_cond_beta_scal_var_name;
  vect_var_name_= aeif_cond_beta_vect_var_name;
  scal_param_name_ = aeif_cond_beta_scal_param_name;
  vect_param_name_ = aeif_cond_beta_vect_param_name;
  rk5_data_struct_.node_type_ = i_aeif_cond_beta_model;
  rk5_data_struct_.i_node_0_ = i_node_0_;

  rk5_.Init(n_nodes, n_var_, n_params_, 0.0, h_, rk5_data_struct_);
  var_arr_ = rk5_.GetYArr();
  params_arr_ = rk5_.GetParamArr();

  port_weight_arr_ = GetParamArr() + n_scal_params_
    + GetVectParamIdx("g0");
  port_weight_arr_step_ = n_params_;
  port_weight_port_step_ = n_vect_params_;

  port_input_arr_ = GetVarArr() + n_scal_var_
    + GetVectVarIdx("g1");
  port_input_arr_step_ = n_var_;
  port_input_port_step_ = n_vect_var_;


  return 0;
}

int aeif_cond_beta::Calibrate(float t_min) {
  rk5_.Calibrate(t_min, h_, rk5_data_struct_);
  
  return 0;
}

template <>
int aeif_cond_beta::UpdateNR<0>(int it, float t1)
{
  return 0;
}

int aeif_cond_beta::Update(int it, float t1) {
  UpdateNR<MAX_PORT_NUM>(it, t1);

  return 0;
}

