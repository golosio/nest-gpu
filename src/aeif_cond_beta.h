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

#ifndef AEIFCONDBETAH
#define AEIFCONDBETAH

#include <iostream>
#include <string>
#include "cuda_error.h"
#include "rk5.h"
#include "node_group.h"
#include "base_neuron.h"
#include "neuron_models.h"

#define MAX_PORT_NUM 20

class aeif_cond_beta : public BaseNeuron
{
 public:
  RungeKutta5<RK5DataStruct> rk5_;
  float h_min_;
  float h_;
  RK5DataStruct rk5_data_struct_;
    
  int Init(int i_node_0, int n_neurons, int n_ports, int i_group);

  int Calibrate(float t_min);
		
  int Update(int it, float t1);
  
  int GetX(int i_neuron, int n_nodes, float *x) {
    return rk5_.GetX(i_neuron, n_nodes, x);
  }
  
  int GetY(int i_var, int i_neuron, int n_nodes, float *y) {
    return rk5_.GetY(i_var, i_neuron, n_nodes, y);
  }
  
  template<int N_PORTS>
    int UpdateNR(int it, float t1);

};

template <>
int aeif_cond_beta::UpdateNR<0>(int it, float t1);

template<int N_PORTS>
int aeif_cond_beta::UpdateNR(int it, float t1)
{
  if (N_PORTS == n_ports_) {
    const int NVAR = N_SCAL_VAR + N_VECT_VAR*N_PORTS;
    const int NPARAMS = N_SCAL_PARAMS + N_VECT_PARAMS*N_PORTS;

    rk5_.Update<NVAR, NPARAMS>(t1, h_min_, rk5_data_struct_);
  }
  else {
    UpdateNR<N_PORTS - 1>(it, t1);
  }

  return 0;
}

#endif
