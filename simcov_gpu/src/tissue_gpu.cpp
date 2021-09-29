#include "tissue.hpp"
#include "utils.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>

curandState *d_state;
// GPU Kernel declarations
__global__ void gpu_set_single_virions(GridPoint* grid_points, int id, float num_virions);
__global__ void gpu_update_virions(int num_active, float diffusion, GridPoint* grid_points);
__global__ void gpu_udpate_epicells(int num_active,
									float infectivity, float virion_production,
									float chemokine_production,
									GridPoint* grid_points);
__global__ void gpu_update_chemokine(int num_active,
								float chemokine_decay_rate,
								float min_chemokine, GridPoint* grid_points);
__global__ void gpu_spr_chemokine(int num_active, float diffusion, GridPoint* grid_points);
__global__ void gpu_add_tcell(bool* success, int id, float min_chemokine,
								int life_time, int &new_num_active,
								int* new_active_grid_points, GridPoint* grid_points);
__global__ void gpu_update_tissue_tcell(int num_active, int &new_num_active,
								int* new_active_grid_points, int incubation_period, GridPoint* grid_points);
__global__ void gpu_set_actives(int &num_active, int &new_num_active,
										int* new_active_grid_points,
										int* active_grid_points, GridPoint* grid_points);
__global__ void gpu_accumulate_concentrations(GridPoint* grid_points,
								int* active_grid_points, int &num_active,
								int* new_active_grid_points, int &new_num_active,
								int size_x, int size_y, int size_z);

//GPU RNG kernels
__global__ void setup_curand(unsigned long long seed,
	unsigned long long offset,
	curandState *state);

//device functions
__device__ bool gpu_try_bind_tcell(int id, GridPoint* grid_points,
									int incubation_period);

Tissue::Tissue(int sz_x, int sz_y, int sz_z,
			int incb_period, int expr_period, int apop_period) {
	size_x = sz_x;
	size_y = sz_y;
	size_z = sz_z;
	num_points = size_x*size_y*size_z;

	cudaMalloc((void**)&grid_points, num_points*sizeof(GridPoint));
	cudaMalloc((void**)&active_grid_points, num_points*sizeof(int));

	incubation_period = incb_period;
	expressing_period = expr_period;
	apoptosis_period = apop_period;

	//initialize grid on CPU to copy over to GPU
	GridPoint* h_grid_points = new GridPoint[num_points];
	int* h_active_grid_points = new int[num_points];

	for(int i = 0; i < num_points; i++) {
		h_grid_points[i].virions = 0.0f;
		h_grid_points[i].nb_virions = 0.0f;
		h_grid_points[i].num_neighbors = 0;
		h_grid_points[i].epicell_status = 0;
		h_grid_points[i].incubation_time_steps = _rnd_gen->get_poisson(incubation_period);
		h_grid_points[i].expressing_time_steps = _rnd_gen->get_poisson(expressing_period);
		h_grid_points[i].apoptotic_time_steps = _rnd_gen->get_poisson(apoptosis_period);
		h_active_grid_points[i] = i; //start all grid points at active (is this correct?)
		h_grid_points[i].idx = i;
	}

	// Copy over to device
	cudaMemcpy(h_grid_points,
				grid_points,
				num_points*sizeof(GridPoint),
				cudaMemcpyHostToDevice);
	cudaMemcpy(h_active_grid_points,
				active_grid_points,
				num_points*sizeof(int),
				cudaMemcpyHostToDevice);

	delete[] h_grid_points;
	delete[] h_active_grid_points;

	num_active = num_points;

	//set up RNG
	cudaMalloc(&d_state, sizeof(curandState));
	setup_curand<<<1,1>>>(1,0,d_state); //TODO: Put in option for seed

}

Tissue::~Tissue(){
	cudaFree(grid_points);
	cudaFree(active_grid_points);
}

//virion initialization (CPU -> GPU)
//These are set in serial by launching multiple GPU
//kernels. Shouldn't be too costly for a one time execution
void Tissue::initialize_infection(std::vector<coord> coords,
							coord sim_size,
							std::vector<float> set_virions) {
	for(unsigned int i = 0; i < coords.size(); i++) {
		int id = to_1d( coords[i].x, coords[i].y, coords[i].z, sim_size.x, sim_size.y, sim_size.z );
		gpu_set_single_virions<<<1,1>>>>(grid_points, id, set_virions[i]);
	}
}

void Tissue::dump_state(int iter) {
	GridPoint* h_grid_points = new GridPoint[num_points];
	cudaMemcpy(grid_points,
				h_grid_points,
				num_points*sizeof(GridPoint),
				cudaMemcpyDeviceToHost);
	for(int i = 0; i < num_points; i++) {
		coord c = to_3d(i, size_x, size_y, size_z);
		int tcell = 0;
		if(grid_points[i].has_tcell) {
			tcell = 1;
		} else {
			tcell = 0;
		}
		printf("%d,%d,%d,%d,%d,%f,%f,%d,%d,%d\n",iter, i, 
										c.x, c.y, c.z,
										grid_points[i].virions,
										grid_points[i].chemokine,
										grid_points[i].epicell_status,
										tcell, grid_points[i].tissue_time_steps);
	}
	delete[] h_grid_points;
}

//GPU implementation
void Tissue::update_virions() {

	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
	gpu_update_virions<<<32*numSMs, 256>>>(num_active, diffusion, grid_points);
}

void Tissue::update_epicells() {
	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
	gpu_udpate_epicells<<<32*numSMs, 256>>>(num_active,
									infectivity, virion_production,
									chemokine_production, GridPoint* grid_points);
}

void Tissue::update_chemokine() {
	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDeviceGetAttribute, 0);
	gpu_update_chemokine<<<32*numSMs, 256>>>(num_active,
								chemokine_decay_rate,
								min_chemokine, grid_points);
	gpu_spr_chemokine<<<32*numSMs, 256>>>(num_active, diffusion, grid_points);
}

//Tcell functions
void Tissue::generate_tcells() {
	num_circulating_tcells += tcell_generation_rate;
	if (num_circulating_tcells < 0) num_circulating_tcells = 0;
}

void Tissue::update_circulating_tcells() {
	double portion_dying = (double)num_circulating_tcells/tcell_vascular_period;
	int num_dying = floor(portion_dying);
	if(_rnd_gen->trial_success(portion_dying - num_dying)) num_dying++;
	num_circulating_tcells -= num_dying;
	if (num_circulating_tcells < 0) num_circulating_tcells = 0;
	double portion_xtravasing = extravasate_fraction * num_circulating_tcells;
  	int num_xtravasing = floor(portion_xtravasing);
  	if (_rnd_gen->trial_success(portion_xtravasing - num_xtravasing)) num_xtravasing++;
  	for(int i = 0; i <  num_xtravasing; i++) {
  		int x = _rnd_gen->get(0, size_x);
		int y = _rnd_gen->get(0, size_y);
		int z = _rnd_gen->get(0, size_z);
		int id = to_1d(x,y,z,size_x,size_y,size_z);
		int life_time = _rnd_gen->get_poisson(tcell_tissue_period);
		bool success;
		bool* d_success;
		cudaMalloc(&d_success, sizeof(bool));
  		gpu_add_tcell<<<1,1>>>(d_success, id,
  							min_chemokine,
  							life_time,
  							new_num_active,
  							new_active_grid_points,
  							grid_points);
  		cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost);
  		if(success) {
  			num_circulating_tcells--;
  		}
  	}
}

bool Tissue::add_tcell_random() {
	int x = _rnd_gen->get(0, size_x);
	int y = _rnd_gen->get(0, size_y);
	int z = _rnd_gen->get(0, size_z);
	int id = to_1d(x,y,z,size_x,size_y,size_z);
	if(grid_points[id].has_tcell) return false;
	if(grid_points[id].chemokine < min_chemokine) return false;
	int tcell_id = tcells_generated;
	tcells_generated += 1;
	grid_points[id].id = tcell_id;
	grid_points[id].moved = true;
	grid_points[id].has_tcell = true;
	grid_points[id].tissue_time_steps = _rnd_gen->get_poisson(tcell_tissue_period);

	if(grid_points[id].idx >= new_num_active) {
		grid_points[id].idx = new_num_active;
		new_active_grid_points[new_num_active] = id;
		new_num_active++;
	}

	return true;
}

void Tissue::update_tissue_tcell() {
	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDeviceGetAttribute, 0);
	gpu_update_tissue_tcell<<<32*numSMs, 256>>>(num_active,
									new_num_active,
									new_active_grid_points,
									incubation_period,
									grid_points);
}

bool Tissue::try_bind_tcell(int id) {
	int status = grid_points[id].epicell_status;
	if(status == 0 || status == 4) {
		return false;
	}
	double binding_prob = 0.0;
	if(status == 2 || status == 3) {
		binding_prob = 1.0;
	} else {
		double scaling = 1.0 - (double)grid_points[id].incubation_time_steps/incubation_period;
		if(scaling < 0) scaling = 0;
		double prob = max_binding_prob*scaling;
		binding_prob = std::min(prob, max_binding_prob);
	}
	if(_rnd_gen->trial_success(binding_prob)) {
		grid_points[id].epicell_status = 3;
		return true;
	}
	return false;
}

//sets actives for next time step
void Tissue::set_actives() {
	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDeviceGetAttribute, 0);
	cudaMalloc((void**)&new_active_grid_points, num_points*sizeof(int));
	gpu_set_actives<<<32*numSMs, 256>>>(num_active, new_num_active, new_active_grid_points,
										active_grid_points, grid_points);
	
	//accumulate steps to determine future actives
	gpu_accumulate_concentrations<<<32*numSMs, 256>>>(GridPoint* grid_points,
								int* active_grid_points, int &num_active,
								int* new_active_grid_points, int &new_num_active,
								int size_x, int size_y, int size_z)
	cudaFree(active_grid_points);
	active_grid_points = new_active_grid_points;
	num_active = new_num_active;
}

__device__ boold gpu_try_bind_tcell(int id, GridPoint* grid_points,
									int incubation_period){
	int status = grid_points[id].epicell_status;
	if(status == 0 || status == 4) {
		return false;
	}
	double binding_prob = 0.0;
	if(status == 2 || status == 3) {
		binding_prob = 1.0;
	} else {
		double scaling = 1.0 - (double)grid_points[id].incubation_time_steps/incubation_period;
		if(scaling < 0) scaling = 0;
		double prob = max_binding_prob*scaling;
		if(prob < max_binding_prob) {
			binding_prob = prob;
		} else {
			binding_prob = max_binding_prob;
		}
	}
	float roll = curand_uniform(d_state + start);
	if(roll <  binding_prob) {
		grid_points[id].epicell_status = 3;
		return true;
	}
	return false;
}

// GPU Kernel definitions
__global__ void gpu_set_single_virions(GridPoint* grid_points, int id, float num_virions){
	grid_points[id] = num_virions;
}

__global__ void gpu_update_virions(int num_active, float diffusion,
									GridPoint* grid_points) {
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = num_active;
	int step = blockDim.x*griddim.x;
	for(int i = start; i < maximum; i += step) {
		float v = grid_points[i].virions;
		float nb_v = grid_points[i].nb_virions;
		int num_nb = grid_points[i].num_neighbors;

		float virions_diffused = v*diffusion;
		float virions_left = v - virions_diffused;
		float avg_nb_virions = (virions_diffused + nb_v * diffusion)/(num_nb + 1);

		grid_points[i].virions = virions_left + avg_nb_virions;
		grid_points[i].nb_virions = 0.0f;
	}
}

__global__ void gpu_udpate_epicells(int num_active,
									float infectivity, float virion_production,
									float chemokine_production,
									GridPoint* grid_points) {
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = num_active;
	int step = blockDim.x*griddim.x;
	float roll;
	for(int i = start; i < maximum; i += step) {
		bool produce_virions = false;
		int idx = active_grid_points[i];
		switch(grid_points[idx].epicell_status) {
			case 0:
				if(grid_points[idx].virions > 0) {
					roll = curand_uniform(d_state + start);
					if(roll < (infectivity * grid_points[i].virions)) {
						grid_points[idx].epicell_status = 1;
					}
				}
				break;
			case 1:
				grid_points[idx].incubation_time_steps--;
				if(grid_points[idx].incubation_time_steps <= 0) {
					grid_points[idx].epicell_status = 2;
				}
				break;
			case 2:
				grid_points[idx].expressing_time_steps--;
				if(grid_points[idx].expressing_time_steps <= 0) {
					grid_points[idx].epicell_status = 4;
				} else {
					produce_virions = true;
				}
				break;
			case 3:
				grid_points[idx].apoptotic_time_steps--;
				if(grid_points[idx].apoptotic_time_steps <= 0) {
					grid_points[idx].epicell_status = 4;
				} else {
					produce_virions = true;
				}
				break;
			default: break;
		}
		if(produce_virions) {
			grid_points[idx].virions += virion_production;
			if((grid_points[i].chemokine + chemokine_production) < 1.0) {
				grid_points[idx].chemokine = (grid_points[i].chemokine + chemokine_production);
			} else {
				grid_points[idx].chemokine = 1.0;
			}
		}
	}
}

__global__ void gpu_update_chemokine(int num_active,
								float chemokine_decay_rate,
								float min_chemokine, GridPoint* grid_points) {
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = num_active;
	int step = blockDim.x*griddim.x;
	//decay
	for(int i = start; i < num_active; i += step) {
		int idx = active_grid_points[i];
		if(grid_points[idx].chemokine > 0) {
			grid_points[idx].chemokine *= (1.0 - chemokine_decay_rate);
			if(grid_points[idx].chemokine < min_chemokine) {
				grid_points[idx].chemokine = 0.0;
			}
		}
	}
}

__global__ void gpu_spr_chemokine(int num_active, float diffusion, GridPoint* grid_points) {
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = num_active;
	int step = blockDim.x*griddim.x;
	//spread
	for(int i = start; i < num_active; i += step) {
		float c = grid_points[i].chemokine;
		float nb_c = grid_points[i].nb_chemokine;
		int num_nb = grid_points[i].num_neighbors;

		float chemokine_diffused = c*diffusion;
		float chemokine_left = c - chemokine_diffused;
		float avg_nb_chemokine = (chemokine_diffused + nb_c * diffusion)/(num_nb + 1);

		grid_points[i].chemokine = chemokine_left + avg_nb_chemokine;
		grid_points[i].nb_chemokine = 0.0f;
	}
}

__global__ void gpu_add_tcell(bool* success, int id, float min_chemokine,
								int life_time, int &new_num_active,
								int* new_active_grid_points, GridPoint* grid_points) {
	*success = false;
	if(grid_points[id].has_tcell) return;
	if(grid_points[id].chemokine < min_chemokine) return;
	int tcell_id = tcells_generated;
	tcells_generated += 1;
	grid_points[id].id = tcell_id;
	grid_points[id].moved = true;
	grid_points[id].has_tcell = true;
	grid_points[id].tissue_time_steps = life_time;

	if(grid_points[id].idx >= new_num_active) {
		grid_points[id].idx = new_num_active;
		new_active_grid_points[new_num_active] = id;
		new_num_active++;
	}

	*success = true;
}

__global__ void gpu_update_tissue_tcell(int num_active, int &new_num_active,
								int* new_active_grid_points, int incubation_period, GridPoint* grid_points) {
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = num_active;
	int step = blockDim.x*griddim.x;
	for(int j = start; j < maximum; j += step) {
		int i = active_grid_points[j];
		if(grid_points[i].has_tcell) {
			if(grid_points[i].moved) {
				grid_points[i].moved = false;
				continue;
			}
		} else {
			continue;
		}
		grid_points[i].tissue_time_steps--;
		if(grid_points[i].tissue_time_steps <= 0) {
			grid_points[i].has_tcell = false;
			continue;
		}
		if(grid_points[i].binding_period != -1) {
			grid_points[i].binding_period--;
			if(grid_points[i].binding_period == 0) grid_points[i].binding_period = -1;
		} else {

			//TODO: Fix this to match random behavior of simcov
			for(int x = -1; x <= 1; x++) {
				for(int y = -1; y <= 1; y++) {
					for(int z = -1; z <= 1; z++) {
						int zz = i / (size_x * size_y);
						int id = i % (size_x * size_y);
						int yy = id / size_x;
						int xx = id % size_x;

						xx += x;
						yy += y;
						zz += z;

						if(xx >= 0 && xx < size_x &&
							yy >= 0 && yy < size_y &&
							zz >= 0 && zz < size_z) {
								int nb_id = xx + yy * size_x + zz * size_x * size_y;
								bool bound = gpu_try_bind_tcell(nb_id, grid_points, incubation_period);
								if(bound) {
									grid_points[i].binding_period = tcell_binding_period;
								}
						}
					}
				}
			}
		}
		if(grid_points[i].binding_period != -1) {
			//TODO: Fix this to match random behavior of simcov
			bool moved = false;
			for(int x = -1; x <= 1; x++) {
				if(moved) break;
				for(int y = -1; y <= 1; y++) {
					if(moved) break;
					for(int z = -1; z <= 1; z++) {
						if(moved) break;
						int zz = i / (size_x * size_y);
						int id = i % (size_x * size_y);
						int yy = id / size_x;
						int xx = id % size_x;

						xx += x;
						yy += y;
						zz += z;

						if(xx >= 0 && xx < size_x &&
							yy >= 0 && yy < size_y &&
							zz >= 0 && zz < size_z) {
							float roll = curand_uniform(d_state);
							if(roll < (1.0/grid_points[i].num_neighbors)) {
								int nb_id = xx + yy * size_x + zz * size_x * size_y;
								if(!grid_points[nb_id].has_tcell) {
									moved = true;
									grid_points[nb_id].has_tcell = true;
									grid_points[nb_id].binding_period = grid_points[i].binding_period;
									grid_points[nb_id].tissue_time_steps = grid_points[i].tissue_time_steps;
									grid_points[nb_id].moved = false;
									grid_points[i].has_tcell = false;
									grid_points[i].binding_period = -1;
									grid_points[i].tissue_time_steps = -1;
									grid_points[i].moved = false;

									if(grid_points[nb_id].idx >= new_num_active) {
										grid_points[nb_id].idx = new_num_active;
										new_active_grid_points[new_num_active] = nb_id;
										new_num_active++;
									}
									if(grid_points[i].idx >= new_num_active) {
										grid_points[i].idx = new_num_active;
										new_active_grid_points[new_num_active] = i;
										new_num_active++;
									}

									break;
								}
							}
						}
					}
				}
			}
		}
	}
}

__global__ void gpu_set_actives(int &num_active, int &new_num_active,
										int* new_active_grid_points,
										int* active_grid_points, GridPoint* grid_points) {
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = num_active;
	int step = blockDim.x*griddim.x;
	for(int i = start; i < maximum; i += step) {
		int idx = active_grid_points[i];
		GridPoint gp = grid_points[idx];
		if((!(gp.epicell_status == 0 || gp.epicell_status == 4)) || (gp.virions > 0 || gp.chemokine > 0 || gp.has_tcell)) {
			new_active_grid_points[new_num_active] = idx;
			gp.idx = idx;
			new_num_active++;
		}
	}
}

__global__ void gpu_accumulate_concentrations(GridPoint* grid_points,
								int* active_grid_points, int &num_active,
								int* new_active_grid_points, int &new_num_active,
								int size_x, int size_y, int size_z) {
	int start = blockIdx.x*blockDim.x + threadIdx.x;
	int maximum = num_active;
	int step = blockDim.x*griddim.x;
	for(int id = start; id < maximum; id += step) {
		grid_points[id].num_neighbors = 0;
		grid_points[id].nb_virions = 0.0f;
		for(int x = -1; x <= 1; x++) {
			for(int y = -1; y <= 1; y++) {
				for(int z = -1; z <= 1; z++) {
					if(!(x==0 && y==0 && z==0)) {

						int zz = id / (sz_x * sz_y);
						int i = id % (sz_x * sz_y);
						int yy = i / sz_x;
						int xx = i % sz_x;

						xx += x;
						yy += y;
						zz += z;

						if(xx >= 0 && xx < sz_x &&
							yy >= 0 && yy < sz_y &&
							zz >= 0 && zz < sz_z) {
							int id_nb = to_1d(xx,yy,zz,sz_x,sz_y,sz_z);
							grid_points[id].nb_virions += grid_points[id_nb].virions;
							grid_points[id].num_neighbors += 1;

							if(grid_points[id_nb].idx >= new_num_active) {
								grid_points[id_nb].idx = new_num_active;
								new_active_grid_points[new_num_active] = id_nb;
								new_num_active++;
							}
							if(grid_points[id].idx >= new_num_active) {
								grid_points[id].idx = new_num_active;
								new_active_grid_points[new_num_active] = id;
								new_num_active++;
							}

						}

					}
				}
			}
		}
	}
	for(int id = start; id < maximum; id += step) {
		grid_points[id].num_neighbors = 0;
		grid_points[id].nb_chemokine = 0.0f;
		for(int x = -1; x <= 1; x++) {
			for(int y = -1; y <= 1; y++) {
				for(int z = -1; z <= 1; z++) {
					if(!(x==0 && y==0 && z==0)) {

						int zz = id / (sz_x * sz_y);
						int i = id % (sz_x * sz_y);
						int yy = i / sz_x;
						int xx = i % sz_x;

						xx += x;
						yy += y;
						zz += z;

						if(xx >= 0 && xx < sz_x &&
							yy >= 0 && yy < sz_y &&
							zz >= 0 && zz < sz_z) {
							int id_nb = to_1d(xx,yy,zz,sz_x,sz_y,sz_z);
							grid_points[id].nb_chemokine += grid_points[id_nb].chemokine;
							grid_points[id].num_neighbors += 1;

							if(grid_points[id_nb].idx >= new_num_active) {
								grid_points[id_nb].idx = new_num_active;
								new_active_grid_points[new_num_active] = id_nb;
								new_num_active++;
							}
							if(grid_points[id].idx >= new_num_active) {
								grid_points[id].idx = new_num_active;
								new_active_grid_points[new_num_active] = id;
								new_num_active++;
							}

						}

					}
				}
			}
		}
	}
}

//curand kernels
__global__ void setup_curand(unsigned long long seed,
	unsigned long long offset,
	curandState *state){
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	curand_init(seed, idx, offset, state);
}