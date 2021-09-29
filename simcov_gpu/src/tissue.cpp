#include "tissue.hpp"
#include "utils.hpp"

Tissue::Tissue(int sz_x, int sz_y, int sz_z,
			int incb_period, int expr_period, int apop_period) {
	size_x = sz_x;
	size_y = sz_y;
	size_z = sz_z;
	num_points = size_x*size_y*size_z;

	grid_points = new GridPoint[num_points];
	active_grid_points = new int[num_points];

	incubation_period = incb_period;
	expressing_period = expr_period;
	apoptosis_period = apop_period;

	for(int i = 0; i < num_points; i++) {
		grid_points[i].virions = 0.0f;
		grid_points[i].nb_virions = 0.0f;
		grid_points[i].num_neighbors = 0;
		grid_points[i].epicell_status = 0;
		grid_points[i].incubation_time_steps = _rnd_gen->get_poisson(incubation_period);
		grid_points[i].expressing_time_steps = _rnd_gen->get_poisson(expressing_period);
		grid_points[i].apoptotic_time_steps = _rnd_gen->get_poisson(apoptosis_period);
		active_grid_points[i] = i; //start all grid points at active (is this correct?)
		grid_points[i].idx = i;
	}
	num_active = num_points;
}

Tissue::~Tissue(){
	delete[] grid_points;
	delete[] active_grid_points;
}

//CPU virion initialization
void Tissue::initialize_infection(std::vector<coord> coords,
							coord sim_size,
							std::vector<float> set_virions) {
	for(unsigned int i = 0; i < coords.size(); i++) {
		int id = to_1d( coords[i].x, coords[i].y, coords[i].z, sim_size.x, sim_size.y, sim_size.z );
		grid_points[id].virions = set_virions[i];
	}
}

void Tissue::dump_state(int iter) {
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
}

//CPU implementation
void Tissue::update_virions() {
	for(int  i = 0; i < num_active; i++) {
		spread_virions(grid_points,
					active_grid_points[i], diffusion);
	}
}

void Tissue::update_epicells() {
	for(int i = 0; i < num_active; i++) {
		bool produce_virions = false;
		int idx = active_grid_points[i];
		switch(grid_points[idx].epicell_status) {
			case 0:
				if(grid_points[idx].virions > 0) {
					if(_rnd_gen->trial_success(infectivity * grid_points[i].virions)) {
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
			grid_points[idx].chemokine = std::min((double)(grid_points[i].chemokine + chemokine_production), 1.0);
		}
	}
}

void Tissue::update_chemokine() {
	//decay
	for(int i = 0; i < num_active; i++) {
		int idx = active_grid_points[i];
		if(grid_points[idx].chemokine > 0) {
			grid_points[idx].chemokine *= (1.0 - chemokine_decay_rate);
			if(grid_points[idx].chemokine < min_chemokine) {
				grid_points[idx].chemokine = 0.0;
			}
		}
	}

	//spread
	for(int i = 0; i < num_active; i++) {
		spread_chemokine(grid_points,
							active_grid_points[i], diffusion);
	}
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
  		bool success = add_tcell_random();
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
	for(int j = 0; j < num_active; j++) {
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
								int nb_id = to_1d(xx,yy,zz,size_x,size_y,size_z);
								bool bound = try_bind_tcell(nb_id);
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
							if(_rnd_gen->trial_success((1.0/grid_points[i].num_neighbors))) {
								int nb_id = to_1d(xx,yy,zz,size_x,size_y,size_z);
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

	new_active_grid_points = new int[num_points];
	new_num_active = 0;
	for(int i = 0; i < num_active; i++) {
		int idx = active_grid_points[i];
		GridPoint gp = grid_points[idx];
		if((!(gp.epicell_status == 0 || gp.epicell_status == 4)) || (gp.virions > 0 || gp.chemokine > 0 || gp.has_tcell)) {
			new_active_grid_points[new_num_active] = idx;
			gp.idx = idx;
			new_num_active++;
		}
	}

	//accumulate steps to determine future actives
	for(int i = 0; i < num_active; i++) {
		accumulate_nb_virions(grid_points, active_grid_points, num_active,
								new_active_grid_points, new_num_active,
								active_grid_points[i], size_x, size_y, size_z);
		accumulate_nb_chemokine(grid_points, active_grid_points, num_active,
								new_active_grid_points, new_num_active,
								active_grid_points[i], size_x, size_y, size_z);
	}

	delete[] active_grid_points;
	active_grid_points = new_active_grid_points;
	num_active = new_num_active;
}