#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

struct coord {
	int x,y,z;
};

inline int to_1d(int x, int y, int z,
					int sz_x, int sz_y, int sz_z) {
	return x + y * sz_x + z * sz_x * sz_y;
}

inline coord to_3d(int id, int sz_x, int sz_y, int sz_z) {
	int z = id / (sz_x * sz_y);
	int i = id % (sz_x * sz_y);
	int y = i / sz_x;
	int x = i % sz_x;
	coord c;
	c.x = x;
	c.y = y;
	c.z = z;
	return c;
}

struct GridPoint {
	//for active list data structure
	int idx = -1;

	//grid point data
	float virions;
	float nb_virions;
	float chemokine;
	float nb_chemokine;
	int num_neighbors;
	int epicell_status;
	int incubation_time_steps;
	int expressing_time_steps;
	int apoptotic_time_steps;

	//tcell data
	int id;
	int binding_period = -1;
	int tissue_time_steps = -1;
	bool moved = true;
	bool has_tcell = false;
};

enum class EpiCellStatus { HEALTHY = 0, INCUBATING = 1, EXPRESSING = 2, APOPTOTIC = 3, DEAD = 4 };

class Tissue {
	public:
		//sim data
		GridPoint* grid_points;
		int* active_grid_points;
		int num_active;
		int* new_active_grid_points;
		int new_num_active;

		//simulation world vars
		int size_x, size_y, size_z;
		int num_points;
		int num_circulating_tcells;
		int tcells_generated = 0;

		//virus params
		float diffusion;
		float infectivity;
		float virion_production;

		//chemokine params
		float chemokine_decay_rate;
		float min_chemokine;
		float chemokine_production;

		//epicell params
		int incubation_period;
		int expressing_period;
		int apoptosis_period;

		//tcell params
		int tcell_tissue_period;
		int tcell_binding_period;
		int tcell_vascular_period;
		int tcell_generation_rate;
		int tcell_initial_delay;
		double extravasate_fraction;
		double max_binding_prob;

		//constructors and deconstructors
		Tissue(int sz_x, int sz_y, int sz_z,
			int incb_period, int expr_period, int apop_period);
		~Tissue();

		//driver functions
		void dump_state(int iter);
		void update_virions();
		void update_epicells();
		void update_chemokine();
		void initialize_infection(std::vector<coord> coords,
							coord sim_size,
							std::vector<float> set_virions);
		void set_actives();
		//tcell funcs
		void update_tissue_tcell();
		void generate_tcells();
		void update_circulating_tcells();
		bool add_tcell_random();
		bool try_bind_tcell(int id);
};

inline void spread_virions(GridPoint* &grid_points,
							int id,
							float diffusion) {
	float v = grid_points[id].virions;
	float nb_v = grid_points[id].nb_virions;
	int num_nb = grid_points[id].num_neighbors;

	float virions_diffused = v*diffusion;
	float virions_left = v - virions_diffused;
	float avg_nb_virions = (virions_diffused + nb_v * diffusion)/(num_nb + 1);

	grid_points[id].virions = virions_left + avg_nb_virions;
	grid_points[id].nb_virions = 0.0f;

}

inline void accumulate_nb_virions(GridPoint* grid_points, int* active_grid_points, int num_active,
							int* new_active_grid_points, int &new_num_active,
							int id, int sz_x, int sz_y, int sz_z) {
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
							printf("### %d,%d\n", id, new_num_active);
						}
						if(grid_points[id].idx >= new_num_active) {
							grid_points[id].idx = new_num_active;
							new_active_grid_points[new_num_active] = id;
							new_num_active++;
							printf("### %d,%d\n", id, new_num_active);
						}

					}

				}
			}
		}
	}
}

inline void accumulate_nb_chemokine(GridPoint* grid_points, int* active_grid_points, int num_active,
							int* new_active_grid_points, int &new_num_active,
							int id, int sz_x, int sz_y, int sz_z) {
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
							printf("### %d,%d\n", id, new_num_active);
						}
						if(grid_points[id].idx >= new_num_active) {
							grid_points[id].idx = new_num_active;
							new_active_grid_points[new_num_active] = id;
							new_num_active++;
							printf("### %d,%d\n", id, new_num_active);
						}

					}

				}
			}
		}
	}
}

inline void spread_chemokine(GridPoint* grid_points,
							int id,
							float diffusion) {
	float c = grid_points[id].chemokine;
	float nb_c = grid_points[id].nb_chemokine;
	int num_nb = grid_points[id].num_neighbors;

	float chemokine_diffused = c*diffusion;
	float chemokine_left = c - chemokine_diffused;
	float avg_nb_chemokine = (chemokine_diffused + nb_c * diffusion)/(num_nb + 1);

	grid_points[id].chemokine = chemokine_left + avg_nb_chemokine;
	grid_points[id].nb_chemokine = 0.0f;

}