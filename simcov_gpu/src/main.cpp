#include "tissue.hpp"
#include "options.hpp"

int main(int argc, char** argv) {

	//params
	Options opt = parse_args(argc, argv);
	float diffusion = 0.1;
	float infectivity = 0.5;
	float virion_production = 100.0;
	float chemokine_decay_rate = 0.1;
	float min_chemokine = 0.00000001;
	float chemokine_production = 0.1;

	Tissue* tissue = new Tissue(opt.sim_size, opt.sim_size, 1, 10, 100, 10);
	tissue->diffusion = diffusion;
	tissue->infectivity = infectivity;
	tissue->virion_production = virion_production;
	tissue->chemokine_decay_rate = chemokine_decay_rate;
	tissue->min_chemokine = min_chemokine;
	tissue->chemokine_production = chemokine_production;

	tissue->tcell_tissue_period = 10;
	tissue->tcell_binding_period = 10;
	tissue->tcell_vascular_period = 10;
	tissue->tcell_generation_rate = 100000;
	tissue->extravasate_fraction = 0.0;
	tissue->max_binding_prob = 0.1;

	//set up initial infection
	coord init_infection;
	coord sim_size;
	init_infection.x = 49;
	init_infection.y = 49;
	init_infection.z = 0;
	sim_size.x = opt.sim_size;
	sim_size.y = opt.sim_size;
	sim_size.z = 1;
	std::vector<coord> coords;
	std::vector<float> virions;
	coords.push_back(init_infection);
	virions.push_back(10000.0f);
	tissue->initialize_infection(coords, sim_size, virions);

	//run sim
	for(int iter = 0; iter < opt.max_iters; iter++){
		tissue->set_actives();
		if(iter%opt.sample_rate == 0) {
			tissue->dump_state(iter);
		}
		if(iter > tissue->tcell_initial_delay) {
			tissue->generate_tcells();
		}
		tissue->update_circulating_tcells();
		tissue->update_tissue_tcell();
		tissue->update_chemokine();
		tissue->update_virions();
		tissue->update_epicells();
	}

	delete tissue;
	return 0;
}