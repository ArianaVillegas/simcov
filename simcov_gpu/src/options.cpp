#include "options.hpp"

Options parse_args(int argc, char** argv) {
	Options o;

	//set defaults
	o.sample_rate = 1;
	o.max_iters = 1000;
	o.sim_size = 100;

	for(int i = 1; i < argc; i++) {
		std::string argument = argv[i];
		std::string indicator = "--";
		if(argument.find(indicator) != std::string::npos) {
			if(argument == "--sample_rate") {
				o.sample_rate = atoi(argv[i+1]);
			}
			if(argument == "--max_iters") {
				o.max_iters = atoi(argv[i+1]);
			}
			if(argument == "--sim_size") {
				o.sim_size = atoi(argv[i+1]);
			}
		}
	}
	return o;
}

int check_args(Options opt) {
	int result = 1;
	return result;
}