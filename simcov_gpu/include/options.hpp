#pragma once
#include <stdlib.h>
#include <string>

struct Options {
	int sample_rate;
	int max_iters;
	int sim_size;
};

Options parse_args(int argc, char** argv);
int check_args(Options opt);