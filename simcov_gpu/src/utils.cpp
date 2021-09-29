#include <chrono>
#include "utils.hpp"

unsigned rnd_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
std::shared_ptr<Random> _rnd_gen = std::make_shared<Random>(rnd_seed);