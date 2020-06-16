#pragma once

#include <stdarg.h>
#include <math.h>
#include <iostream>
#include <map>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <upcxx/upcxx.hpp>

#include "upcxx_utils/log.hpp"
#include "upcxx_utils/progress_bar.hpp"
#include "upcxx_utils/flat_aggr_store.hpp"
#include "upcxx_utils/timers.hpp"

#include "utils.hpp"


using upcxx::rank_me;
using upcxx::rank_n;

struct GridCoords {
  int64_t x, y, z;

  GridCoords() {}

  GridCoords(int64_t x, int64_t y, int64_t z) : x(x), y(y), z(z) {}

  // create a grid point from 1d
  GridCoords(int64_t i, const GridCoords &grid_size) {
    z = i / (grid_size.x * grid_size.y);
    i = i % (grid_size.x * grid_size.y);
    y = i / grid_size.x;
    x = i % grid_size.x;
  }

  // create a random grid point
  GridCoords(Random &rnd_gen, const GridCoords &grid_size) {
    x = rnd_gen.get(0, grid_size.x);
    y = rnd_gen.get(0, grid_size.y);
    z = rnd_gen.get(0, grid_size.z);
  }

  bool operator==(const GridCoords &coords) {
    return x == coords.x && y == coords.y && z == coords.z;
  }

  bool operator!=(const GridCoords &coords) {
    return x != coords.x || y != coords.y || z != coords.z;
  }

  int64_t to_1d(const GridCoords &grid_size) const {
    return x + y * grid_size.x + z * grid_size.x * grid_size.y;
  }

  string str() {
    return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
  }
};

struct TCell {
  int64_t id;
};

enum class EpiCellStatus { Healthy, Incubating, Dead };
enum class InfectionResult { Success, AlreadyInfected, NotHealthy };

struct EpiCell {
  int64_t id;
  EpiCellStatus status = EpiCellStatus::Healthy;
  int64_t num_steps_infected;

  string str() {
    return std::to_string(id);
  }
};

struct GridPoint {
  int64_t id;
  GridCoords coords;
  // empty space is nullptr
  EpiCell *epicell = nullptr;
  // there are two vectors of tcells, one of which contains tcells for the current round,
  // and one of which contains tcells for the next round
  vector<TCell> tcells_backing_1 = {};
  vector<TCell> tcells_backing_2 = {};
  // a pointer to the currently active tcells vector
  vector<TCell> *tcells = nullptr;
  bool virus = false;

  string str() {
    ostringstream oss;
    oss << id << " " << coords.str() << " " << epicell->str() << " " << virus;
    return oss.str();
  }
};

class Tissue {
 private:
  // these are static so they don't have to be passed in RPCs
  static int block_size;

  using grid_points_t = upcxx::dist_object<vector<GridPoint>>;
  grid_points_t grid_points;

  vector<GridPoint>::iterator grid_point_iter;

  // TODO: maintain a list of active cells (i.e. those that need to be updated, e.g because of cytokines, viruses, T-cells).
  // This will reduce computation by avoiding having to iterate through the whole space.

  intrank_t get_rank_for_grid_point(const GridCoords &coords) {
    int64_t id = coords.to_1d(Tissue::grid_size);
    int64_t block_i = id / Tissue::block_size;
    return block_i % rank_n();
  }

  static GridPoint &get_local_grid_point(grid_points_t &grid_points, int64_t id, const GridCoords &coords) {
    int64_t block_i = id / Tissue::block_size / rank_n();
    int64_t i = id % Tissue::block_size + block_i * Tissue::block_size;
    assert(i < grid_points->size());
    GridPoint &grid_point = (*grid_points)[i];
    assert(grid_point.id == id);
    assert(grid_point.coords.x == coords.x);
    assert(grid_point.coords.y == coords.y);
    assert(grid_point.coords.z == coords.z);
    return grid_point;
  }

 public:
  static GridCoords grid_size;

  Tissue() : grid_points({}) {}

  ~Tissue() {}

  int64_t get_num_grid_points() {
    return Tissue::grid_size.x * Tissue::grid_size.y * Tissue::grid_size.z;
  }

  upcxx::future<InfectionResult> infect_epicell(GridCoords coords) {
    int64_t id = coords.to_1d(Tissue::grid_size);
    return upcxx::rpc(
        get_rank_for_grid_point(coords),
        [](grid_points_t &grid_points, int64_t id, GridCoords coords) {
          GridPoint &grid_point = Tissue::get_local_grid_point(grid_points, id, coords);
          if (grid_point.virus) return InfectionResult::AlreadyInfected;
          if (grid_point.epicell->status != EpiCellStatus::Healthy) return InfectionResult::NotHealthy;
          grid_point.virus = true;
          grid_point.epicell->num_steps_infected = 0;
          grid_point.epicell->status = EpiCellStatus::Incubating;
          return InfectionResult::Success;
        },
        grid_points, id, coords);
  }

  upcxx::future<> add_tcell(GridCoords coords, int64_t tcell_id) {
    // FIXME: this should be an aggregating store
    int64_t id = coords.to_1d(Tissue::grid_size);
    return upcxx::rpc(
        get_rank_for_grid_point(coords),
        [](grid_points_t &grid_points, int64_t id, GridCoords coords, int64_t tcell_id) {
          GridPoint &grid_point = Tissue::get_local_grid_point(grid_points, id, coords);
          if (grid_point.tcells == nullptr) grid_point.tcells = &grid_point.tcells_backing_1;
          // add the tcell to the future tcells vector, i.e. not the one pointed to by tcells
          auto next_tcells =
              (grid_point.tcells == &grid_point.tcells_backing_1 ? &grid_point.tcells_backing_2 : &grid_point.tcells_backing_1);
          next_tcells->push_back({.id = tcell_id});
        },
        grid_points, id, coords, tcell_id);
  }

  int64_t update_tcells(bool check_switch=false) {
    auto switch_tcells_vectors = [](vector<TCell> *tcells_backing_new, vector<TCell> *tcells_backing_old) -> vector<TCell>* {
      tcells_backing_old->clear();
      return tcells_backing_new;
    };
    int64_t num_old = 0, num_new = 0;
    // switch all the non-empty tcells vectors and clear the old ones
    for (GridPoint &grid_point : *grid_points) {
      if (grid_point.tcells) {
        num_old += grid_point.tcells->size();
        if (grid_point.tcells == &grid_point.tcells_backing_1)
          grid_point.tcells = switch_tcells_vectors(&grid_point.tcells_backing_2, &grid_point.tcells_backing_1);
        else
          grid_point.tcells = switch_tcells_vectors(&grid_point.tcells_backing_1, &grid_point.tcells_backing_2);
        num_new += grid_point.tcells->size();
      }
    }
#ifdef DEBUG
    if (check_switch) {
      auto all_num_old = upcxx::reduce_one(num_old, upcxx::op_fast_add, 0).wait();
      auto all_num_new = upcxx::reduce_one(num_new, upcxx::op_fast_add, 0).wait();
      // FIXME: this only applies if the number of tcells overall never changes
      if (!upcxx::rank_me() && all_num_new != all_num_old) DIE("tcell counts don't match, old ", all_num_old, " new ", all_num_new);
      barrier();
    }
#endif
    return num_new;
  }

  void construct(GridCoords grid_size) {
    BarrierTimer timer(__FILEFUNC__, false, true);
    Tissue::grid_size = grid_size;
    int64_t num_grid_points = get_num_grid_points();
    int64_t max_block_size = num_grid_points / rank_n();
    // we want the blocks to be cubes, so find a size that divides all three dimensions
    auto block_dim = gcd(gcd(Tissue::grid_size.x, Tissue::grid_size.y), Tissue::grid_size.z);
    Tissue::block_size = block_dim * block_dim * block_dim;
    // reduce block size until we can have at least one per process
    while (Tissue::block_size > max_block_size) {
      block_dim /= 2;
      Tissue::block_size = block_dim * block_dim * block_dim;
    }
    if (block_dim == 1)
      WARN("Using a block size of 1: this will result in a lot of communication. You should change the dimensions.");
    int64_t x_blocks = Tissue::grid_size.x / block_dim;
    int64_t y_blocks = Tissue::grid_size.y / block_dim;
    int64_t z_blocks = Tissue::grid_size.z / block_dim;
    int64_t num_blocks = num_grid_points / Tissue::block_size;
    SLOG("Dividing ", num_grid_points, " grid points into ", num_blocks, " blocks of size ", Tissue::block_size,
         " (", block_dim, "^3)\n");
    int64_t blocks_per_rank = ceil((double)num_blocks / rank_n());
    grid_points->reserve(blocks_per_rank * Tissue::block_size);
    auto mem_reqd = sizeof(GridPoint) * blocks_per_rank * Tissue::block_size;
    SLOG("Total initial memory required per process is ", get_size_str(mem_reqd), "\n");
    // FIXME: it may be more efficient (less communication) to have contiguous blocks
    // this is the quick & dirty approach
    for (int64_t i = 0; i < blocks_per_rank; i++) {
      int64_t start_id = (i * rank_n() + rank_me()) * Tissue::block_size;
      if (start_id >= num_grid_points) break;
      for (auto id = start_id; id < start_id + Tissue::block_size; id++) {
        GridCoords coords(id, Tissue::grid_size);
        // FIXME: this is an epicall on every grid point.
        // They should be placed according to the underlying lung structure (gaps, etc)
        grid_points->push_back({.id = id, .coords = coords,
             .epicell = new EpiCell({.id = id, .status = EpiCellStatus::Healthy, .num_steps_infected = 0}),
             .tcells_backing_1 = {}, .tcells_backing_2 = {}, .tcells = nullptr});
      }
    }
/*
#ifdef DEBUG
    for (auto &grid_point : (*grid_points)) {
      DBG("grid point ", grid_point.str(), "\n");
    }
#endif
*/
    barrier();
  }

  GridPoint *get_first_local_grid_point() {
    grid_point_iter = grid_points->begin();
    if (grid_point_iter == grid_points->end()) return nullptr;
    auto grid_point = &(*grid_point_iter);
    ++grid_point_iter;
    return grid_point;
  }

  GridPoint *get_next_local_grid_point() {
    if (grid_point_iter == grid_points->end()) return nullptr;
    auto grid_point = &(*grid_point_iter);
    ++grid_point_iter;
    return grid_point;
  }

};

inline GridCoords Tissue::grid_size;
inline int Tissue::block_size;
