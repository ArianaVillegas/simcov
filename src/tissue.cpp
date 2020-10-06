#include "tissue.hpp"

using namespace std;

GridCoords::GridCoords(int64_t i, const GridCoords &grid_size) {
  z = i / (grid_size.x * grid_size.y);
  i = i % (grid_size.x * grid_size.y);
  y = i / grid_size.x;
  x = i % grid_size.x;
}

GridCoords::GridCoords(shared_ptr<Random> rnd_gen, const GridCoords &grid_size) {
  x = rnd_gen->get(0, grid_size.x);
  y = rnd_gen->get(0, grid_size.y);
  z = rnd_gen->get(0, grid_size.z);
}

void GridCoords::set_rnd(shared_ptr<Random> rnd_gen, const GridCoords &grid_size) {
  x = rnd_gen->get(0, grid_size.x);
  y = rnd_gen->get(0, grid_size.y);
  z = rnd_gen->get(0, grid_size.z);
}


TCell::TCell(const string &id, GridCoords &coords) : id(id) {
  vascular_period = _rnd_gen->get_normal_distr(_options->tcell_vascular_period);
  tissue_period = _rnd_gen->get_normal_distr(_options->tcell_tissue_period);
}


string EpiCell::str() {
  ostringstream oss;
  oss << id << " " << EpiCellStatusStr[(int)status];
  return oss.str();
}

void EpiCell::infect() {
  assert(status == EpiCellStatus::HEALTHY);
  status = EpiCellStatus::INCUBATING;
  infection_period = _rnd_gen->get_normal_distr(_options->infection_period);
  incubation_period = _rnd_gen->get_normal_distr(_options->incubation_period);
  initial_incubation_period = incubation_period;
  apoptosis_period = _rnd_gen->get_normal_distr(_options->apoptosis_period);
}

bool EpiCell::transition_to_expressing() {
  assert(status == EpiCellStatus::INCUBATING);
  incubation_period--;
  infection_period--;
  if (incubation_period > 0) return false;
  status = EpiCellStatus::EXPRESSING;
  return true;
}

bool EpiCell::apoptosis_death() {
  assert(status == EpiCellStatus::APOPTOTIC);
  apoptosis_period--;
  infection_period--;
  if (apoptosis_period > 0 && infection_period > 0) return false;
  status = EpiCellStatus::DEAD;
  return true;
}

bool EpiCell::infection_death() {
  infection_period--;
  if (infection_period > 0) return false;
  status = EpiCellStatus::DEAD;
  return true;
}

bool EpiCell::is_active() {
  return (status != EpiCellStatus::HEALTHY && status != EpiCellStatus::DEAD);
}

double EpiCell::get_binding_prob() {
  return min((double)(initial_incubation_period - incubation_period) / initial_incubation_period,
             1.0);
}

string GridPoint::str() const {
  ostringstream oss;
  oss << "id " << id << ", xyz " << coords.str() << ", epi " << (epicell ? epicell->str() : "none")
      << ", v " << virus << ", c " << chemokine;
  return oss.str();
}

bool GridPoint::is_active() {
  // it could be incubating but without anything else set
  return ((epicell && epicell->is_active()) || virus > 0 || chemokine > 0 || tcell);
}



intrank_t Tissue::get_rank_for_grid_point(const GridCoords &coords) {
  int64_t id = coords.to_1d(Tissue::grid_size);
  int64_t block_i = id / Tissue::block_size;
  return block_i % rank_n();
}

GridPoint *Tissue::get_local_grid_point(grid_points_t &grid_points, const GridCoords &coords) {
  auto id = coords.to_1d(Tissue::grid_size);
  int64_t block_i = id / Tissue::block_size / rank_n();
  int64_t i = id % Tissue::block_size + block_i * Tissue::block_size;
  assert(i < grid_points->size());
  GridPoint *grid_point = &(*grid_points)[i];
  assert(grid_point->id == id);
  assert(grid_point->coords.x == coords.x);
  assert(grid_point->coords.y == coords.y);
  assert(grid_point->coords.z == coords.z);
  return grid_point;
}

vector<GridCoords> Tissue::get_neighbors(GridCoords c, GridCoords grid_size) {
  vector<GridCoords> n = {};
  int64_t newx, newy, newz;
  for (int64_t i = -1; i <= 1; i++) {
    for (int64_t j = -1; j <= 1; j++) {
      for (int64_t k = -1; k <= 1; k++) {
        newx = c.x + i;
        newy = c.y + j;
        newz = c.z + k;
        if ((newx >= 0 && newx < grid_size.x) && (newy >= 0 && newy < grid_size.y) &&
            (newz >= 0 && newz < grid_size.z)) {
          if (newx != c.x || newy != c.y || newz != c.z) n.push_back(GridCoords(newx, newy, newz));
        }
      }
    }
  }
  return n;
}

int64_t Tissue::get_num_grid_points() {
  return Tissue::grid_size.x * Tissue::grid_size.y * Tissue::grid_size.z;
}

int64_t Tissue::get_num_local_grid_points() {
  return grid_points->size();
}

void Tissue::set_infected_epicell(GridCoords coords) {
  upcxx::rpc(
      get_rank_for_grid_point(coords),
      [](grid_points_t &grid_points, new_active_grid_points_t &new_active_grid_points,
         GridCoords coords) {
        GridPoint *grid_point = Tissue::get_local_grid_point(grid_points, coords);
        DBG("set infected for grid point ", grid_point, " ", grid_point->str(), "\n");
        grid_point->virus = 0.01;
        new_active_grid_points->insert({grid_point, true});
      },
      grid_points, new_active_grid_points, coords)
      .wait();
}

void Tissue::accumulate_concentrations(grid_to_conc_map_t &concs_to_update,
                                       IntermittentTimer &timer) {
  timer.start();
  // accumulate updates for each target rank
  unordered_map<intrank_t, vector<pair<GridCoords, array<double, 2>>>> target_rank_updates;
  for (auto& [coords_1d, concentrations] : concs_to_update) {
    upcxx::progress();
    GridCoords coords(coords_1d, Tissue::grid_size);
    target_rank_updates[get_rank_for_grid_point(coords)].push_back({coords, concentrations});
  }
  future<> fut_chain = make_future<>();
  // dispatch all updates to each target rank in turn
  for (auto& [target_rank, update_vector] : target_rank_updates) {
    upcxx::progress();
    auto fut = upcxx::rpc(
        target_rank,
        [](grid_points_t &grid_points, new_active_grid_points_t &new_active_grid_points,
           upcxx::view<pair<GridCoords, array<double, 2>>> update_vector) {
          for (auto &update_pair : update_vector) {
            auto &coords = update_pair.first;
            GridPoint *grid_point = Tissue::get_local_grid_point(grid_points, coords);
            new_active_grid_points->insert({grid_point, true});
            // just accumulate the concentrations. We will adjust them to be the average
            // of all neighbors later
            grid_point->nb_chemokine += update_pair.second[0];
            grid_point->nb_virus += update_pair.second[1];
          }
        },
        grid_points, new_active_grid_points, upcxx::make_view(update_vector));
    fut_chain = when_all(fut_chain, fut);
 }
  fut_chain.wait();
  timer.stop();
}

double Tissue::get_chemokine(GridCoords coords) {
  return upcxx::rpc(
             get_rank_for_grid_point(coords),
             [](grid_points_t &grid_points, GridCoords coords) {
               GridPoint *grid_point = Tissue::get_local_grid_point(grid_points, coords);
               return grid_point->chemokine;
             },
             grid_points, coords)
      .wait();
}

list<TCell> *Tissue::get_circulating_tcells() {
  return &(*circulating_tcells);
}

void Tissue::add_circulating_tcell(GridCoords coords, TCell tcell) {
  upcxx::rpc(
      get_rank_for_grid_point(coords),
      [](dist_object<list<TCell>> &circulating_tcells, TCell tcell) {
        circulating_tcells->push_back(tcell);
      },
      circulating_tcells, tcell)
      .wait();
}

bool Tissue::try_add_tissue_tcell(GridCoords coords, TCell tcell, bool extravasate) {
  return upcxx::rpc(
             get_rank_for_grid_point(coords),
             [](grid_points_t &grid_points, new_active_grid_points_t &new_active_grid_points,
                GridCoords coords, TCell tcell, bool extravasate) {
               GridPoint *grid_point = Tissue::get_local_grid_point(grid_points, coords);
               if (grid_point->tcell) return false;
               if (extravasate && !_rnd_gen->trial_success(grid_point->chemokine)) return false;
               new_active_grid_points->insert({grid_point, true});
               tcell.moved = true;
               grid_point->tcell = new TCell(tcell);
               return true;
             },
             grid_points, new_active_grid_points, coords, tcell, extravasate)
      .wait();
}

static int get_cube_block_dim(int64_t num_grid_points) {
  int min_cubes_per_rank = 2;
  int block_dim = 1;
  int min_dim = min(min(Tissue::grid_size.x, Tissue::grid_size.y), Tissue::grid_size.z);
  for (int i = 1; i < min_dim; i++) {
    // only allow dims that divide each main dimension perfectly
    if (remainder(Tissue::grid_size.x, i) || remainder(Tissue::grid_size.y, i) ||
        remainder(Tissue::grid_size.z, i)) {
      DBG("dim ", i, " does not divide all main dimensions cleanly\n");
      continue;
    }
    size_t cube = (size_t)pow((double)i, 3.0);
    size_t num_cubes = num_grid_points / cube;
    DBG("cube size ", cube, " num cubes ", num_cubes, "\n");
    if (num_cubes < rank_n() * min_cubes_per_rank) {
      DBG("not enough cubes ", num_cubes, " < ", rank_n() * min_cubes_per_rank, "\n");
      break;
    }
    // there is a remainder - this is not a perfect division
    if (remainder(num_grid_points, cube)) {
      DBG("there is a remainder - don't use\n");
      continue;
    } else {
      DBG("selected dim ", i, "\n");
      block_dim = i;
    }
  }
  return block_dim;
}

static int get_square_block_dim(int64_t num_grid_points) {
  int min_squares_per_rank = 2;
  int block_dim = 1;
  int min_dim = min(Tissue::grid_size.x, Tissue::grid_size.y);
  for (int i = 1; i < min_dim; i++) {
    // only allow dims that divide each main dimension perfectly
    if (remainder(Tissue::grid_size.x, i) || remainder(Tissue::grid_size.y, i)) {
      DBG("dim ", i, " does not divide all main dimensions cleanly\n");
      continue;
    }
    size_t square = (size_t)pow((double)i, 2.0);
    size_t num_squares = num_grid_points / square;
    DBG("square size ", square, " num squares ", num_squares, "\n");
    if (num_squares < rank_n() * min_squares_per_rank) {
      DBG("not enough squares ", num_squares, " < ", rank_n() * min_squares_per_rank, "\n");
      break;
    }
    // there is a remainder - this is not a perfect division
    if (remainder(num_grid_points, square)) {
      DBG("there is a remainder - don't use\n");
      continue;
    } else {
      DBG("selected dim ", i, "\n");
      block_dim = i;
    }
  }
  return block_dim;
}

void Tissue::construct(GridCoords grid_size) {
  auto remainder = [](int64_t numerator, int64_t denominator) -> bool {
    return ((double)numerator / denominator - (numerator / denominator) != 0);
  };
  BarrierTimer timer(__FILEFUNC__, false, true);
  Tissue::grid_size = grid_size;
  int64_t num_grid_points = get_num_grid_points();
  // find the biggest cube that perfectly divides the grid and gives enough
  // data for at least two cubes per rank (for load
  // balance)
  // This is a trade-off: the more data is blocked, the better the locality,
  // but load balance could be a problem if not all
  // ranks get the same number of cubes. Also, having bigger cubes could lead
  // to load imbalance if all of the computation is
  // happening within a cube.
  int block_dim = (Tissue::grid_size.z > 1 ? get_cube_block_dim(num_grid_points) :
                                             get_square_block_dim(num_grid_points));
  if (block_dim == 1)
    SWARN("Using a block size of 1: this will result in a lot of "
          "communication. You should change the dimensions.");
  Tissue::block_size = (Tissue::grid_size.z > 1 ? block_dim * block_dim * block_dim :
                                                  block_dim * block_dim);
  int64_t x_blocks = Tissue::grid_size.x / block_dim;
  int64_t y_blocks = Tissue::grid_size.y / block_dim;
  int64_t z_blocks = (Tissue::grid_size.z > 1 ? Tissue::grid_size.z / block_dim : 1);
  int64_t num_blocks = num_grid_points / Tissue::block_size;
  if (Tissue::grid_size.z > 1)
    SLOG("Dividing ", num_grid_points, " grid points into ", num_blocks, " blocks of size ",
         Tissue::block_size, " (", block_dim, "^3)\n");
  else
    SLOG("Dividing ", num_grid_points, " grid points into ", num_blocks, " squares of size ",
         Tissue::block_size, " (", block_dim, "^2)\n");

  int64_t blocks_per_rank = ceil((double)num_blocks / rank_n());
  SLOG_VERBOSE("Each process has ", blocks_per_rank, " blocks\n");
  grid_points->reserve(blocks_per_rank * Tissue::block_size);
  auto mem_reqd = sizeof(GridPoint) * blocks_per_rank * Tissue::block_size;
  SLOG("Total initial memory required per process is a max of ", get_size_str(mem_reqd), "\n");
  // FIXME: it may be more efficient (less communication) to have contiguous blocks
  // this is the quick & dirty approach
  for (int64_t i = 0; i < blocks_per_rank; i++) {
    int64_t start_id = (i * rank_n() + rank_me()) * Tissue::block_size;
    if (start_id >= num_grid_points) break;
    for (auto id = start_id; id < start_id + Tissue::block_size; id++) {
      assert(id < num_grid_points);
      GridCoords coords(id, Tissue::grid_size);
      auto neighbors = get_neighbors(coords, grid_size);
      // infectable epicells should be placed according to the underlying lung structure
      // (gaps, etc)
      EpiCell *epicell = new EpiCell(id);
      epicell->infectable = true;
      grid_points->emplace_back(GridPoint({id, coords, neighbors, epicell}));
#ifdef DEBUG
      DBG("adding grid point ", id, " at ", coords.str(), "\n");
      ostringstream oss;
      for (auto nb_coords : neighbors) {
        oss << nb_coords.str() << " ";
      }
      DBG("nbs: ", oss.str(), "\n");
#endif
    }
  }
  barrier();
}

pair<size_t, size_t> Tissue::dump_blocks(const string &fname, const string &header_str,
                                         ViewObject view_object) {
  auto fileno = open(fname.c_str(), O_WRONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  if (fileno == -1) DIE("Cannot open file ", fname, ": ", strerror(errno), "\n");
  size_t tot_bytes_written = 0;
  if (!upcxx::rank_me()) {
    tot_bytes_written = pwrite(fileno, header_str.c_str(), header_str.length(), 0);
    if (tot_bytes_written != header_str.length())
      DIE("Could not write all ", header_str.length(), " bytes: only wrote ", tot_bytes_written);
  }
  //DBG("Writing samples to ", fname, "\n");
  //DBG("header size is ", header_str.length(), "\n");
  size_t grid_points_written = 0;
  int64_t num_grid_points = get_num_grid_points();
  int64_t num_blocks = num_grid_points / Tissue::block_size;
  int64_t blocks_per_rank = ceil((double)num_blocks / rank_n());
  size_t buf_size = Tissue::block_size;
  unsigned char *buf = new unsigned char[buf_size];
  for (int64_t i = 0; i < blocks_per_rank; i++) {
    int64_t start_id = (i * rank_n() + rank_me()) * Tissue::block_size;
    if (start_id >= num_grid_points) break;
    for (auto id = start_id; id < start_id + Tissue::block_size; id++) {
      assert(id < num_grid_points);
      GridCoords coords(id, Tissue::grid_size);
      GridPoint *grid_point = Tissue::get_local_grid_point(grid_points, coords);
      unsigned char val = 0;
      if (view_object == ViewObject::TCELL_TISSUE) {
        if (grid_point->tcell) val = 255;
      } else if (view_object == ViewObject::EPICELL) {
        if (grid_point->epicell) {
          switch (grid_point->epicell->status) {
            case EpiCellStatus::HEALTHY: val = 0; break;
            case EpiCellStatus::INCUBATING: val = 1; break;
            case EpiCellStatus::EXPRESSING: val = 2; break;
            case EpiCellStatus::APOPTOTIC: val = 3; break;
            case EpiCellStatus::DEAD: val = 4; break;
          }
        }
      } else if (view_object == ViewObject::VIRUS) {
        if (grid_point->virus < 0) DIE("virus is negative ", grid_point->virus);
        val = 255 * grid_point->virus;
        if (grid_point->virus > 0 && val == 0) val = 1;
      } else if (view_object == ViewObject::CHEMOKINE) {
        if (grid_point->chemokine < 0) DIE("chemokine is negative ", grid_point->chemokine);
        val = 255 * grid_point->chemokine;
        if (grid_point->chemokine > 0 && val == 0) val = 1;
      }
      grid_points_written++;
      buf[id - start_id] = val;
    }
    size_t fpos = start_id + header_str.length();
    auto bytes_written = pwrite(fileno, buf, buf_size, fpos);
    if (bytes_written != buf_size)
      DIE("Could not write all ", buf_size, " bytes; only wrote ", bytes_written, "\n");
    tot_bytes_written += bytes_written;
    //DBG("wrote block ", i, ", ", bytes_written, " bytes at position ", fpos, "\n");
  }
  delete[] buf;
  close(fileno);
  return {tot_bytes_written, grid_points_written};
}

GridPoint *Tissue::get_first_local_grid_point() {
  grid_point_iter = grid_points->begin();
  if (grid_point_iter == grid_points->end()) return nullptr;
  auto grid_point = &(*grid_point_iter);
  ++grid_point_iter;
  return grid_point;
}

GridPoint *Tissue::get_next_local_grid_point() {
  if (grid_point_iter == grid_points->end()) return nullptr;
  auto grid_point = &(*grid_point_iter);
  ++grid_point_iter;
  return grid_point;
}

GridPoint *Tissue::get_first_active_grid_point() {
  active_grid_point_iter = active_grid_points.begin();
  if (active_grid_point_iter == active_grid_points.end()) return nullptr;
  auto grid_point = active_grid_point_iter->first;
  ++active_grid_point_iter;
  return grid_point;
}

GridPoint *Tissue::get_next_active_grid_point() {
  if (active_grid_point_iter == active_grid_points.end()) return nullptr;
  auto grid_point = active_grid_point_iter->first;
  ++active_grid_point_iter;
  return grid_point;
}

void Tissue::set_active(GridPoint *grid_point) {
  new_active_grid_points->insert({grid_point, true});
}

void Tissue::erase_active(GridPoint *grid_point) {
  active_grid_points.erase(grid_point);
}

void Tissue::add_new_actives(IntermittentTimer &timer) {
  timer.start();
  DBG("add ", new_active_grid_points->size(), " new active grid points\n");
  for (auto elem : *new_active_grid_points) {
    DBG("inserting from new active ", elem.first, " ", elem.first->str(), "\n");
    active_grid_points.insert(elem);
  }
  new_active_grid_points->clear();
  timer.stop();
}

size_t Tissue::get_num_actives() {
  return active_grid_points.size();
}

#ifdef DEBUG
void Tissue::check_actives(int time_step) {
  for (int64_t i = 0; i < grid_points->size(); i++) {
    GridPoint *grid_point = &(*grid_points)[i];
    bool found_active = (active_grid_points.find(grid_point) != active_grid_points.end());
    if (grid_point->is_active() && !found_active)
      DIE(time_step, ": active grid point ", grid_point->str(), " not found in active list");
    if (!grid_point->is_active() && found_active)
      DIE(time_step, ": inactive grid point ", grid_point->str(), " found in active list");
  }
}
#endif
