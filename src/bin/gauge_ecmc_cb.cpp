#include <iostream>

#include "../gauge/GaugeField.h"
#include "../mpi/HalosExchange.h"
#include "../mpi/MpiTopology.h"
#include "../observables/observables_mpi.h"

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // MPI
    int n_core_dims = 2;
    mpi::MpiTopology topo(n_core_dims);

    // Lattice creation + RNG
    int L = 5;
    GeometryCB geo(L);
    GaugeField field(geo);
    std::random_device rd;
    std::mt19937_64 rng(rd() + topo.rank);
    field.hot_start(rng);

    // Test plaquette

    // HalosObs
    HaloObs halo_obs(geo);
    double p = mpi::haloobs::mean_plaquette_global(field, geo, halo_obs, topo);
    if (topo.rank == 0) {
        std::cout << "WITH HALOS ===============================\n";
        std::cout << "Mean plaquette before shift : " << p << "\n";
    }
    if (topo.rank == 0) {
        std::cout << "NO HALOS ===============================\n";
    }
    // HalosCB
    HalosCB halos_cb(geo);
    mpi::ecmccb::fill_and_exchange(field, geo, halos_cb, topo);
    p = mpi::nohalo::mean_plaquette_global(field, geo, topo);
    if (topo.rank == 0) {
        std::cout << "Mean plaquette before shift : " << p << "\n";
    }

    // Shift
    ShiftParams sp{shift_type::pos, halo_coord::X, 3};
    HalosShift halos_shift(sp.L_shift, geo);
    mpi::shiftcb::random_shift(field, geo, halos_shift, topo, sp, rng);

    p = mpi::haloobs::mean_plaquette_global(field, geo, halo_obs, topo);
    if (topo.rank == 0) {
        std::cout << "WITH HALOS ===============================\n";
        std::cout << "Mean plaquette after shift : " << p << "\n";
    }
    if (topo.rank == 0) {
        std::cout << "NO HALOS ===============================\n";
    }
    // HalosCB
    mpi::ecmccb::fill_and_exchange(field, geo, halos_cb, topo);
    p = mpi::nohalo::mean_plaquette_global(field, geo, topo);
    if (topo.rank == 0) {
        std::cout << "Mean plaquette after shift : " << p << "\n";
    }

    // Check staples
    for (int t = 0; t < L; t++) {
        for (int z = 0; z < L; z++) {
            for (int y = 0; y < L; y++) {
                for (int x = 0; x < L; x++) {
                    size_t site = geo.index(x, y, z, t);  // x
                    for (int mu = 0; mu < 4; mu++) {
                        if (!geo.is_frozen(site, mu)) {
                            for (int j = 0; j < 6; j++) {
                                for (int i = 0; i < 3; i++) {
                                    if (geo.links_staples[geo.index_staples(site, mu, j, i)]
                                            .first == SIZE_MAX) {
                                        std::cerr << "Undefined staples link !\n";
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "All staples links checked !\n";

    // End MPI
    MPI_Finalize();
}
