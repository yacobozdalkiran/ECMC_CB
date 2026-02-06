#include <iostream>

#include "../gauge/GaugeField.h"
#include "../mpi/HalosExchange.h"
#include "../mpi/MpiTopology.h"
#include "../observables/observables_mpi.h"

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Lattice creation + RNG
    int L = 5;
    GeometryCB geo(L);
    GaugeField field(geo);
    std::random_device rd;
    std::mt19937_64 rng(123);
    field.hot_start(rng);

    // MPI
    int n_core_dims = 2;
    mpi::MpiTopology topo(n_core_dims);

    // Halos
    HalosCB halos_cb(geo);

    // Test plaquette

    HaloObs halo_obs(geo);
    double p = mpi::haloobs::mean_plaquette_global(field, geo, halo_obs, topo);
    if (topo.rank == 0) {
        std::cout << "WITH HALOS ===============================\n";
        std::cout << "Mean plaquette before shift : " << p << "\n";
    }

    // Test shift
    ShiftParams sp{shift_type::pos, halo_coord::X, 3};
    HalosShift halos_shift(sp.L_shift, geo);
    if (topo.rank == 0) {
        std::cout << "Attempting shift...\n";
    }
    mpi::shiftcb::shift(field, geo, halos_shift, topo, sp);
    if (topo.rank == 0) {
        std::cout << "Shift succeded !\n";
    }

    p = mpi::haloobs::mean_plaquette_global(field, geo, halo_obs, topo);
    if (topo.rank == 0) {
        std::cout << "Mean plaquette after shift : " << p << "\n";
    }

    // Test fill halos ecmc (in field)
    if (topo.rank == 0) {
        std::cout << "NO HALOS ===============================\n";
    }
    mpi::ecmccb::fill_and_exchange(field, geo, halos_cb, topo);
    p = mpi::nohalo::mean_plaquette_global(field, geo, topo);
    if (topo.rank == 0) {
        std::cout << "Mean plaquette before shift : " << p << "\n";
    }

    if (topo.rank == 0) {
        std::cout << "Attempting shift...\n";
    }
    mpi::shiftcb::shift(field, geo, halos_shift, topo, sp);
    if (topo.rank == 0) {
        std::cout << "Shift succeded !\n";
    }
    mpi::ecmccb::fill_and_exchange(field, geo, halos_cb, topo);
    p = mpi::nohalo::mean_plaquette_global(field, geo, topo);
    if (topo.rank == 0) {
        std::cout << "Mean plaquette after shift : " << p << "\n";
    }

    // End MPI
    MPI_Finalize();
}
