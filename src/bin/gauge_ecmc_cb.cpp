#include <iostream>

#include "../ecmc/ecmc_mpi_cb.h"
#include "../gauge/GaugeField.h"
#include "../heatbath/heatbath_mpi.h"
#include "../mpi/HalosExchange.h"
#include "../mpi/MpiTopology.h"
#include "../observables/observables_mpi.h"

void ecmc() {
    // MPI
    int n_core_dims = 2;
    mpi::MpiTopology topo(n_core_dims);

    // Lattice creation + RNG
    int L = 4;
    GeometryCB geo(L);
    GaugeField field(geo);
    std::random_device rd;
    std::mt19937_64 rng(rd() + topo.rank);
    field.hot_start(rng);

    // Test ECMC
    HalosCB halo_cb(geo);
    mpi::haloscb::fill_and_exchange(field, geo, halo_cb, topo);

    // Params ECMC
    ECMCParams ep{.beta = 6.0,
                  .N_samples = 20,
                  .param_theta_sample = 900,
                  .param_theta_refresh = 1400,
                  .poisson = false,
                  .epsilon_set = 0.15};

    // Shift objects
    HalosShift halo_shift(geo.L, geo);
    ShiftParams sp{.stype = pos, .coord = UNSET, .L_shift = 0};

    int Ntot = 1000;
    int Nloc = 10;

    for (int i = 0; i < Ntot; i++) {
        for (int j = 0; j < Nloc; j++) {
            // Even parity :
            if (topo.rank == 0) {
                std::cout << "Even parity :\n";
            }
            parity active_parity = even;
            mpi::ecmccb::samples_improved(field, geo, ep, rng, topo, active_parity, halo_cb);

            // Odd parity :
            if (topo.rank == 0) {
                std::cout << "Odd parity :\n";
            }
            active_parity = odd;
            mpi::ecmccb::samples_improved(field, geo, ep, rng, topo, active_parity, halo_cb);
        }
        // Random shift
        mpi::shiftcb::random_shift(field, geo, halo_shift, topo, sp, rng);
    }
}

void hb() {
    // MPI
    int n_core_dims = 2;
    mpi::MpiTopology topo(n_core_dims);

    // Lattice creation + RNG
    int L = 4;
    GeometryCB geo(L);
    GaugeField field(geo);
    std::random_device rd;
    std::mt19937_64 rng(rd() + topo.rank);
    field.hot_start(rng);

    // Test HB
    HalosCB halo_cb(geo);
    HbParams hp{.beta = 6.0, .N_samples = 20, .N_hits = 3, .N_sweeps = 3};
    mpi::haloscb::fill_and_exchange(field, geo, halo_cb, topo);

    // Shift objects
    HalosShift halo_shift(geo.L, geo);
    ShiftParams sp{.stype = pos, .coord = UNSET, .L_shift = 0};

    int Ntot = 1000;
    int Nloc = 10;

    for (int i = 0; i < Ntot; i++) {
        for (int j = 0; j < Nloc; j++) {
            // Even parity :
            if (topo.rank == 0) {
                std::cout << "Even parity :\n";
            }
            parity active_parity = even;
            mpi::heatbathcb::samples(field, geo, topo, hp, rng, active_parity, halo_cb);

            // Odd parity :
            if (topo.rank == 0) {
                std::cout << "Odd parity :\n";
            }
            active_parity = odd;
            mpi::heatbathcb::samples(field, geo, topo, hp, rng, active_parity, halo_cb);
        }

        // Random shift
        mpi::shiftcb::random_shift(field, geo, halo_shift, topo, sp, rng);
    }
}

void check_rd_shift() {
    // MPI
    int n_core_dims = 2;
    mpi::MpiTopology topo(n_core_dims);

    // Lattice creation + RNG
    int L = 4;
    GeometryCB geo(L);
    GaugeField field(geo);
    std::random_device rd;
    std::mt19937_64 rng(rd() + topo.rank);
    field.hot_start(rng);

    // Test HB
    HalosCB halo_cb(geo);

    // Shift objects
    HalosShift halo_shift(geo.L, geo);
    ShiftParams sp{.stype = pos, .coord = UNSET, .L_shift = 0};

    mpi::haloscb::fill_and_exchange(field, geo, halo_cb, topo);
    double p = mpi::nohalo::mean_plaquette_global(field, geo, topo, halo_cb);
    if (topo.rank == 0) {
        std::cout << "Before shift P = " << p << "\n";
    }
    mpi::shiftcb::random_shift(field, geo, halo_shift, topo, sp, rng);
    mpi::haloscb::fill_and_exchange(field, geo, halo_cb, topo);
    p = mpi::nohalo::mean_plaquette_global(field, geo, topo, halo_cb);
    if (topo.rank == 0) {
        std::cout << "After shift P = " << p << "\n";
    }
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    ecmc();
    // End MPI
    MPI_Finalize();
}
