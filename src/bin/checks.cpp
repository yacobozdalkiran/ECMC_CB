
#include <iostream>

#include "../gauge/GaugeField.h"
#include "../mpi/HalosExchange.h"
#include "../mpi/MpiTopology.h"
#include "../observables/observables_mpi.h"


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
    mpi::haloscb::fill_and_exchange(field, geo, halo_cb, topo);

    // Shift objects
    HalosShift halo_shift(geo.L, geo);
    ShiftParams sp{.stype = pos, .coord = UNSET, .L_shift = 0};

    double p = mpi::nohalo::mean_plaquette_global(field, geo, topo, halo_cb);
    if (topo.rank == 0) {
        std::cout << "Before shift P = " << p << "\n";
    }
    mpi::shiftcb::random_shift(field, geo, halo_shift, topo, sp, halo_cb, rng);
    p = mpi::nohalo::mean_plaquette_global(field, geo, topo, halo_cb);
    if (topo.rank == 0) {
        std::cout << "After shift P = " << p << "\n";
    }
}

void check_shift(){
    // MPI
    int n_core_dims = 2;
    mpi::MpiTopology topo(n_core_dims);

    // Lattice creation + RNG
    int L = 4;
    GeometryCB geo(L);
    GaugeField field(geo);
    std::random_device rd;
    std::mt19937_64 rng(rd() + topo.rank);
    field.cold_start();

    // Halos pour observables
    HalosCB halo_cb(geo);

    // Shift objects
    HalosShift halo_shift(geo.L, geo);
    ShiftParams sp{.stype = pos, .coord = X, .L_shift = 1};

    //Toutes les matrices sont a l'identité sauf celle là
    field.view_link(geo.index(3,0,0,0),0) = SU3::Zero();
    if (topo.rank == 0){
        std::cout << field.view_link_const(geo.index(3,0,0,0),0) << "\n\n";
    }
    mpi::haloscb::fill_and_exchange(field, geo, halo_cb, topo);

    //On shifte
    mpi::shiftcb::shift(field, geo, halo_shift, topo, sp);
    mpi::haloscb::fill_and_exchange(field, geo, halo_cb, topo);
    if (topo.rank == 0){
        std::cout << field.view_link_const(geo.index(0,0,0,0),0) << "\n\n";
        std::cout << field.view_link_const(geo.index(1,0,0,0),0) << "\n\n";
        std::cout << field.view_link_const(geo.index(2,0,0,0),0) << "\n\n";
        std::cout << field.view_link_const(geo.index(3,0,0,0),0) << "\n\n";
    }
}

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);
    check_shift();
    check_rd_shift();
    MPI_Finalize();
}
