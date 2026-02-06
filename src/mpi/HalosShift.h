//
// Created by ozdalkiran-l on 1/12/26.
//

#ifndef INC_4D_MPI_HALO_H
#define INC_4D_MPI_HALO_H

#include "../gauge/GaugeField.h"
#include <iostream>
#include "../mpi/types.h"


struct ShiftParams {
    shift_type stype=pos;
    halo_coord coord=UNSET;
    int L_shift=0;
};


//Halos used to shift the gauge configurations between all nodes
class HalosShift {
public:
    std::vector<Complex> send;
    std::vector<Complex> recv;
    int L_shift; //Length of the max shift, such that V_halo = L*L*L*L_shift
    int L; //Size of the square lattice
    int V_halo; //Volume (number of sites) of the halos

    explicit HalosShift(int L_shift_, const GeometryCB &geo) {
        L_shift = L_shift_;
        L = geo.L;
        V_halo = L*L*L*L_shift;
        send.resize(V_halo*4*9);
        recv.resize(V_halo*4*9);
    }

    //Index function for local coordinates x,y,z,t of halo send/recv
    [[nodiscard]] size_t index_halo(int x, int y, int z, int t, const ShiftParams &sp) const {
        size_t index{};
        if (sp.coord == X)
            index = ((static_cast<size_t>(t) * L + z) * L + y) * L_shift + x;
        if (sp.coord == Y)
            index = ((static_cast<size_t>(t) * L + z) * L_shift+ y) * L + x;
        if (sp.coord == Z)
            index =  ((static_cast<size_t>(t) * L_shift + z) * L + y)* L + x;
        if (sp.coord == T)
            index =  ((static_cast<size_t>(t)*L + z) * L + y)* L + x;
        return index;
    }

    //Non const mapping of halo_send to SU3 matrices
    Eigen::Map<SU3> view_halo_send(size_t site, int mu) {
        return Eigen::Map<SU3>(&send[(site * 4 + mu) * 9]);
    }

    //Const mapping of halo_send to SU3 matrices
    [[nodiscard]] Eigen::Map<const SU3> view_halo_send_const(size_t site, int mu) const {
        return Eigen::Map<const SU3>(&send[(site * 4 + mu) * 9]);
    }

    //Non const mapping of halo_recv to SU3 matrices
    Eigen::Map<SU3> view_halo_rec(size_t site, int mu) {
        return Eigen::Map<SU3>(&recv[(site * 4 + mu) * 9]);
    }

    //Const mapping of halo_recv to SU3 matrices
    [[nodiscard]] Eigen::Map<const SU3> view_halo_rec_const(size_t site, int mu) const {
        return Eigen::Map<const SU3>(&recv[(site * 4 + mu) * 9]);
    }
};

#endif
