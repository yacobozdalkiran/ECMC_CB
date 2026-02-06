//
// Created by ozdalkiran-l on 1/9/26.
//

#ifndef INC_4D_MPI_HALOSEXCHANGE_H
#define INC_4D_MPI_HALOSEXCHANGE_H

#include "../io/params.h"
#include "HalosObs.h"
#include "HalosCB.h"
#include "HalosShift.h"
#include "MpiTopology.h"

namespace mpi::shiftcb {
void fill_halo_send(const GaugeField& field, const GeometryCB& geo, HalosShift& halo,
                    const ShiftParams& sp);
void shift_field(GaugeField& field, const GeometryCB& geo, HalosShift& halo, const ShiftParams& sp);
void exchange_halos(HalosShift& halo, mpi::MpiTopology& topo, const ShiftParams& sp,
                    MPI_Request* req);
void fill_lattice_with_halo_recv(GaugeField& field, const GeometryCB& geo, HalosShift& halo, const ShiftParams& sp);
void shift(GaugeField& field, const GeometryCB& geo, HalosShift& halo, MpiTopology& topo,
           const ShiftParams& sp);
};  // namespace mpi::shiftcb

namespace mpi::ecmccb {
void fill_halos_ecmc(const GaugeField& field, const GeometryCB& geo, HalosCB& halo);
void exchange_halos_ecmc(GaugeField& field, const GeometryCB& geo, const HalosCB& halo,
                         mpi::MpiTopology& topo);
void fill_and_exchange(GaugeField& field, const GeometryCB& geo, HalosCB& halo,
                       mpi::MpiTopology& topo);
}  // namespace mpi::ecmccb

namespace mpi::haloobs{
    void exchange_halos_obs(HaloObs &halo_obs, mpi::MpiTopology &topo, MPI_Request* reqs);
    void fill_halo_obs_send(const GaugeField &field, const GeometryCB &geo, HaloObs &halo_obs);
}

#endif  // INC_4D_MPI_HALOSEXCHANGE_H
