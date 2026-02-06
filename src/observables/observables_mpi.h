//
// Created by ozdalkiran-l on 1/8/26.
//

#ifndef INC_4D_MPI_OBSERVABLES_H
#define INC_4D_MPI_OBSERVABLES_H

#include "../gauge/GaugeField.h"
#include "../mpi/HalosObs.h"
#include "../mpi/MpiTopology.h"

namespace mpi::haloobs {
    double sum_plaquette_bulk(const GaugeField &field, const GeometryCB &geo);
    double sum_plaquette_boundaries(const GaugeField &field, const GeometryCB &geo, const HaloObs &halo_obs);
    double mean_plaquette_local(const GaugeField &field, const GeometryCB &geo, HaloObs &halo_obs, mpi::MpiTopology &topo);
    double mean_plaquette_global(const GaugeField &field, const GeometryCB &geo, HaloObs &halo_obs, mpi::MpiTopology &topo);
}

namespace mpi::nohalo{
    double mean_plaquette_local(const GaugeField &field, const GeometryCB &geo);
    double mean_plaquette_global(const GaugeField &field, const GeometryCB &geo, MpiTopology &topo);
}

#endif //INC_4D_MPI_OBSERVABLES_H
