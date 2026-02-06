//
// Created by ozdalkiran-l on 1/8/26.
//

#include "observables_mpi.h"

#include "../mpi/HalosExchange.h"

// Returns the sum of retr/3 of plaquettes of the bulk (i.e. coords btw 0 and L-2)
double mpi::haloobs::sum_plaquette_bulk(const GaugeField& field, const GeometryCB& geo) {
    double sum = 0.0;
    SU3 U1, U2, U3, U4;
    for (int t = 0; t < geo.L - 1; t++) {
        for (int z = 0; z < geo.L - 1; z++) {
            for (int y = 0; y < geo.L - 1; y++) {
                for (int x = 0; x < geo.L - 1; x++) {
                    size_t site = geo.index(x, y, z, t);
                    for (int mu = 0; mu < 4; mu++) {
                        for (int nu = mu + 1; nu < 4; nu++) {
                            U1 = field.view_link_const(site, mu);
                            U2 = field.view_link_const(geo.get_neigh(site, mu, up), nu);
                            U3 = field.view_link_const(geo.get_neigh(site, nu, up), mu).adjoint();
                            U4 = field.view_link_const(site, nu).adjoint();
                            sum += (U1 * U2 * U3 * U4).trace().real() / 3.0;
                        }
                    }
                }
            }
        }
    }
    return sum;
}

// Returns the sum of retr/3 of plaquettes on the boundaries of the field using the halos
double mpi::haloobs::sum_plaquette_boundaries(const GaugeField& field, const GeometryCB& geo,
                                              const HaloObs& halo_obs) {
    double sum_boundaries = 0.0;
    int L = geo.L;
    SU3 U1, U2, U3, U4;
    for (int t = 0; t < L; t++) {
        for (int z = 0; z < L; z++) {
            for (int y = 0; y < L; y++) {
                for (int x = 0; x < L; x++) {
                    // We treat only the boundary sites
                    if (x == L - 1 || y == L - 1 || z == L - 1 || t == L - 1) {
                        for (int mu = 0; mu < 4; mu++) {
                            for (int nu = mu + 1; nu < 4; nu++) {
                                U1 = halo_obs.get_link_with_halo_obs(field, geo, x, y, z, t, mu);
                                U2 = halo_obs.get_link_with_halo_obs(field, geo, x + (mu == 0),
                                                                     y + (mu == 1), z + (mu == 2),
                                                                     t + (mu == 3), nu);
                                U3 = halo_obs
                                         .get_link_with_halo_obs(field, geo, x + (nu == 0),
                                                                 y + (nu == 1), z + (nu == 2),
                                                                 t + (nu == 3), mu)
                                         .adjoint();
                                U4 = halo_obs.get_link_with_halo_obs(field, geo, x, y, z, t, nu)
                                         .adjoint();
                                sum_boundaries += (U1 * U2 * U3 * U4).trace().real() / 3.0;
                            }
                        }
                    }
                }
            }
        }
    }
    return sum_boundaries;
}

// Returns the sum of retr of plaquettes of the local field (includes halo filling and exchange)
double mpi::haloobs::mean_plaquette_local(const GaugeField& field, const GeometryCB& geo,
                                          HaloObs& halo_obs, mpi::MpiTopology& topo) {
    double local_mean_plaquette = 0.0;
    MPI_Request reqs[16];
    mpi::haloobs::fill_halo_obs_send(field, geo, halo_obs);
    mpi::haloobs::exchange_halos_obs(halo_obs, topo, reqs);
    // During the exchanges, we compute the plaquette in the bulk
    local_mean_plaquette += mpi::haloobs::sum_plaquette_bulk(field, geo);
    // Now we wait for all the nodes to end the exchanges
    MPI_Waitall(16, reqs, MPI_STATUSES_IGNORE);
    // Finally we compute the mean plaquette of the boundary sites
    local_mean_plaquette += sum_plaquette_boundaries(field, geo, halo_obs);
    return local_mean_plaquette;
}

// Returns the mean plaquette of the global lattice
double mpi::haloobs::mean_plaquette_global(const GaugeField& field, const GeometryCB& geo,
                                           HaloObs& halo_obs, mpi::MpiTopology& topo) {
    double local_mean_plaquette = mean_plaquette_local(field, geo, halo_obs, topo);
    double global_mean_plaquette = 0.0;
    MPI_Allreduce(&local_mean_plaquette, &global_mean_plaquette, 1, MPI_DOUBLE, MPI_SUM,
                  topo.cart_comm);
    global_mean_plaquette /= 6.0 * geo.V * topo.size;
    return global_mean_plaquette;
}

//Computation of mean plaquette with halos embedded in field (needs field halos exchange first)
double mpi::nohalo::mean_plaquette_local(const GaugeField& field, const GeometryCB& geo) {
    double sum = 0.0;
    SU3 U1, U2, U3, U4;
    for (int t = 0; t < geo.L; t++) {
        for (int z = 0; z < geo.L; z++) {
            for (int y = 0; y < geo.L; y++) {
                for (int x = 0; x < geo.L; x++) {
                    size_t site = geo.index(x, y, z, t);
                    for (int mu = 0; mu < 4; mu++) {
                        for (int nu = mu + 1; nu < 4; nu++) {
                            U1 = field.view_link_const(site, mu);
                            U2 = field.view_link_const(geo.get_neigh(site, mu, up), nu);
                            U3 = field.view_link_const(geo.get_neigh(site, nu, up), mu).adjoint();
                            U4 = field.view_link_const(site, nu).adjoint();
                            sum += (U1 * U2 * U3 * U4).trace().real() / 3.0;
                        }
                    }
                }
            }
        }
    }
    return sum;
}

//Computation of global mean plaquette with halos embedded in field (needs field halos exchanges first)
double mpi::nohalo::mean_plaquette_global(const GaugeField &field, const GeometryCB &geo, MpiTopology &topo){
    double local_mean_plaquette = mean_plaquette_local(field, geo);
    double global_mean_plaquette = 0.0;
    MPI_Allreduce(&local_mean_plaquette, &global_mean_plaquette, 1, MPI_DOUBLE, MPI_SUM,
                  topo.cart_comm);
    global_mean_plaquette /= 6.0 * geo.V * topo.size;
    return global_mean_plaquette;
}
