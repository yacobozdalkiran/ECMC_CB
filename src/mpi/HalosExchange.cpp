//
// Created by ozdalkiran-l on 1/9/26.
//

#define NDIMS 4

#include "HalosExchange.h"

#include <iostream>

#include "MpiTopology.h"

// Fills the send halo with the corresponding part of the lattice :
// from coord = L - L_shift to coord = L-1 if stype = pos
// from coord = 0 to coord = L_shift-1 if stype = neg
void mpi::shiftcb::fill_halo_send(const GaugeField& field, const GeometryCB& geo, HalosShift& halo,
                                  const ShiftParams& sp) {
    if (sp.coord == X) {
        for (int t = 0; t < geo.L; t++) {
            for (int z = 0; z < geo.L; z++) {
                for (int y = 0; y < geo.L; y++) {
                    for (int x_local = 0; x_local < halo.L_shift; x_local++) {
                        int x_glob = x_local;
                        if (sp.stype == pos) x_glob = x_local + geo.L - halo.L_shift;
                        size_t site_glob = geo.index(x_glob, y, z, t);
                        size_t site_loc = halo.index_halo(x_local, y, z, t, sp);
                        for (int mu = 0; mu < 4; mu++) {
                            halo.view_halo_send(site_loc, mu) =
                                field.view_link_const(site_glob, mu);
                        }
                    }
                }
            }
        }
    }

    if (sp.coord == Y) {
        for (int t = 0; t < geo.L; t++) {
            for (int z = 0; z < geo.L; z++) {
                for (int y_local = 0; y_local < halo.L_shift; y_local++) {
                    for (int x = 0; x < geo.L; x++) {
                        int y_glob = y_local;
                        if (sp.stype == pos) y_glob = y_local + geo.L - halo.L_shift;
                        size_t site_glob = geo.index(x, y_glob, z, t);
                        size_t site_loc = halo.index_halo(x, y_local, z, t, sp);
                        for (int mu = 0; mu < 4; mu++) {
                            halo.view_halo_send(site_loc, mu) =
                                field.view_link_const(site_glob, mu);
                        }
                    }
                }
            }
        }
    }

    if (sp.coord == Z) {
        for (int t = 0; t < geo.L; t++) {
            for (int z_local = 0; z_local < halo.L_shift; z_local++) {
                for (int y = 0; y < geo.L; y++) {
                    for (int x = 0; x < geo.L; x++) {
                        int z_glob = z_local;
                        if (sp.stype == pos) z_glob = z_local + geo.L - halo.L_shift;
                        size_t site_glob = geo.index(x, y, z_glob, t);
                        size_t site_loc = halo.index_halo(x, y, z_local, t, sp);
                        for (int mu = 0; mu < 4; mu++) {
                            halo.view_halo_send(site_loc, mu) =
                                field.view_link_const(site_glob, mu);
                        }
                    }
                }
            }
        }
    }

    if (sp.coord == T) {
        for (int t_local = 0; t_local < halo.L_shift; t_local++) {
            for (int z = 0; z < geo.L; z++) {
                for (int y = 0; y < geo.L; y++) {
                    for (int x = 0; x < geo.L; x++) {
                        int t_glob = t_local;
                        if (sp.stype == pos) t_glob = t_local + geo.L - halo.L_shift;
                        size_t site_glob = geo.index(x, y, z, t_glob);
                        size_t site_loc = halo.index_halo(x, y, z, t_local, sp);
                        for (int mu = 0; mu < 4; mu++) {
                            halo.view_halo_send(site_loc, mu) =
                                field.view_link_const(site_glob, mu);
                        }
                    }
                }
            }
        }
    }
}

// Shifts the value of all the links of the lattice of L_shift in direction +coord if stype = pos,
// -coord if stpye = neg
void mpi::shiftcb::shift_field(GaugeField& field, const GeometryCB& geo, HalosShift& halo,
                               const ShiftParams& sp) {
    int L = geo.L;
    int L_shift = halo.L_shift;
    halo_coord coord = sp.coord;
    // We want to avoid cache misses -> we follow the layout of the gauge field and have to write a
    // different loop each time
    if (sp.stype == pos) {
        // The content of the field btw 0 and (L-1-L_shift) -> L_shift - L-1
        if (coord == X) {
            for (int t = 0; t < L; t++) {
                for (int z = 0; z < L; z++) {
                    for (int y = 0; y < L; y++) {
                        for (int x = L - 1; x >= L_shift; x--) {
                            size_t site = geo.index(x, y, z, t);
                            size_t copied_site = geo.index(x - L_shift, y, z, t);
                            for (int mu = 0; mu < 4; mu++) {
                                field.view_link(site, mu) = field.view_link_const(copied_site, mu);
                            }
                        }
                    }
                }
            }
        }

        if (coord == Y) {
            for (int t = 0; t < L; t++) {
                for (int z = 0; z < L; z++) {
                    for (int y = L - 1; y >= L_shift; y--) {
                        for (int x = 0; x < L; x++) {
                            size_t site = geo.index(x, y, z, t);
                            size_t copied_site = geo.index(x, y - L_shift, z, t);
                            for (int mu = 0; mu < 4; mu++) {
                                field.view_link(site, mu) = field.view_link_const(copied_site, mu);
                            }
                        }
                    }
                }
            }
        }

        if (coord == Z) {
            for (int t = 0; t < L; t++) {
                for (int z = L - 1; z >= L_shift; z--) {
                    for (int y = 0; y < L; y++) {
                        for (int x = 0; x < L; x++) {
                            size_t site = geo.index(x, y, z, t);
                            size_t copied_site = geo.index(x, y, z - L_shift, t);
                            for (int mu = 0; mu < 4; mu++) {
                                field.view_link(site, mu) = field.view_link_const(copied_site, mu);
                            }
                        }
                    }
                }
            }
        }
        if (coord == T) {
            for (int t = L - 1; t >= L_shift; t--) {
                for (int z = 0; z < L; z++) {
                    for (int y = 0; y < L; y++) {
                        for (int x = 0; x < L; x++) {
                            size_t site = geo.index(x, y, z, t);
                            size_t copied_site = geo.index(x, y, z, t - L_shift);
                            for (int mu = 0; mu < 4; mu++) {
                                field.view_link(site, mu) = field.view_link_const(copied_site, mu);
                            }
                        }
                    }
                }
            }
        }
    }
    if (sp.stype == neg) {
        // The content of the field btw L_shift and (L-1) -> 0 - (L-1-L_shift)
        if (coord == X) {
            for (int t = 0; t < L; t++) {
                for (int z = 0; z < L; z++) {
                    for (int y = 0; y < L; y++) {
                        for (int x = 0; x < L - L_shift; x++) {
                            size_t site = geo.index(x, y, z, t);
                            size_t copied_site = geo.index(x + L_shift, y, z, t);
                            for (int mu = 0; mu < 4; mu++) {
                                field.view_link(site, mu) = field.view_link_const(copied_site, mu);
                            }
                        }
                    }
                }
            }
        }

        if (coord == Y) {
            for (int t = 0; t < L; t++) {
                for (int z = 0; z < L; z++) {
                    for (int y = 0; y < L - L_shift; y++) {
                        for (int x = 0; x < L; x++) {
                            size_t site = geo.index(x, y, z, t);
                            size_t copied_site = geo.index(x, y + L_shift, z, t);
                            for (int mu = 0; mu < 4; mu++) {
                                field.view_link(site, mu) = field.view_link_const(copied_site, mu);
                            }
                        }
                    }
                }
            }
        }

        if (coord == Z) {
            for (int t = 0; t < L; t++) {
                for (int z = 0; z < L - L_shift; z++) {
                    for (int y = 0; y < L; y++) {
                        for (int x = 0; x < L; x++) {
                            size_t site = geo.index(x, y, z, t);
                            size_t copied_site = geo.index(x, y, z + L_shift, t);
                            for (int mu = 0; mu < 4; mu++) {
                                field.view_link(site, mu) = field.view_link_const(copied_site, mu);
                            }
                        }
                    }
                }
            }
        }

        if (coord == T) {
            for (int t = 0; t < L - L_shift; t++) {
                for (int z = 0; z < L; z++) {
                    for (int y = 0; y < L; y++) {
                        for (int x = 0; x < L; x++) {
                            size_t site = geo.index(x, y, z, t);
                            size_t copied_site = geo.index(x, y, z, t + L_shift);
                            for (int mu = 0; mu < 4; mu++) {
                                field.view_link(site, mu) = field.view_link_const(copied_site, mu);
                            }
                        }
                    }
                }
            }
        }
    }
}

// Fills the halo_recv of the dest node with the content of the halo_send of the source node
void mpi::shiftcb::exchange_halos(HalosShift& halo, mpi::MpiTopology& topo, const ShiftParams& sp,
                                  MPI_Request* reqs) {
    int source{}, dest{};
    if (sp.coord == X) {
        if (sp.stype == pos) {
            source = topo.x0;
            dest = topo.xL;
        }
        if (sp.stype == neg) {
            source = topo.xL;
            dest = topo.x0;
        }
    }
    if (sp.coord == Y) {
        if (sp.stype == pos) {
            source = topo.y0;
            dest = topo.yL;
        }
        if (sp.stype == neg) {
            source = topo.yL;
            dest = topo.y0;
        }
    }
    if (sp.coord == Z) {
        if (sp.stype == pos) {
            source = topo.z0;
            dest = topo.zL;
        }
        if (sp.stype == neg) {
            source = topo.zL;
            dest = topo.z0;
        }
    }
    if (sp.coord == T) {
        if (sp.stype == pos) {
            source = topo.t0;
            dest = topo.tL;
        }
        if (sp.stype == neg) {
            source = topo.tL;
            dest = topo.t0;
        }
    }
    // Non blocking exchange -> we can shift during the exchanges
    MPI_Irecv(halo.recv.data(), 2 * 9 * 4 * halo.V_halo, MPI_DOUBLE, source, 10, topo.cart_comm,
              &reqs[0]);
    MPI_Isend(halo.send.data(), 2 * 9 * 4 * halo.V_halo, MPI_DOUBLE, dest, 10, topo.cart_comm,
              &reqs[1]);

    // MPI_Sendrecv(halo.send.data(), 2 * 9 * 4 * halo.V_halo, MPI_DOUBLE, dest, 0,
    //     halo.recv.data(), 2 * 9 * 4 * halo.V_halo,MPI_DOUBLE, source, 0,
    //     topo.cart_comm, MPI_STATUS_IGNORE);
}

// Replace the values of the corresponding links of the lattice with those of halo_rec
void mpi::shiftcb::fill_lattice_with_halo_recv(GaugeField& field, const GeometryCB& geo,
                                               HalosShift& halo, const ShiftParams& sp) {
    int L = geo.L;
    int L_shift = halo.L_shift;
    // If positive shift, the local coordinates of the halo corresponds to the coordinates of the
    // lattice
    if (sp.stype == pos) {
        if (sp.coord == X) {
            for (int t = 0; t < L; t++) {
                for (int z = 0; z < L; z++) {
                    for (int y = 0; y < L; y++) {
                        for (int x = 0; x < L_shift; x++) {
                            size_t index_field = geo.index(x, y, z, t);
                            size_t index_halo = halo.index_halo(x, y, z, t, sp);
                            for (int mu = 0; mu < 4; mu++) {
                                field.view_link(index_field, mu) =
                                    halo.view_halo_rec_const(index_halo, mu);
                            }
                        }
                    }
                }
            }
        }
        if (sp.coord == Y) {
            for (int t = 0; t < L; t++) {
                for (int z = 0; z < L; z++) {
                    for (int y = 0; y < L_shift; y++) {
                        for (int x = 0; x < L; x++) {
                            size_t index_field = geo.index(x, y, z, t);
                            size_t index_halo = halo.index_halo(x, y, z, t, sp);
                            for (int mu = 0; mu < 4; mu++) {
                                field.view_link(index_field, mu) =
                                    halo.view_halo_rec_const(index_halo, mu);
                            }
                        }
                    }
                }
            }
        }
        if (sp.coord == Z) {
            for (int t = 0; t < L; t++) {
                for (int z = 0; z < L_shift; z++) {
                    for (int y = 0; y < L; y++) {
                        for (int x = 0; x < L; x++) {
                            size_t index_field = geo.index(x, y, z, t);
                            size_t index_halo = halo.index_halo(x, y, z, t, sp);
                            for (int mu = 0; mu < 4; mu++) {
                                field.view_link(index_field, mu) =
                                    halo.view_halo_rec_const(index_halo, mu);
                            }
                        }
                    }
                }
            }
        }
        if (sp.coord == T) {
            for (int t = 0; t < L_shift; t++) {
                for (int z = 0; z < L; z++) {
                    for (int y = 0; y < L; y++) {
                        for (int x = 0; x < L; x++) {
                            size_t index_field = geo.index(x, y, z, t);
                            size_t index_halo = halo.index_halo(x, y, z, t, sp);
                            for (int mu = 0; mu < 4; mu++) {
                                field.view_link(index_field, mu) =
                                    halo.view_halo_rec_const(index_halo, mu);
                            }
                        }
                    }
                }
            }
        }
    }
    // If negative shift, the coordinates of the halo are shifted of L-L_shift to get the correct
    // coordinates on the field
    if (sp.stype == neg) {
        if (sp.coord == X) {
            for (int t = 0; t < L; t++) {
                for (int z = 0; z < L; z++) {
                    for (int y = 0; y < L; y++) {
                        for (int x = 0; x < L_shift; x++) {
                            size_t index_field = geo.index(x + L - L_shift, y, z, t);
                            size_t index_halo = halo.index_halo(x, y, z, t, sp);
                            for (int mu = 0; mu < 4; mu++) {
                                field.view_link(index_field, mu) =
                                    halo.view_halo_rec_const(index_halo, mu);
                            }
                        }
                    }
                }
            }
        }

        if (sp.coord == Y) {
            for (int t = 0; t < L; t++) {
                for (int z = 0; z < L; z++) {
                    for (int y = 0; y < L_shift; y++) {
                        for (int x = 0; x < L; x++) {
                            size_t index_field = geo.index(x, y + L - L_shift, z, t);
                            size_t index_halo = halo.index_halo(x, y, z, t, sp);
                            for (int mu = 0; mu < 4; mu++) {
                                field.view_link(index_field, mu) =
                                    halo.view_halo_rec_const(index_halo, mu);
                            }
                        }
                    }
                }
            }
        }

        if (sp.coord == Z) {
            for (int t = 0; t < L; t++) {
                for (int z = 0; z < L_shift; z++) {
                    for (int y = 0; y < L; y++) {
                        for (int x = 0; x < L; x++) {
                            size_t index_field = geo.index(x, y, z + L - L_shift, t);
                            size_t index_halo = halo.index_halo(x, y, z, t, sp);
                            for (int mu = 0; mu < 4; mu++) {
                                field.view_link(index_field, mu) =
                                    halo.view_halo_rec_const(index_halo, mu);
                            }
                        }
                    }
                }
            }
        }

        if (sp.coord == T) {
            for (int t = 0; t < L_shift; t++) {
                for (int z = 0; z < L; z++) {
                    for (int y = 0; y < L; y++) {
                        for (int x = 0; x < L; x++) {
                            size_t index_field = geo.index(x, y, z, t + L - L_shift);
                            size_t index_halo = halo.index_halo(x, y, z, t, sp);
                            for (int mu = 0; mu < 4; mu++) {
                                field.view_link(index_field, mu) =
                                    halo.view_halo_rec_const(index_halo, mu);
                            }
                        }
                    }
                }
            }
        }
    }
}

// Performs a shift in every node in direction coord, of type stype (positive of negative) and of
// length L_shift
void mpi::shiftcb::shift(GaugeField& field, const GeometryCB& geo, HalosShift& halo,
                         MpiTopology& topo, const ShiftParams& sp) {
    if (halo.L_shift < sp.L_shift) {
        if (topo.rank == 0) std::cerr << "Wrong halo size : too small\n";
        return;
    }
    if (sp.coord == UNSET) {
        if (topo.rank == 0) std::cerr << "UNSET coordinate for shift in ShiftParam\n";
        return;
    }
    if (sp.stype == unset) {
        if (topo.rank == 0) std::cerr << "unset shift type for shift in ShiftParam\n";
        return;
    }
    if (sp.L_shift <= 0) {
        if (topo.rank == 0) std::cerr << "L_shift too small : " << sp.L_shift << "\n";
    }
    MPI_Request reqs[2];
    fill_halo_send(field, geo, halo, sp);
    exchange_halos(halo, topo, sp, reqs);
    shift_field(field, geo, halo, sp);
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    fill_lattice_with_halo_recv(field, geo, halo, sp);
}

// Fills the halos with links of coord 0 or L-1 depending
void mpi::ecmccb::fill_halos_ecmc(const GaugeField& field, const GeometryCB& geo, HalosCB& halo) {
    int L = geo.L;
    for (int c3 = 0; c3 < geo.L; c3++) {
        for (int c2 = 0; c2 < geo.L; c2++) {
            for (int c1 = 0; c1 < geo.L; c1++) {
                size_t fieldx0 = geo.index(0, c1, c2, c3);
                size_t fieldy0 = geo.index(c1, 0, c2, c3);
                size_t fieldz0 = geo.index(c1, c2, 0, c3);
                size_t fieldt0 = geo.index(c1, c2, c3, 0);
                size_t fieldxL = geo.index(L - 1, c1, c2, c3);
                size_t fieldyL = geo.index(c1, L - 1, c2, c3);
                size_t fieldzL = geo.index(c1, c2, L - 1, c3);
                size_t fieldtL = geo.index(c1, c2, c3, L - 1);
                size_t site_halo = geo.index_halo_ecmc(c1, c2, c3);
                for (int mu = 0; mu < 4; mu++) {
                    halo.view_link(fx0, site_halo, mu) = field.view_link_const(fieldx0, mu);
                    halo.view_link(fy0, site_halo, mu) = field.view_link_const(fieldy0, mu);
                    halo.view_link(fz0, site_halo, mu) = field.view_link_const(fieldz0, mu);
                    halo.view_link(ft0, site_halo, mu) = field.view_link_const(fieldt0, mu);
                    halo.view_link(fxL, site_halo, mu) = field.view_link_const(fieldxL, mu);
                    halo.view_link(fyL, site_halo, mu) = field.view_link_const(fieldyL, mu);
                    halo.view_link(fzL, site_halo, mu) = field.view_link_const(fieldzL, mu);
                    halo.view_link(ftL, site_halo, mu) = field.view_link_const(fieldtL, mu);
                }
            }
        }
    }
}

// Fills the halos in GaugeField with content of HalosCB of neighbors
void mpi::ecmccb::exchange_halos_ecmc(GaugeField& field, const GeometryCB& geo, const HalosCB& halo,
                                      mpi::MpiTopology& topo) {
    MPI_Request reqs[16];
    int L = geo.L;
    int V = geo.V;
    int V_halo = geo.V_halo;
    // Setting up the recv
    MPI_Irecv(&field.links[(V + 0 * V_halo) * 4 * 9], L * L * L * 4 * 9 * 2, MPI_DOUBLE,
              topo.xL, 0, topo.cart_comm, &reqs[0]);
    MPI_Irecv(&field.links[(V + 1 * V_halo) * 4 * 9], L * L * L * 4 * 9 * 2, MPI_DOUBLE,
              topo.yL, 1, topo.cart_comm, &reqs[1]);
    MPI_Irecv(&field.links[(V + 2 * V_halo) * 4 * 9], L * L * L * 4 * 9 * 2, MPI_DOUBLE,
              topo.zL, 2, topo.cart_comm, &reqs[2]);
    MPI_Irecv(&field.links[(V + 3 * V_halo) * 4 * 9], L * L * L * 4 * 9 * 2, MPI_DOUBLE,
              topo.tL, 3, topo.cart_comm, &reqs[3]);
    MPI_Irecv(&field.links[(V + 4 * V_halo) * 4 * 9], L * L * L * 4 * 9 * 2, MPI_DOUBLE,
              topo.x0, 4, topo.cart_comm, &reqs[4]);
    MPI_Irecv(&field.links[(V + 5 * V_halo) * 4 * 9], L * L * L * 4 * 9 * 2, MPI_DOUBLE,
              topo.y0, 5, topo.cart_comm, &reqs[5]);
    MPI_Irecv(&field.links[(V + 6 * V_halo) * 4 * 9], L * L * L * 4 * 9 * 2, MPI_DOUBLE,
              topo.z0, 6, topo.cart_comm, &reqs[6]);
    MPI_Irecv(&field.links[(V + 7 * V_halo) * 4 * 9], L * L * L * 4 * 9 * 2, MPI_DOUBLE,
              topo.t0, 7, topo.cart_comm, &reqs[7]);
    // Setting up the sends
    MPI_Isend(halo.x0.data(), L * L * L * 4 * 9 * 2, MPI_DOUBLE, topo.x0, 0, topo.cart_comm,
              &reqs[8]);
    MPI_Isend(halo.y0.data(), L * L * L * 4 * 9 * 2, MPI_DOUBLE, topo.y0, 1, topo.cart_comm,
              &reqs[9]);
    MPI_Isend(halo.z0.data(), L * L * L * 4 * 9 * 2, MPI_DOUBLE, topo.z0, 2, topo.cart_comm,
              &reqs[10]);
    MPI_Isend(halo.t0.data(), L * L * L * 4 * 9 * 2, MPI_DOUBLE, topo.t0, 3, topo.cart_comm,
              &reqs[11]);
    MPI_Isend(halo.xL.data(), L * L * L * 4 * 9 * 2, MPI_DOUBLE, topo.xL, 4, topo.cart_comm,
              &reqs[12]);
    MPI_Isend(halo.yL.data(), L * L * L * 4 * 9 * 2, MPI_DOUBLE, topo.yL, 5, topo.cart_comm,
              &reqs[13]);
    MPI_Isend(halo.zL.data(), L * L * L * 4 * 9 * 2, MPI_DOUBLE, topo.zL, 6, topo.cart_comm,
              &reqs[14]);
    MPI_Isend(halo.tL.data(), L * L * L * 4 * 9 * 2, MPI_DOUBLE, topo.tL, 7, topo.cart_comm,
              &reqs[15]);
    // Waiting for sync
    MPI_Waitall(16, reqs, MPI_STATUSES_IGNORE);
}

// Fills the halos send and exchanges with the field
void mpi::ecmccb::fill_and_exchange(GaugeField& field, const GeometryCB& geo, HalosCB& halo,
                                    mpi::MpiTopology& topo) {
    fill_halos_ecmc(field, geo, halo);
    exchange_halos_ecmc(field, geo, halo, topo);
}

// Sends and receive all the send buffers of halo_obs into the correct recv buffers
void mpi::haloobs::exchange_halos_obs(HaloObs& halo_obs, mpi::MpiTopology& topo,
                                      MPI_Request* reqs) {
    // Set up the recv buffers
    MPI_Irecv(halo_obs.fx0_recv.data(), halo_obs.V * 4 * 18, MPI_DOUBLE, topo.x0, 0, topo.cart_comm,
              &reqs[0]);
    MPI_Irecv(halo_obs.fxL_recv.data(), halo_obs.V * 4 * 18, MPI_DOUBLE, topo.xL, 1, topo.cart_comm,
              &reqs[1]);
    MPI_Irecv(halo_obs.fy0_recv.data(), halo_obs.V * 4 * 18, MPI_DOUBLE, topo.y0, 2, topo.cart_comm,
              &reqs[2]);
    MPI_Irecv(halo_obs.fyL_recv.data(), halo_obs.V * 4 * 18, MPI_DOUBLE, topo.yL, 3, topo.cart_comm,
              &reqs[3]);
    MPI_Irecv(halo_obs.fz0_recv.data(), halo_obs.V * 4 * 18, MPI_DOUBLE, topo.z0, 4, topo.cart_comm,
              &reqs[4]);
    MPI_Irecv(halo_obs.fzL_recv.data(), halo_obs.V * 4 * 18, MPI_DOUBLE, topo.zL, 5, topo.cart_comm,
              &reqs[5]);
    MPI_Irecv(halo_obs.ft0_recv.data(), halo_obs.V * 4 * 18, MPI_DOUBLE, topo.t0, 6, topo.cart_comm,
              &reqs[6]);
    MPI_Irecv(halo_obs.ftL_recv.data(), halo_obs.V * 4 * 18, MPI_DOUBLE, topo.tL, 7, topo.cart_comm,
              &reqs[7]);
    // Initiates the sends
    MPI_Isend(halo_obs.fxL_send.data(), halo_obs.V * 4 * 18, MPI_DOUBLE, topo.xL, 0, topo.cart_comm,
              &reqs[8]);
    MPI_Isend(halo_obs.fx0_send.data(), halo_obs.V * 4 * 18, MPI_DOUBLE, topo.x0, 1, topo.cart_comm,
              &reqs[9]);
    MPI_Isend(halo_obs.fyL_send.data(), halo_obs.V * 4 * 18, MPI_DOUBLE, topo.yL, 2, topo.cart_comm,
              &reqs[10]);
    MPI_Isend(halo_obs.fy0_send.data(), halo_obs.V * 4 * 18, MPI_DOUBLE, topo.y0, 3, topo.cart_comm,
              &reqs[11]);
    MPI_Isend(halo_obs.fzL_send.data(), halo_obs.V * 4 * 18, MPI_DOUBLE, topo.zL, 4, topo.cart_comm,
              &reqs[12]);
    MPI_Isend(halo_obs.fz0_send.data(), halo_obs.V * 4 * 18, MPI_DOUBLE, topo.z0, 5, topo.cart_comm,
              &reqs[13]);
    MPI_Isend(halo_obs.ftL_send.data(), halo_obs.V * 4 * 18, MPI_DOUBLE, topo.tL, 6, topo.cart_comm,
              &reqs[14]);
    MPI_Isend(halo_obs.ft0_send.data(), halo_obs.V * 4 * 18, MPI_DOUBLE, topo.t0, 7, topo.cart_comm,
              &reqs[15]);
    // The Wait_all is after the computation of the observable on the bulk of the lattice to
    // optimize data flux
}

// Fills the send buffers of halo_obs with the corresponding faces of the gauge field
void mpi::haloobs::fill_halo_obs_send(const GaugeField& field, const GeometryCB& geo,
                                      HaloObs& halo_obs) {
    int L = geo.L;
    for (int c3 = 0; c3 < L; c3++) {
        for (int c2 = 0; c2 < L; c2++) {
            for (int c1 = 0; c1 < L; c1++) {
                size_t site_halo = halo_obs.index_halo_obs(c1, c2, c3);
                size_t sitefx0 = geo.index(0, c1, c2, c3);
                size_t sitefxL = geo.index(L - 1, c1, c2, c3);
                size_t sitefy0 = geo.index(c1, 0, c2, c3);
                size_t sitefyL = geo.index(c1, L - 1, c2, c3);
                size_t sitefz0 = geo.index(c1, c2, 0, c3);
                size_t sitefzL = geo.index(c1, c2, L - 1, c3);
                size_t siteft0 = geo.index(c1, c2, c3, 0);
                size_t siteftL = geo.index(c1, c2, c3, L - 1);
                for (int mu = 0; mu < 4; mu++) {
                    halo_obs.view_link_halo_obs(fx0, send, site_halo, mu) =
                        field.view_link_const(sitefx0, mu);
                    halo_obs.view_link_halo_obs(fxL, send, site_halo, mu) =
                        field.view_link_const(sitefxL, mu);
                    halo_obs.view_link_halo_obs(fy0, send, site_halo, mu) =
                        field.view_link_const(sitefy0, mu);
                    halo_obs.view_link_halo_obs(fyL, send, site_halo, mu) =
                        field.view_link_const(sitefyL, mu);
                    halo_obs.view_link_halo_obs(fz0, send, site_halo, mu) =
                        field.view_link_const(sitefz0, mu);
                    halo_obs.view_link_halo_obs(fzL, send, site_halo, mu) =
                        field.view_link_const(sitefzL, mu);
                    halo_obs.view_link_halo_obs(ft0, send, site_halo, mu) =
                        field.view_link_const(siteft0, mu);
                    halo_obs.view_link_halo_obs(ftL, send, site_halo, mu) =
                        field.view_link_const(siteftL, mu);
                }
            }
        }
    }
}
