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

//Halos used to compute observables (mean plaquette for instance)
class HaloObs {
public:
    std::vector<Complex> fx0_recv; //Halo containing the face x=L-1 of neighbor in direction x-1
    std::vector<Complex> fxL_recv; //Halo containing the face x=0 of neighbor in direction x+1
    std::vector<Complex> fy0_recv; //Halo containing the face y=L-1 of neighbor in direction y-1
    std::vector<Complex> fyL_recv; //Halo containing the face y=0 of neighbor in direction y+1
    std::vector<Complex> fz0_recv; //Halo containing the face z=L-1 of neighbor in direction z-1
    std::vector<Complex> fzL_recv; //Halo containing the face z=0 of neighbor in direction z+1
    std::vector<Complex> ft0_recv; //Halo containing the face t=L-1 of neighbor in direction t-1
    std::vector<Complex> ftL_recv; //Halo containing the face t=0 of neighbor in direction t+1

    std::vector<Complex> fx0_send; //Halo containing the face x=0 of the current node
    std::vector<Complex> fxL_send; //Halo containing the face x=L-1 of the current node
    std::vector<Complex> fy0_send; //Halo containing the face y=0 of the current node
    std::vector<Complex> fyL_send; //Halo containing the face y=L-1 of the current node
    std::vector<Complex> fz0_send; //Halo containing the face z=0 of the current node
    std::vector<Complex> fzL_send; //Halo containing the face z=L-1 of the current node
    std::vector<Complex> ft0_send; //Halo containing the face t=0 of the current node
    std::vector<Complex> ftL_send; //Halo containing the face t=L-1 of the current node

    int L;
    int V;

    explicit HaloObs(const GeometryCB &geo) {
        L = geo.L;
        V = L*L*L;
        fx0_recv.resize(L*L*L*9*4);
        fxL_recv.resize(L*L*L*9*4);
        fy0_recv.resize(L*L*L*9*4);
        fyL_recv.resize(L*L*L*9*4);
        fz0_recv.resize(L*L*L*9*4);
        fzL_recv.resize(L*L*L*9*4);
        ft0_recv.resize(L*L*L*9*4);
        ftL_recv.resize(L*L*L*9*4);

        fx0_send.resize(L*L*L*9*4);
        fxL_send.resize(L*L*L*9*4);
        fy0_send.resize(L*L*L*9*4);
        fyL_send.resize(L*L*L*9*4);
        fz0_send.resize(L*L*L*9*4);
        fzL_send.resize(L*L*L*9*4);
        ft0_send.resize(L*L*L*9*4);
        ftL_send.resize(L*L*L*9*4);
    }

    //Returns the index of the site corresponding to (c1,c2,c3) with order x,y,z,t
    [[nodiscard]] size_t index_halo_obs(int c1, int c2, int c3) const {
        return (c3*L + c2)*L+c1;
    }

    //Non const mapping of the halos of HaloObs instance to SU3 matrices
    [[nodiscard]] Eigen::Map<SU3> view_link_halo_obs(face f, buf b, size_t site, int mu) {
        if (b == send) {
            if (f == fx0) return Eigen::Map<SU3>(&fx0_send[(site * 4 + mu) * 9]);
            if (f == fxL) return Eigen::Map<SU3>(&fxL_send[(site * 4 + mu) * 9]);
            if (f == fy0) return Eigen::Map<SU3>(&fy0_send[(site * 4 + mu) * 9]);
            if (f == fyL) return Eigen::Map<SU3>(&fyL_send[(site * 4 + mu) * 9]);
            if (f == fz0) return Eigen::Map<SU3>(&fz0_send[(site * 4 + mu) * 9]);
            if (f == fzL) return Eigen::Map<SU3>(&fzL_send[(site * 4 + mu) * 9]);
            if (f == ft0) return Eigen::Map<SU3>(&ft0_send[(site * 4 + mu) * 9]);
            if (f == ftL) return Eigen::Map<SU3>(&ftL_send[(site * 4 + mu) * 9]);
        }
        if (b==recv) {
            if (f == fx0) return Eigen::Map<SU3>(&fx0_recv[(site * 4 + mu) * 9]);
            if (f == fxL) return Eigen::Map<SU3>(&fxL_recv[(site * 4 + mu) * 9]);
            if (f == fy0) return Eigen::Map<SU3>(&fy0_recv[(site * 4 + mu) * 9]);
            if (f == fyL) return Eigen::Map<SU3>(&fyL_recv[(site * 4 + mu) * 9]);
            if (f == fz0) return Eigen::Map<SU3>(&fz0_recv[(site * 4 + mu) * 9]);
            if (f == fzL) return Eigen::Map<SU3>(&fzL_recv[(site * 4 + mu) * 9]);
            if (f == ft0) return Eigen::Map<SU3>(&ft0_recv[(site * 4 + mu) * 9]);
            if (f == ftL) return Eigen::Map<SU3>(&ftL_recv[(site * 4 + mu) * 9]);
        }
        std::cerr << "Wrong access\n";
        return Eigen::Map<SU3>(nullptr);
    }

    //Const mapping of the halos of HaloObs instance to SU3 matrices
    [[nodiscard]] Eigen::Map<const SU3> view_link_halo_obs_const(face f, buf b, size_t site, int mu) const {
        if (b == send) {
            if (f == fx0) return Eigen::Map<SU3 const>(&fx0_send[(site * 4 + mu) * 9]);
            if (f == fxL) return Eigen::Map<SU3 const>(&fxL_send[(site * 4 + mu) * 9]);
            if (f == fy0) return Eigen::Map<SU3 const>(&fy0_send[(site * 4 + mu) * 9]);
            if (f == fyL) return Eigen::Map<SU3 const>(&fyL_send[(site * 4 + mu) * 9]);
            if (f == fz0) return Eigen::Map<SU3 const>(&fz0_send[(site * 4 + mu) * 9]);
            if (f == fzL) return Eigen::Map<SU3 const>(&fzL_send[(site * 4 + mu) * 9]);
            if (f == ft0) return Eigen::Map<SU3 const>(&ft0_send[(site * 4 + mu) * 9]);
            if (f == ftL) return Eigen::Map<SU3 const>(&ftL_send[(site * 4 + mu) * 9]);
        }
        if (b==recv) {
            if (f == fx0) return Eigen::Map<SU3 const>(&fx0_recv[(site * 4 + mu) * 9]);
            if (f == fxL) return Eigen::Map<SU3 const>(&fxL_recv[(site * 4 + mu) * 9]);
            if (f == fy0) return Eigen::Map<SU3 const>(&fy0_recv[(site * 4 + mu) * 9]);
            if (f == fyL) return Eigen::Map<SU3 const>(&fyL_recv[(site * 4 + mu) * 9]);
            if (f == fz0) return Eigen::Map<SU3 const>(&fz0_recv[(site * 4 + mu) * 9]);
            if (f == fzL) return Eigen::Map<SU3 const>(&fzL_recv[(site * 4 + mu) * 9]);
            if (f == ft0) return Eigen::Map<SU3 const>(&ft0_recv[(site * 4 + mu) * 9]);
            if (f == ftL) return Eigen::Map<SU3 const>(&ftL_recv[(site * 4 + mu) * 9]);
        }
        std::cerr << "Wrong access\n";
        return Eigen::Map<const SU3>(nullptr);
    }

    //Returns the link at (x,y,z,t,mu) taking into account the halos_obs
    [[nodiscard]] SU3 get_link_with_halo_obs(const GaugeField &field, const GeometryCB &geo, int x, int y, int z, int t, int mu) const {
        if (x>=0 && y>=0 && z>=0 && t>=0 && x<=L-1 && y<=L-1 && z<=L-1 && t<=L-1) {
            size_t site = geo.index(x,y,z,t);
            return field.view_link_const(site, mu);
        }
        size_t site_halo{};
        face f{};
        if (x==-1) {
            site_halo = index_halo_obs(y,z,t);
            f = fx0;
        }
        else if (y==-1) {
            site_halo = index_halo_obs(x,z,t);
            f = fy0;
        }
        else if (z==-1) {
            site_halo = index_halo_obs(x,y,t);
            f = fz0;
        }
        else if (t==-1) {
            site_halo = index_halo_obs(x,y,z);
            f = ft0;
        }
        else if (x ==L) {
            site_halo = index_halo_obs(y,z,t);
            f = fxL;
        }
        else if (y ==L) {
            site_halo = index_halo_obs(x,z,t);
            f = fyL;
        }
        else if (z ==L) {
            site_halo = index_halo_obs(x,y,t);
            f = fzL;
        }
        else if (t ==L) {
            site_halo = index_halo_obs(x,y,z);
            f = ftL;
        }
        return view_link_halo_obs_const(f, recv, site_halo, mu);
    }
};

#endif //INC_4D_MPI_HALO_H
