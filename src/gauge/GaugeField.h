//
// Created by ozdalkiran-l on 1/8/26.
//

#ifndef INC_4D_MPI_GAUGEFIELD_H
#define INC_4D_MPI_GAUGEFIELD_H

#include <Eigen/Dense>
#include <vector>
#include <complex>
#include <random>
#include "../geometry/GeometryCB.h"

using Complex = std::complex<double>;
using SU3 = Eigen::Matrix<std::complex<double>, 3, 3, Eigen::RowMajor>;

class GaugeField {
    int L;
    int T;
    size_t V;
    size_t V_halo;

public:
    std::vector<Complex> links;

    //Initializes a squared cold gauge conf with halos
    explicit GaugeField(const GeometryCB &geo):L(geo.L), T(geo.L), V(geo.V), V_halo(geo.V_halo), links((V+8*V_halo)*4*9, Complex(0.0,0.0)) {
        for (size_t site = 0; site < V; site++) {
            for (int mu = 0; mu<4; mu++) {
                view_link(site, mu) = SU3::Identity();
            }
        }
    }

    void hot_start(std::mt19937_64 &rng);
    void cold_start();

    //Non const mapping of links to SU3 matrices
    Eigen::Map<SU3> view_link(size_t site, int mu) {
        return Eigen::Map<SU3>(&links[(site * 4 + mu) * 9]);
    }

    //Const mapping of links to SU3 matrices
    [[nodiscard]] Eigen::Map<const SU3> view_link_const(size_t site, int mu) const{
        return Eigen::Map<const SU3>(&links[(site * 4 + mu) * 9]);
    }

    void projection_su3(size_t site, int mu);
    void project_field_su3();
};

#endif //INC_4D_MPI_GAUGEFIELD_H
