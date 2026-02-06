#pragma once

#include "../gauge/GaugeField.h"
#include "../geometry/GeometryCB.h"
#include <iostream>
#include "types.h"

//Send halos used during ECMC with Checkboard MPI (we unpack directly in the field)
class HalosCB {
public:
    int L;
    std::vector<Complex> xL; //Halo containing the x=L links of current node
    std::vector<Complex> yL; //Halo containing the y=L links of current node
    std::vector<Complex> zL; //Halo containing the z=L links of current node
    std::vector<Complex> tL; //Halo containing the t=L links of current node
    std::vector<Complex> x0; //Halo containing the x=0 links of current node
    std::vector<Complex> y0; //Halo containing the y=0 links of current node
    std::vector<Complex> z0; //Halo containing the z=0 links of current node
    std::vector<Complex> t0; //Halo containing the t=0 links of current node


    explicit HalosCB(const GeometryCB& geo) {
        L = geo.L;
        xL.resize(L*L*L*4*9);
        yL.resize(L*L*L*4*9);
        zL.resize(L*L*L*4*9);
        tL.resize(L*L*L*4*9);
        x0.resize(L*L*L*4*9);
        y0.resize(L*L*L*4*9);
        z0.resize(L*L*L*4*9);
        t0.resize(L*L*L*4*9);
    }

    //Non const mapping of the halos to SU3 matrices
    [[nodiscard]] Eigen::Map<SU3> view_link(face f, size_t site, int mu) {
        if (f==fx0) return Eigen::Map<SU3>(&x0[(site*4+mu)*9]);
        if (f==fy0) return Eigen::Map<SU3>(&y0[(site*4+mu)*9]);
        if (f==fz0) return Eigen::Map<SU3>(&z0[(site*4+mu)*9]);
        if (f==ft0) return Eigen::Map<SU3>(&t0[(site*4+mu)*9]);
        if (f==fxL) return Eigen::Map<SU3>(&xL[(site*4+mu)*9]);
        if (f==fyL) return Eigen::Map<SU3>(&yL[(site*4+mu)*9]);
        if (f==fzL) return Eigen::Map<SU3>(&zL[(site*4+mu)*9]);
        if (f==ftL) return Eigen::Map<SU3>(&tL[(site*4+mu)*9]);
        std::cerr << "Wrong acces\n";
        return Eigen::Map<SU3>(nullptr);
    }

    //Const mapping of the halos to SU3 matrices
    [[nodiscard]] Eigen::Map<const SU3> view_link_const(face f, size_t site, int mu) const {
        if (f==fx0) return Eigen::Map<const SU3>(&x0[(site*4+mu)*9]);
        if (f==fy0) return Eigen::Map<const SU3>(&y0[(site*4+mu)*9]);
        if (f==fz0) return Eigen::Map<const SU3>(&z0[(site*4+mu)*9]);
        if (f==ft0) return Eigen::Map<const SU3>(&t0[(site*4+mu)*9]);
        if (f==fxL) return Eigen::Map<const SU3>(&xL[(site*4+mu)*9]);
        if (f==fyL) return Eigen::Map<const SU3>(&yL[(site*4+mu)*9]);
        if (f==fzL) return Eigen::Map<const SU3>(&zL[(site*4+mu)*9]);
        if (f==ftL) return Eigen::Map<const SU3>(&tL[(site*4+mu)*9]);
        std::cerr << "Wrong acces\n";
        return Eigen::Map<const SU3>(nullptr);
    }

};
