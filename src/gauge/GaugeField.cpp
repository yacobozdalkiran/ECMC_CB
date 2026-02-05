//
// Created by ozdalkiran-l on 1/8/26.
//

#include "GaugeField.h"
#include "../su3/utils.h"


//Initialises the gauge field with random SU3 matrices
void GaugeField::hot_start(std::mt19937_64 &rng) {
    for (size_t site = 0; site < V; site++) {
        for (int mu = 0; mu < 4; mu++) {
            view_link(site, mu) = random_su3(rng);
        }
    }
}

//Initialises the gauge field with identity matrices
void GaugeField::cold_start() {
    for (size_t site = 0; site < V; site++) {
        for (int mu = 0; mu < 4; mu++) {
            view_link(site, mu) = Eigen::Matrix3cd::Identity();
        }
    }
}

//Projects a link on SU3 using Gramm-Schmidt
void GaugeField::projection_su3(size_t site, int mu) {
    auto U = view_link(site, mu);

    SU3 temp = U;

    Eigen::Vector3cd c0 = temp.col(0);
    c0.normalize();

    Eigen::Vector3cd c1 = temp.col(1);
    c1 -= c0 * c0.dot(c1);
    c1.normalize();

    Eigen::Vector3cd c2;
    c2(0) = std::conj(c0(1)*c1(2) - c0(2)*c1(1));
    c2(1) = std::conj(c0(2)*c1(0) - c0(0)*c1(2));
    c2(2) = std::conj(c0(0)*c1(1) - c0(1)*c1(0));

    temp.col(0) = c0;
    temp.col(1) = c1;
    temp.col(2) = c2;

    U = temp;
}

//Projects the whole field on SU3
void GaugeField::project_field_su3() {
    for (size_t site=0; site<V; site++) {
        for (int mu=0; mu<4; mu++) {
            projection_su3(site, mu);
        }
    }
}
