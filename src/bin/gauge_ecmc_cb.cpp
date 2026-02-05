#include <iostream>
#include "../gauge/GaugeField.h"

int main(int argc, char* argv[]){
    int L = 5;
    GeometryCB geo(L);
    GaugeField field(geo);
    std::mt19937_64 rng(123);
    field.hot_start(rng);
    std::cout << field.view_link_const(12, 0);
}
