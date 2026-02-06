#pragma once

//Direction of the shift
enum shift_type {
    unset,
    pos, //if the shift is in direction +coord
    neg //if the shift is in direction -coord
};

//Possible coordinates of the shifts
enum halo_coord {
    UNSET,
    X,
    Y,
    Z,
    T
};

//Used to access the different buffers of HaloObs and HaloECMC
enum face {
    fx0, fxL,
    fy0, fyL,
    fz0, fzL,
    ft0, ftL,
};

//Used to acces the different buffers of HaloObs
enum buf {
    send,
    recv
};
