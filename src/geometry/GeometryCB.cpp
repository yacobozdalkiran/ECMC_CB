#include "GeometryCB.h"

#include <cstdint>

GeometryCB::GeometryCB(int L_) {
    L = L_;
    V = L * L * L * L;
    V_halo = L * L * L;
    neighbors.resize((V + 8 * V_halo) * 8, SIZE_MAX);
    for (int t = 0; t < L; t++) {
        for (int z = 0; z < L; z++) {
            for (int y = 0; y < L; y++) {
                for (int x = 0; x < L; x++) {
                    size_t i = index_w_halo(x, y, z, t);
                    // Interior (field)
                    if (x + 1 < L) neighbors[index_neigh(i, 0, up)] = index((x + 1), y, z, t);
                    if (x - 1 >= 0) neighbors[index_neigh(i, 0, down)] = index((x - 1), y, z, t);
                    if (y + 1 < L) neighbors[index_neigh(i, 1, up)] = index(x, (y + 1), z, t);
                    if (y - 1 >= 0) neighbors[index_neigh(i, 1, down)] = index(x, (y - 1), z, t);
                    if (z + 1 < L) neighbors[index_neigh(i, 2, up)] = index(x, y, (z + 1), t);
                    if (z - 1 >= 0) neighbors[index_neigh(i, 2, down)] = index(x, y, (z - 1), t);
                    if (t + 1 < L) neighbors[index_neigh(i, 3, up)] = index(x, y, z, (t + 1));
                    if (t - 1 >= 0) neighbors[index_neigh(i, 3, down)] = index(x, y, z, (t - 1));
                    // Boundaries (halos)
                    if (x + 1 == L) neighbors[index_neigh(i, 0, up)] = V + index_halo_ecmc(y, z, t);
                    if (y + 1 == L)
                        neighbors[index_neigh(i, 1, up)] = V + V_halo + index_halo_ecmc(x, z, t);
                    if (z + 1 == L)
                        neighbors[index_neigh(i, 2, up)] =
                            V + 2 * V_halo + index_halo_ecmc(x, y, t);
                    if (t + 1 == L)
                        neighbors[index_neigh(i, 3, up)] =
                            V + 3 * V_halo + index_halo_ecmc(x, y, z);
                    if (x - 1 == -1)
                        neighbors[index_neigh(i, 0, down)] =
                            V + 4 * V_halo + index_halo_ecmc(y, z, t);
                    if (y - 1 == -1)
                        neighbors[index_neigh(i, 1, down)] =
                            V + 5 * V_halo + index_halo_ecmc(x, z, t);
                    if (z - 1 == -1)
                        neighbors[index_neigh(i, 2, down)] =
                            V + 6 * V_halo + index_halo_ecmc(x, y, t);
                    if (t - 1 == -1)
                        neighbors[index_neigh(i, 3, down)] =
                            V + 7 * V_halo + index_halo_ecmc(x, y, z);
                }
            }
        }
    }

    // Intra-halo
    for (int c1 = 0; c1 < L; c1++) {
        for (int c2 = 0; c2 < L; c2++) {
            for (int c3 = 0; c3 < L; c3++) {
                size_t i = index_w_halo(L, c1, c2, c3);
                neighbors[index_neigh(i, 0, down)] = index(L - 1, c1, c2, c3);
                if (c1 + 1 < L)
                    neighbors[index_neigh(i, 1, up)] = V + index_halo_ecmc(c1 + 1, c2, c3);
                if (c1 - 1 >= 0)
                    neighbors[index_neigh(i, 1, down)] = V + index_halo_ecmc(c1 - 1, c2, c3);
                if (c2 + 1 < L)
                    neighbors[index_neigh(i, 2, up)] = V + index_halo_ecmc(c1, c2 + 1, c3);
                if (c2 - 1 >= 0)
                    neighbors[index_neigh(i, 2, down)] = V + index_halo_ecmc(c1, c2 - 1, c3);
                if (c3 + 1 < L)
                    neighbors[index_neigh(i, 3, up)] = V + index_halo_ecmc(c1, c2, c3 + 1);
                if (c3 - 1 >= 0)
                    neighbors[index_neigh(i, 3, down)] = V + index_halo_ecmc(c1, c2, c3 - 1);

                i = index_w_halo(c1, L, c2, c3);
                neighbors[index_neigh(i, 1, down)] = index(c1, L - 1, c2, c3);
                if (c1 + 1 < L)
                    neighbors[index_neigh(i, 0, up)] = V + V_halo + index_halo_ecmc(c1 + 1, c2, c3);
                if (c1 - 1 >= 0)
                    neighbors[index_neigh(i, 0, down)] =
                        V + V_halo + index_halo_ecmc(c1 - 1, c2, c3);
                if (c2 + 1 < L)
                    neighbors[index_neigh(i, 2, up)] = V + V_halo + index_halo_ecmc(c1, c2 + 1, c3);
                if (c2 - 1 >= 0)
                    neighbors[index_neigh(i, 2, down)] =
                        V + V_halo + index_halo_ecmc(c1, c2 - 1, c3);
                if (c3 + 1 < L)
                    neighbors[index_neigh(i, 3, up)] = V + V_halo + index_halo_ecmc(c1, c2, c3 + 1);
                if (c3 - 1 >= 0)
                    neighbors[index_neigh(i, 3, down)] =
                        V + V_halo + index_halo_ecmc(c1, c2, c3 - 1);

                i = index_w_halo(c1, c2, L, c3);
                neighbors[index_neigh(i, 2, down)] = index(c1, c2, L - 1, c3);
                if (c1 + 1 < L)
                    neighbors[index_neigh(i, 0, up)] =
                        V + 2 * V_halo + index_halo_ecmc(c1 + 1, c2, c3);
                if (c1 - 1 >= 0)
                    neighbors[index_neigh(i, 0, down)] =
                        V + 2 * V_halo + index_halo_ecmc(c1 - 1, c2, c3);
                if (c2 + 1 < L)
                    neighbors[index_neigh(i, 1, up)] =
                        V + 2 * V_halo + index_halo_ecmc(c1, c2 + 1, c3);
                if (c2 - 1 >= 0)
                    neighbors[index_neigh(i, 1, down)] =
                        V + 2 * V_halo + index_halo_ecmc(c1, c2 - 1, c3);
                if (c3 + 1 < L)
                    neighbors[index_neigh(i, 3, up)] =
                        V + 2 * V_halo + index_halo_ecmc(c1, c2, c3 + 1);
                if (c3 - 1 >= 0)
                    neighbors[index_neigh(i, 3, down)] =
                        V + 2 * V_halo + index_halo_ecmc(c1, c2, c3 - 1);

                i = index_w_halo(c1, c2, c3, L - 1);
                neighbors[index_neigh(i, 3, down)] = index(c1, c2, c3, L);
                if (c1 + 1 < L)
                    neighbors[index_neigh(i, 0, up)] =
                        V + 3 * V_halo + index_halo_ecmc(c1 + 1, c2, c3);
                if (c1 - 1 >= 0)
                    neighbors[index_neigh(i, 0, down)] =
                        V + 3 * V_halo + index_halo_ecmc(c1 - 1, c2, c3);
                if (c2 + 1 < L)
                    neighbors[index_neigh(i, 1, up)] =
                        V + 3 * V_halo + index_halo_ecmc(c1, c2 + 1, c3);
                if (c2 - 1 >= 0)
                    neighbors[index_neigh(i, 1, down)] =
                        V + 3 * V_halo + index_halo_ecmc(c1, c2 - 1, c3);
                if (c3 + 1 < L)
                    neighbors[index_neigh(i, 2, up)] =
                        V + 3 * V_halo + index_halo_ecmc(c1, c2, c3 + 1);
                if (c3 - 1 >= 0)
                    neighbors[index_neigh(i, 2, down)] =
                        V + 3 * V_halo + index_halo_ecmc(c1, c2, c3 - 1);

                i = index_w_halo(-1, c1, c2, c3);
                neighbors[index_neigh(i, 0, up)] = index(0, c1, c2, c3);
                if (c1 + 1 < L)
                    neighbors[index_neigh(i, 1, up)] =
                        V + 4 * V_halo + index_halo_ecmc(c1 + 1, c2, c3);
                if (c1 - 1 >= 0)
                    neighbors[index_neigh(i, 1, down)] =
                        V + 4 * V_halo + index_halo_ecmc(c1 - 1, c2, c3);
                if (c2 + 1 < L)
                    neighbors[index_neigh(i, 2, up)] =
                        V + 4 * V_halo + index_halo_ecmc(c1, c2 + 1, c3);
                if (c2 - 1 >= 0)
                    neighbors[index_neigh(i, 2, down)] =
                        V + 4 * V_halo + index_halo_ecmc(c1, c2 - 1, c3);
                if (c3 + 1 < L)
                    neighbors[index_neigh(i, 3, up)] =
                        V + 4 * V_halo + index_halo_ecmc(c1, c2, c3 + 1);
                if (c3 - 1 >= 0)
                    neighbors[index_neigh(i, 3, down)] =
                        V + 4 * V_halo + index_halo_ecmc(c1, c2, c3 - 1);

                i = index_w_halo(c1, -1, c2, c3);
                neighbors[index_neigh(i, 1, up)] = index(c1, 0, c2, c3);
                if (c1 + 1 < L)
                    neighbors[index_neigh(i, 0, up)] =
                        V + 5 * V_halo + index_halo_ecmc(c1 + 1, c2, c3);
                if (c1 - 1 >= 0)
                    neighbors[index_neigh(i, 0, down)] =
                        V + 5 * V_halo + index_halo_ecmc(c1 - 1, c2, c3);
                if (c2 + 1 < L)
                    neighbors[index_neigh(i, 2, up)] =
                        V + 5 * V_halo + index_halo_ecmc(c1, c2 + 1, c3);
                if (c2 - 1 >= 0)
                    neighbors[index_neigh(i, 2, down)] =
                        V + 5 * V_halo + index_halo_ecmc(c1, c2 - 1, c3);
                if (c3 + 1 < L)
                    neighbors[index_neigh(i, 3, up)] =
                        V + 5 * V_halo + index_halo_ecmc(c1, c2, c3 + 1);
                if (c3 - 1 >= 0)
                    neighbors[index_neigh(i, 3, down)] =
                        V + 5 * V_halo + index_halo_ecmc(c1, c2, c3 - 1);

                i = index_w_halo(c1, c2, -1, c3);
                neighbors[index_neigh(i, 2, up)] = index(c1, c2, 0, c3);
                if (c1 + 1 < L)
                    neighbors[index_neigh(i, 0, up)] =
                        V + 6 * V_halo + index_halo_ecmc(c1 + 1, c2, c3);
                if (c1 - 1 >= 0)
                    neighbors[index_neigh(i, 0, down)] =
                        V + 6 * V_halo + index_halo_ecmc(c1 - 1, c2, c3);
                if (c2 + 1 < L)
                    neighbors[index_neigh(i, 1, up)] =
                        V + 6 * V_halo + index_halo_ecmc(c1, c2 + 1, c3);
                if (c2 - 1 >= 0)
                    neighbors[index_neigh(i, 1, down)] =
                        V + 6 * V_halo + index_halo_ecmc(c1, c2 - 1, c3);
                if (c3 + 1 < L)
                    neighbors[index_neigh(i, 3, up)] =
                        V + 6 * V_halo + index_halo_ecmc(c1, c2, c3 + 1);
                if (c3 - 1 >= 0)
                    neighbors[index_neigh(i, 3, down)] =
                        V + 6 * V_halo + index_halo_ecmc(c1, c2, c3 - 1);

                i = index_w_halo(c1, c2, c3, -1);
                neighbors[index_neigh(i, 3, up)] = index(c1, c2, c3, 0);
                if (c1 + 1 < L)
                    neighbors[index_neigh(i, 0, up)] =
                        V + 7 * V_halo + index_halo_ecmc(c1 + 1, c2, c3);
                if (c1 - 1 >= 0)
                    neighbors[index_neigh(i, 0, down)] =
                        V + 7 * V_halo + index_halo_ecmc(c1 - 1, c2, c3);
                if (c2 + 1 < L)
                    neighbors[index_neigh(i, 1, up)] =
                        V + 7 * V_halo + index_halo_ecmc(c1, c2 + 1, c3);
                if (c2 - 1 >= 0)
                    neighbors[index_neigh(i, 1, down)] =
                        V + 7 * V_halo + index_halo_ecmc(c1, c2 - 1, c3);
                if (c3 + 1 < L)
                    neighbors[index_neigh(i, 2, up)] =
                        V + 7 * V_halo + index_halo_ecmc(c1, c2, c3 + 1);
                if (c3 - 1 >= 0)
                    neighbors[index_neigh(i, 2, down)] =
                        V + 7 * V_halo + index_halo_ecmc(c1, c2, c3 - 1);
            }
        }
    }

    frozen.resize((V + 8 * V_halo) * 4, false);
    // Frozen links are those that step out of the lattice core or belong to halos
    for (int t = -1; t <= L; t++) {
        for (int z = -1; z <= L; z++) {
            for (int y = -1; y <= L; y++) {
                for (int x = -1; x <= L; x++) {
                    bool link_is_frozen = false;
                    if (x == -1 or y == -1 or z == -1 or t == -1) link_is_frozen = true;
                    if (x == L or y == L or z == L or t == L) link_is_frozen = true;
                    for (int mu = 0; mu < 4; mu++) {
                        if (x == L - 1 and mu == 0) link_is_frozen = true;
                        if (y == L - 1 and mu == 1) link_is_frozen = true;
                        if (z == L - 1 and mu == 2) link_is_frozen = true;
                        if (t == L - 1 and mu == 3) link_is_frozen = true;
                        size_t i = index_w_halo(x, y, z, t);
                        frozen[index_frozen(i, mu)] = link_is_frozen;
                    }
                }
            }
        }
    }

    links_staples.resize(V * 4 * 6 * 3, std::make_pair(SIZE_MAX, -1));
    // 4 links per site, 6 staples per link, 3 links per staple
    for (int t = 0; t < L; t++) {
        for (int z = 0; z < L; z++) {
            for (int y = 0; y < L; y++) {
                for (int x = 0; x < L; x++) {
                    size_t site = index(x, y, z, t);  // x
                    for (int mu = 0; mu < 4; mu++) {
                        if (!is_frozen(site, mu)) {
                            int j = 0;
                            for (int nu = 0; nu < 4; nu++) {
                                if (nu == mu) continue;

                                size_t xmu = get_neigh(site, mu, up);     // x+mu
                                size_t xnu = get_neigh(site, nu, up);     // x+nu
                                size_t xmunu = get_neigh(xmu, nu, down);  // x+mu-nu
                                size_t xmnu = get_neigh(site, nu, down);  // x-nu

                                links_staples[index_staples(site, mu, j, 0)] = {xmu, nu};
                                links_staples[index_staples(site, mu, j, 1)] = {xnu, mu};
                                links_staples[index_staples(site, mu, j, 2)] = {site, nu};
                                links_staples[index_staples(site, mu, j + 1, 0)] = {xmunu, nu};
                                links_staples[index_staples(site, mu, j + 1, 1)] = {xmnu, mu};
                                links_staples[index_staples(site, mu, j + 1, 2)] = {xmnu, nu};

                                j += 2;
                            }
                        }
                    }
                }
            }
        }
    }
}
