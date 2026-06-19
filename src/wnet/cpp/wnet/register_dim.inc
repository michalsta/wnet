// Per-dimension registration body. Included by each dim_register_<N>.cpp after
// it does `#define WNET_DIM <N>`. Compiling one of these per dimension is what
// lets ninja parallelize the (dominant) per-dim template instantiation cost.
//
// Not a standalone TU: WNET_DIM must be defined before inclusion.

#ifndef WNET_DIM
#error "register_dim.inc requires WNET_DIM to be defined (use a dim_register_<N>.cpp stub)"
#endif

#include "register_dim.hpp"

void REGISTER_DIM_NAME(WNET_DIM)(REGISTER_DIM_PARAMS) {
    // VectorDistribution (int + float intensity) and the four *Scaler classes.
    EXPOSE_VECTOR_DISTRIBUTION(WNET_DIM)

    // WassersteinNetworkFactory::create overloads (int + float intensity).
    factory
        .def_static("create", &WNetFactory::create<VectorDistribution<WNET_DIM, double, LEMON_INT>>,
            nb::arg("empirical_spectrum"), nb::arg("theoretical_spectra"),
            nb::arg("distance_metric"), nb::arg("max_dist"), nb::arg("p") = 1.0)
        .def_static("create", &WNetFactory::create<VectorDistribution<WNET_DIM, double, double>>,
            nb::arg("empirical_spectrum"), nb::arg("theoretical_spectra"),
            nb::arg("distance_metric"), nb::arg("max_dist"), nb::arg("p") = 1.0);

    // update_positions_and_solve / _and_get_gradient overloads on both networks.
    net_ii
        BIND_UPDATE_AND_SOLVE(WNetII, int64_t, WNET_DIM)
        BIND_UPDATE_AND_GET_GRADIENT(WNetII, int64_t, WNET_DIM);
    net_if
        BIND_UPDATE_AND_SOLVE(WNetIF, double, WNET_DIM)
        BIND_UPDATE_AND_GET_GRADIENT(WNetIF, double, WNET_DIM);
}
