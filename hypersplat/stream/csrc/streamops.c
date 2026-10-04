/*
 * Native hot paths of the streaming pipeline (hypersplat/stream/native.py loads this via
 * ctypes; plain C99, no Python headers, so one .so serves every interpreter).
 *
 * pose_nn_distance
 *   For each flagged camera, the distance to its nearest other camera in pose space:
 *       |c_i - c_j| / spacing + angle(R_i, R_j) / angle_unit_deg
 *   (store.py's eviction redundancy). O(N^2) with a 3x3 trace per pair; replaces a Python
 *   double loop that took ~1.1 s for 400 keyframes.
 *
 * weighted_pick
 *   Inverse-CDF draw from unnormalized non-negative weights (the trainer's view sampler).
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define RAD2DEG 57.29577951308232

/* c2w: n row-major 4x4 camera-to-world matrices (rotation assumed orthonormal).
 * query: n flags; out[i] = nearest-neighbour distance for query[i] != 0, else NAN. */
void pose_nn_distance(const double *c2w, int64_t n, const uint8_t *query, double spacing,
                      double angle_unit_deg, double *out)
{
    const double inv_spacing = 1.0 / (spacing > 1e-12 ? spacing : 1e-12);
    const double inv_unit = RAD2DEG / (angle_unit_deg > 1e-12 ? angle_unit_deg : 1e-12);
    for (int64_t i = 0; i < n; ++i) {
        if (!query[i]) {
            out[i] = NAN;
            continue;
        }
        const double *a = c2w + 16 * i;
        double best = INFINITY;
        for (int64_t j = 0; j < n; ++j) {
            if (j == i)
                continue;
            const double *b = c2w + 16 * j;
            const double dx = a[3] - b[3], dy = a[7] - b[7], dz = a[11] - b[11];
            double d = sqrt(dx * dx + dy * dy + dz * dz) * inv_spacing;
            if (d >= best)
                continue; /* the angle term is non-negative: cannot improve */
            /* trace(Ra^T Rb) = sum_k sum_r Ra[r][k] * Rb[r][k] */
            double tr = 0.0;
            for (int r = 0; r < 3; ++r)
                for (int k = 0; k < 3; ++k)
                    tr += a[4 * r + k] * b[4 * r + k];
            double c = 0.5 * (tr - 1.0);
            c = c > 1.0 ? 1.0 : (c < -1.0 ? -1.0 : c);
            d += acos(c) * inv_unit;
            if (d < best)
                best = d;
        }
        out[i] = best;
    }
}

/* Index of the draw for uniform u in [0, 1); -1 if all weights are zero. */
int64_t weighted_pick(const double *w, int64_t n, double u)
{
    double total = 0.0;
    for (int64_t i = 0; i < n; ++i)
        total += w[i] > 0.0 ? w[i] : 0.0;
    if (total <= 0.0)
        return -1;
    const double target = u * total;
    double acc = 0.0;
    int64_t last = -1;
    for (int64_t i = 0; i < n; ++i) {
        if (w[i] <= 0.0)
            continue;
        acc += w[i];
        last = i;
        if (acc > target)
            return i;
    }
    return last; /* rounding at the top end */
}
