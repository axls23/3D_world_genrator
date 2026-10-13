// Device side of the stream's ACE -> 3DGUT pose path (hypersplat/stream/cuda/__init__.py).
//
// apply_deltas: one thread per DELTA record (hypersplat/stream/ring.py DELTA_DTYPE, 192 B),
// all records of one launch on distinct slots (the host splits repeats into later launches,
// so per-slot order is kept). Per record:
//   1. stale guard     map_version older than the slot's -> dropped
//   2. registration    conf below threshold or no pose -> slot unregistered (tier 0)
//   3. move            camera centre distance to the slot's filtered pose / path extent
//   4. pose filter     c2w_f <- R_f Exp(a Log(R_f^T R)), c_f + a (c - c_f)  (a = 1: off)
//                      an outlier move (> outlier_thr) resets the filter to the measurement:
//                      a real correction passes at once, jitter is averaged
//   5. tier FSM        new 0 -> provisional 1 -> stable 2 (stable_after moves <= promote_thr
//                      in a row) ; outlier demotes stable -> provisional ; anchored 3 (sidecar)
//   6. trainer pose    c2w_trainer = normalize(M @ c2w_f) as datasets/normalize.py
//                      transform_cameras does (rotation rows rescaled to unit norm), float32
// Thresholds are inputs: the host keeps the running jitter statistics of the moves this
// kernel reports. Everything in double: a handful of records per step, FP64 rate is moot.

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>

namespace {

struct __align__(8) DeltaRec {   // must match ring.DELTA_DTYPE
    uint64_t seq;
    int64_t step;
    int64_t kid;
    int64_t mapper_tick;
    int32_t map_version;
    uint8_t kind;
    uint8_t anchored;
    uint8_t pad0[2];
    float conf;
    float inlier_frac;
    float move;
    float focal;
    double w2c[16];
    int32_t n_seeds;
    uint8_t pad1[4];
};
static_assert(sizeof(DeltaRec) == 192, "DeltaRec layout");
static_assert(offsetof(DeltaRec, map_version) == 32, "DeltaRec layout");
static_assert(offsetof(DeltaRec, conf) == 40, "DeltaRec layout");
static_assert(offsetof(DeltaRec, w2c) == 56, "DeltaRec layout");
static_assert(offsetof(DeltaRec, n_seeds) == 184, "DeltaRec layout");

enum Result : int8_t { APPLIED = 0, STALE = 1, UNREGISTERED = 2, FROZEN = 3, OUTLIER = 4, BAD_SLOT = 5 };
enum Tier : uint8_t { NEW = 0, PROVISIONAL = 1, STABLE = 2, ANCHORED = 3 };

struct Params {
    double conf_threshold;
    double extent;         // camera path extent (ACE units); move = distance / extent
    double alpha;          // pose filter gain, 1 = no filtering
    double promote_thr;    // move <= this counts towards stable
    double outlier_thr;    // move > this: correction (filter reset, stable -> provisional)
    int stable_after;
    int capacity;
};

// Rotation (row-major 3x3) helpers -------------------------------------------------------
__device__ inline void mat3_mul(const double *A, const double *B, double *C) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            C[3 * r + c] = A[3 * r] * B[c] + A[3 * r + 1] * B[3 + c] + A[3 * r + 2] * B[6 + c];
}

__device__ inline void mat3_tmul(const double *A, const double *B, double *C) {   // A^T B
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            C[3 * r + c] = A[r] * B[c] + A[3 + r] * B[3 + c] + A[6 + r] * B[6 + c];
}

// SO(3) log -> axis * angle
__device__ inline void so3_log(const double *R, double *w) {
    const double tr = R[0] + R[4] + R[8];
    double c = 0.5 * (tr - 1.0);
    c = c > 1.0 ? 1.0 : (c < -1.0 ? -1.0 : c);
    const double th = acos(c);
    const double vx = R[7] - R[5], vy = R[2] - R[6], vz = R[3] - R[1];
    if (th < 1e-9) {
        w[0] = 0.5 * vx; w[1] = 0.5 * vy; w[2] = 0.5 * vz;
        return;
    }
    if (M_PI - th < 1e-6) {   // near pi: axis from the symmetric part
        double ax = sqrt(fmax(0.0, 0.5 * (R[0] + 1.0)));
        double ay = sqrt(fmax(0.0, 0.5 * (R[4] + 1.0)));
        double az = sqrt(fmax(0.0, 0.5 * (R[8] + 1.0)));
        if (R[1] + R[3] < 0) ay = -ay;
        if (R[2] + R[6] < 0) az = -az;
        w[0] = ax * th; w[1] = ay * th; w[2] = az * th;
        return;
    }
    const double k = th / (2.0 * sin(th));
    w[0] = k * vx; w[1] = k * vy; w[2] = k * vz;
}

// SO(3) exp (Rodrigues)
__device__ inline void so3_exp(const double *w, double *R) {
    const double th = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    double a, b;
    if (th < 1e-9) { a = 1.0; b = 0.5; }
    else { a = sin(th) / th; b = (1.0 - cos(th)) / (th * th); }
    const double x = w[0], y = w[1], z = w[2];
    R[0] = 1.0 - b * (y * y + z * z); R[1] = -a * z + b * x * y;        R[2] = a * y + b * x * z;
    R[3] = a * z + b * x * y;         R[4] = 1.0 - b * (x * x + z * z); R[5] = -a * x + b * y * z;
    R[6] = -a * y + b * x * z;        R[7] = a * x + b * y * z;         R[8] = 1.0 - b * (x * x + y * y);
}

__global__ void apply_deltas_kernel(const DeltaRec *__restrict__ recs, const int32_t *__restrict__ slots, int k,
                                    Params p, const double *__restrict__ M,
                                    uint8_t *has_pose, uint8_t *tier, uint8_t *stable_cnt, int32_t *ver,
                                    float *conf, double *c2w_f, float *c2w_trainer,
                                    float *out_move, int8_t *out_result, uint8_t *out_tier) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    const DeltaRec &r = recs[i];
    const int s = slots[i];
    out_move[i] = NAN;
    if (s < 0 || s >= p.capacity) { out_result[i] = BAD_SLOT; out_tier[i] = 0; return; }
    if (r.map_version < ver[s]) { out_result[i] = STALE; out_tier[i] = tier[s]; return; }
    ver[s] = r.map_version;

    bool finite = true;
    for (int j = 0; j < 16; ++j) finite = finite && isfinite(r.w2c[j]);
    if (!finite || r.conf < p.conf_threshold) {
        conf[s] = r.conf;
        has_pose[s] = 0;
        tier[s] = NEW;
        stable_cnt[s] = 0;
        out_result[i] = UNREGISTERED;
        out_tier[i] = NEW;
        return;
    }
    if (tier[s] == ANCHORED && has_pose[s]) { out_result[i] = FROZEN; out_tier[i] = ANCHORED; return; }

    // Measured c2w = inv(w2c) for a rigid w2c: R^T, -R^T t
    double Rw[9] = {r.w2c[0], r.w2c[1], r.w2c[2], r.w2c[4], r.w2c[5], r.w2c[6], r.w2c[8], r.w2c[9], r.w2c[10]};
    double Rm[9], cm[3];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) Rm[3 * a + b] = Rw[3 * b + a];
    for (int a = 0; a < 3; ++a)
        cm[a] = -(Rm[3 * a] * r.w2c[3] + Rm[3 * a + 1] * r.w2c[7] + Rm[3 * a + 2] * r.w2c[11]);

    double *F = c2w_f + 16 * s;
    double Rf[9], cf[3];
    int8_t result = APPLIED;
    float move = NAN;
    if (has_pose[s]) {
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) Rf[3 * a + b] = F[4 * a + b];
            cf[a] = F[4 * a + 3];
        }
        const double dx = cm[0] - cf[0], dy = cm[1] - cf[1], dz = cm[2] - cf[2];
        move = (float)(sqrt(dx * dx + dy * dy + dz * dz) / fmax(p.extent, 1e-12));
        if (move > p.outlier_thr) {
            result = OUTLIER;   // a correction, not jitter: take it whole
            for (int j = 0; j < 9; ++j) Rf[j] = Rm[j];
            for (int a = 0; a < 3; ++a) cf[a] = cm[a];
        } else {
            double D[9], w[3], E[9], Rn[9];
            mat3_tmul(Rf, Rm, D);
            so3_log(D, w);
            for (int a = 0; a < 3; ++a) w[a] *= p.alpha;
            so3_exp(w, E);
            mat3_mul(Rf, E, Rn);
            for (int j = 0; j < 9; ++j) Rf[j] = Rn[j];
            for (int a = 0; a < 3; ++a) cf[a] += p.alpha * (cm[a] - cf[a]);
        }
    } else {
        for (int j = 0; j < 9; ++j) Rf[j] = Rm[j];
        for (int a = 0; a < 3; ++a) cf[a] = cm[a];
    }

    // Tier FSM
    uint8_t t = tier[s];
    uint8_t sc = stable_cnt[s];
    if (r.anchored) {
        t = ANCHORED;
    } else if (!has_pose[s]) {
        t = PROVISIONAL; sc = 0;
    } else if (result == OUTLIER) {
        sc = 0;
        if (t == STABLE) t = PROVISIONAL;
    } else if (move <= p.promote_thr) {
        sc = sc < 255 ? sc + 1 : sc;
        if (sc >= p.stable_after && t < STABLE) t = STABLE;
    } else {
        sc = 0;
    }
    tier[s] = t;
    stable_cnt[s] = sc;
    has_pose[s] = 1;
    conf[s] = r.conf;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) F[4 * a + b] = Rf[3 * a + b];
        F[4 * a + 3] = cf[a];
    }
    F[12] = 0.0; F[13] = 0.0; F[14] = 0.0; F[15] = 1.0;

    // Trainer frame: M @ c2w_f, rotation rows rescaled by the norm of row 0 (normalize.py)
    double T[16];
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b)
            T[4 * a + b] = M[4 * a] * F[b] + M[4 * a + 1] * F[4 + b] + M[4 * a + 2] * F[8 + b] + M[4 * a + 3] * F[12 + b];
    const double sc0 = sqrt(T[0] * T[0] + T[1] * T[1] + T[2] * T[2]);
    float *out = c2w_trainer + 16 * s;
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b)
            out[4 * a + b] = (float)((a < 3 && b < 3) ? T[4 * a + b] / sc0 : T[4 * a + b]);

    out_move[i] = move;
    out_result[i] = result;
    out_tier[i] = t;
}

#define CHECK_CUDA_T(x, dt) \
    TORCH_CHECK((x).is_cuda() && (x).is_contiguous() && (x).scalar_type() == (dt), #x " must be a contiguous CUDA " #dt)

}  // namespace

// recs (k, 192) uint8, slots (k,) int32; state tensors are updated in place, outputs written.
void apply_deltas(torch::Tensor recs, torch::Tensor slots, torch::Tensor M,
                  torch::Tensor has_pose, torch::Tensor tier, torch::Tensor stable_cnt, torch::Tensor ver,
                  torch::Tensor conf, torch::Tensor c2w_f, torch::Tensor c2w_trainer,
                  torch::Tensor out_move, torch::Tensor out_result, torch::Tensor out_tier,
                  double conf_threshold, double extent, double alpha, double promote_thr,
                  double outlier_thr, int64_t stable_after) {
    CHECK_CUDA_T(recs, torch::kUInt8);
    TORCH_CHECK(recs.dim() == 2 && recs.size(1) == (int64_t)sizeof(DeltaRec), "recs must be (k, 192)");
    CHECK_CUDA_T(slots, torch::kInt32);
    CHECK_CUDA_T(M, torch::kFloat64);
    CHECK_CUDA_T(has_pose, torch::kUInt8);
    CHECK_CUDA_T(tier, torch::kUInt8);
    CHECK_CUDA_T(stable_cnt, torch::kUInt8);
    CHECK_CUDA_T(ver, torch::kInt32);
    CHECK_CUDA_T(conf, torch::kFloat32);
    CHECK_CUDA_T(c2w_f, torch::kFloat64);
    CHECK_CUDA_T(c2w_trainer, torch::kFloat32);
    CHECK_CUDA_T(out_move, torch::kFloat32);
    CHECK_CUDA_T(out_result, torch::kInt8);
    CHECK_CUDA_T(out_tier, torch::kUInt8);
    const int k = (int)recs.size(0);
    if (k == 0) return;
    TORCH_CHECK(slots.numel() == k && out_move.numel() >= k && out_result.numel() >= k && out_tier.numel() >= k,
                "slots / outputs must hold k entries");
    const at::cuda::CUDAGuard guard(recs.device());
    Params p{conf_threshold, extent, alpha, promote_thr, outlier_thr, (int)stable_after, (int)has_pose.numel()};
    const int threads = 128;
    apply_deltas_kernel<<<(k + threads - 1) / threads, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const DeltaRec *>(recs.data_ptr<uint8_t>()), slots.data_ptr<int32_t>(), k, p,
        M.data_ptr<double>(), has_pose.data_ptr<uint8_t>(), tier.data_ptr<uint8_t>(),
        stable_cnt.data_ptr<uint8_t>(), ver.data_ptr<int32_t>(), conf.data_ptr<float>(),
        c2w_f.data_ptr<double>(), c2w_trainer.data_ptr<float>(), out_move.data_ptr<float>(),
        out_result.data_ptr<int8_t>(), out_tier.data_ptr<uint8_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_deltas", &apply_deltas, "ACE pose deltas -> device pose/tier table (stream)");
}
