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
 *
 * ring_*
 *   Single-producer / single-consumer ring of fixed-size records in shared memory (a /dev/shm
 *   file mapped by the ACE sidecar and the trainer, two interpreters and two torch builds).
 *   head is written only by the producer, tail only by the consumer, each on its own cache
 *   line; records are published with a release store of head and consumed after an acquire
 *   load of it, so a reader never sees a half-written record. Capacity is a power of two.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

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

/* ---------------------------------------------------------------- shared-memory ring */

#define RING_MAGIC 0x31474e4952534848ULL /* "HHSRING1" */
#define RING_HEADER 256

typedef struct {
    uint64_t magic;
    uint64_t rec_size;  /* bytes per record, multiple of 8 */
    uint64_t capacity;  /* records, power of two */
    uint8_t pad0[64 - 24];
    uint64_t head;      /* producer: records published */
    uint64_t full;      /* producer: records refused because the ring was full */
    uint8_t pad1[64 - 16];
    uint64_t tail;      /* consumer: records consumed */
    uint8_t pad2[64 - 8];
    uint8_t pad3[RING_HEADER - 192];
} RingHeader;

/* Bytes a ring of capacity records of rec_size bytes needs. */
int64_t ring_bytes(int64_t rec_size, int64_t capacity)
{
    return RING_HEADER + rec_size * capacity;
}

/* Initialize a zeroed mapping; returns 0, or -1 for a bad size / capacity. */
int64_t ring_init(void *base, int64_t bytes, int64_t rec_size, int64_t capacity)
{
    if (rec_size <= 0 || rec_size % 8 || capacity <= 0 || (capacity & (capacity - 1)) ||
        bytes < ring_bytes(rec_size, capacity))
        return -1;
    RingHeader *h = (RingHeader *)base;
    memset(h, 0, RING_HEADER);
    h->rec_size = (uint64_t)rec_size;
    h->capacity = (uint64_t)capacity;
    __atomic_store_n(&h->magic, RING_MAGIC, __ATOMIC_RELEASE);
    return 0;
}

/* Capacity of an initialized ring with this record size, else -1. */
int64_t ring_check(const void *base, int64_t rec_size)
{
    const RingHeader *h = (const RingHeader *)base;
    if (__atomic_load_n(&h->magic, __ATOMIC_ACQUIRE) != RING_MAGIC || h->rec_size != (uint64_t)rec_size)
        return -1;
    return (int64_t)h->capacity;
}

/* Publish up to n records; returns how many fit (the rest count as refused). */
int64_t ring_push(void *base, const void *recs, int64_t n)
{
    RingHeader *h = (RingHeader *)base;
    uint8_t *data = (uint8_t *)base + RING_HEADER;
    const uint64_t cap = h->capacity, rs = h->rec_size;
    const uint64_t head = h->head; /* own counter */
    const uint64_t tail = __atomic_load_n(&h->tail, __ATOMIC_ACQUIRE);
    uint64_t k = cap - (head - tail);
    if ((uint64_t)n < k)
        k = (uint64_t)n;
    const uint64_t at = head & (cap - 1);
    const uint64_t first = k < cap - at ? k : cap - at;
    memcpy(data + at * rs, recs, first * rs);
    memcpy(data, (const uint8_t *)recs + first * rs, (k - first) * rs);
    __atomic_store_n(&h->head, head + k, __ATOMIC_RELEASE);
    if ((uint64_t)n > k)
        __atomic_store_n(&h->full, h->full + ((uint64_t)n - k), __ATOMIC_RELAXED);
    return (int64_t)k;
}

/* Consume up to max_n records into out; returns how many. */
int64_t ring_pop(void *base, void *out, int64_t max_n)
{
    RingHeader *h = (RingHeader *)base;
    const uint8_t *data = (const uint8_t *)base + RING_HEADER;
    const uint64_t cap = h->capacity, rs = h->rec_size;
    const uint64_t tail = h->tail; /* own counter */
    const uint64_t head = __atomic_load_n(&h->head, __ATOMIC_ACQUIRE);
    uint64_t k = head - tail;
    if ((uint64_t)max_n < k)
        k = (uint64_t)max_n;
    const uint64_t at = tail & (cap - 1);
    const uint64_t first = k < cap - at ? k : cap - at;
    memcpy(out, data + at * rs, first * rs);
    memcpy((uint8_t *)out + first * rs, data, (k - first) * rs);
    __atomic_store_n(&h->tail, tail + k, __ATOMIC_RELEASE);
    return (int64_t)k;
}

/* stats[0..2] = records waiting, published in total, refused in total. */
void ring_stats(const void *base, int64_t *stats)
{
    const RingHeader *h = (const RingHeader *)base;
    const uint64_t head = __atomic_load_n(&h->head, __ATOMIC_ACQUIRE);
    const uint64_t tail = __atomic_load_n(&h->tail, __ATOMIC_ACQUIRE);
    stats[0] = (int64_t)(head - tail);
    stats[1] = (int64_t)head;
    stats[2] = (int64_t)__atomic_load_n(&h->full, __ATOMIC_RELAXED);
}
