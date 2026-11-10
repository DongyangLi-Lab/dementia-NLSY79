#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

// Computes W * X^T in the same orientation as the saved output projection.
// Diagnostic only: tests periodic FP16 accumulator rounding between K blocks.
extern "C" __global__ void projection_periodic_round(
    const half* x, const half* w, const half* bias, half* output,
    int features, int rows, int inner, int interval, int rounding_mode) {
    int m = blockIdx.x * 16;
    int n = blockIdx.y * 16;
    if (m + 16 > features || n + 16 > rows) return;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> partial;
    wmma::fill_fragment(c, 0.0f);
    wmma::fill_fragment(partial, 0.0f);
    for (int k = 0; k < inner; k += 16) {
        wmma::load_matrix_sync(a, w + m * inner + k, inner);
        wmma::load_matrix_sync(b, x + n * inner + k, inner);
        if (rounding_mode == 3) {
            wmma::mma_sync(partial, a, b, partial);
            if ((k + 16) % interval == 0 || k + 16 == inner) {
                for (int j = 0; j < c.num_elements; ++j) c.x[j] += partial.x[j];
                wmma::fill_fragment(partial, 0.0f);
            }
        } else {
            wmma::mma_sync(c, a, b, c);
        }
        if (interval > 0 && (k + 16) % interval == 0 && k + 16 < inner) {
            for (int j = 0; j < c.num_elements; ++j) {
                if (rounding_mode == 1) {
                    c.x[j] = wmma::__float_to_tf32(c.x[j]);
                } else if (rounding_mode == 2) {
                    unsigned bits = __float_as_uint(c.x[j]);
                    bits = (bits + 0xfffu + ((bits >> 13) & 1u)) & 0xffffe000u;
                    c.x[j] = __uint_as_float(bits);
                } else {
                    c.x[j] = __half2float(__float2half_rn(c.x[j]));
                }
            }
        }
    }
    __shared__ float tile[256];
    wmma::store_matrix_sync(tile, c, 16, wmma::mem_row_major);
    __syncwarp();
    for (int index = threadIdx.x; index < 256; index += 32) {
        int dm = index / 16;
        int dn = index % 16;
        output[(n + dn) * features + m + dm] =
            __float2half_rn(tile[index] + __half2float(bias[m + dm]));
    }
}
