//image encoder

#include "encoder.cuh"
#include "../kernels/matmul.cuh"
#include <cuda_runtime.h>
#include <math.h>

__global__ void patch_extractor_kernel(
    float* patches, const float* images, int batch, int C, int H, int W, int patch_size,int num_patches_h)//num_patches_h - num of patches along one dimension(=/patch_size)
    {
        int b = blockIdx.z;// batch item
        int p = blockIdx.y;// patch index
        int d = blockIdx.x * blockDim.x + threadIdx.x;// position in flattened patch.
        // 3D grid with z for batch, y for patch index, x for position within patch.
        //d run from 0 to patch_dim - 1 = p*p*C - 1.

        int patch_dim = patch_size * patch_size * C;
        if (d >= patch_dim) return;// boundary check
        
        int ph = p / num_patches_h;// patch row in the patch grid
        int pw = p % num_patches_h;// patch column in the patch grid

        int c = d / (patch_size * patch_size);// channel index within the patch
        int py = (d % (patch_size * patch_size)) / patch_size;// pixel row y  within the patch
        int px = d % patch_size;// pixel column x within the patch
        //decompose flat index d into channel, within-patch -y, within-patch-x.

        int iy = ph * patch_size + py;// absolute y in the full image
        int ix = pw * patch_size + px;// absolute x in the full image

    }
