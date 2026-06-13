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

         patches[b * (num_patches_h * num_patches_h) * patch_dim + p * patch_dim + d] = images[b * C * H * W + c * H * W + iy * W + ix];
         //destination indext, flat row-major for (batch, num_patches, patch_dim), source index : NCHW format- b*C*H*W skips to the right batch item, c*H*W skips to the right channel, iy*w + ix is the 2D pixel position. 

         
    }

__global__  void add_positional_embedding_kernel(float* output, const float* pos_emb, int batch, int num_patches, int d_model){
            
            int idx = blockIdx.x * blockDim.x + threadIdx.x;// idx is a flat index over the entire (batch x num_patches x d_model) output tensor
            if(idx >= batch * num_patches * d_model) return;
            int p = (idx / d_model) % num_patches;// idx/d_model, which token(batch_item*num_patches + patch_index) and then %num_patches strips out the batch dimension to get hte patch position
            output[idx] += pos_emb[p * d_model + (idx % d_model)];// idx % d_model features dimension.
}

void encoder_fwd(){
    
}