#include "kernel.h"
#include "tensor.h"
#include "poly.h"
#include <stdio.h>
#include <stdint.h>
#include <mma.h>
#include <stdlib.h> 
#include <time.h>

using namespace nvcuda;

__device__ int postive_mod(int x, int y){
    if(x >=0 ) return x;
    else return x + y;
}


#define V_BARRETT 929445

__device__ int16_t barrett_reduce_int32_t(int32_t a) //和%时间无差别
{
  int64_t t;
  t = V_BARRETT * ((int64_t) a);
  t >>= 32;
  t *= FPTRU_Q;
  //printf("%ld,%d,",t,a);
  if( a - t >= FPTRU_Q){
    return a - t - FPTRU_Q;
  }
  return a - t;
}

__global__ void wmma_ker_padding(int8_t *array_a, int8_t *array_b, int32_t *array_c) {
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, TC_X, TC_Y, TC_Z, int8_t, wmma::row_major> a_frag;
   // wklee, Read a in row major and b in column major
   wmma::fragment<wmma::matrix_b, TC_X, TC_Y, TC_Z, int8_t, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, TC_X, TC_Y, TC_Z, int32_t> c_frag;

   // Each warp compute 16 elements along index i
   uint32_t tot_warpID = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
   //每一个是41个Warp
   uint32_t warpgroup = tot_warpID / 41;
   uint32_t warpID = tot_warpID - warpgroup * 41;

   uint32_t ldA_offset, ldB_offset, row_idx, col_idx, st_offset;
   row_idx = warpID%((B_y)/TC_Z)*TC_Z;
   col_idx = warpID/((B_y)/TC_Z)*TC_X;//TODO:这里会不会有问题呢？
   st_offset = col_idx + row_idx * A_x ; 
   int8_t * a = &array_a[A_x * A_y * warpgroup];
   int8_t * b = &array_b[B_x * B_y * warpgroup];
   int32_t * c = &array_c[A_x * B_y * warpgroup];
   
   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0);  
   for (int i = 0; i < (A_y)/TC_Y; i ++)    
    {
     
      ldA_offset = col_idx*(A_y) + i*TC_Y; //TODO:明天继续改下面的，注意范围的限制
      ldB_offset = row_idx*(B_x) + i*TC_Y; 
      wmma::load_matrix_sync(a_frag, a + ldA_offset , A_y);   
      wmma::load_matrix_sync(b_frag, b + ldB_offset  , B_x);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }    
    wmma::store_matrix_sync(c + st_offset , c_frag, A_x, wmma::mem_col_major);    
}

/*2024-6-26:
输入:BATCH个多项式a，将其转化为矩阵的形式
线程组织形式<<<FPTRU_N,FPTRU_N>>>*/
__global__ void init_matrix_a(poly* a,int8_t *ah,int8_t *al){
    int bid=blockIdx.x;
    int tid=threadIdx.x;
    /*ah[bid * A_y + tid] = 0;
    al[bid * A_y + tid] = 0;

    
    //__shared__ int16_t s_f[N];

    if(tid < N && bid < N){ */ //这里的分支是否是有必要的
        //s_f[tid] = f[tid];
        //__syncthreads();
        for(int num=0;num<BATCH_SIZE;num++){
            int16_t tmp = a[num].coeffs[postive_mod(bid - tid , N)];
            if (bid != 0 && tid - bid >= 0){
                tmp += a[num].coeffs[postive_mod(N - 1 - (tid -bid) , N)];
            }
            ah[num * A_x * A_y + bid * A_y + tid] = tmp >> 7;
            al[num * A_x * A_y + bid * A_y + tid] = tmp & (0x7f);
        }
    //}
}


__global__ void init_matrix_a_v3(poly* a,int8_t *ah,int8_t *al){
    int bidx=blockIdx.x;
    int tid=threadIdx.x;
    /*ah[bid * A_y + tid] = 0;
    al[bid * A_y + tid] = 0;

    
    //__shared__ int16_t s_f[N];

    if(tid < N && bid < N){ */ //这里的分支是否是有必要的
        //s_f[tid] = f[tid];
        //__syncthreads();

        __shared__ poly s_a;
        if(!tid){
            s_a = a[bidx];
        }
        __syncthreads();
        for(int bid=0;bid<FPTRU_N;bid++){
            int16_t tmp = s_a.coeffs[postive_mod(bid - tid , N)];
            if (bid != 0 && tid - bid >= 0){
                tmp += s_a.coeffs[postive_mod(N - 1 - (tid -bid) , N)];
            }
            ah[bidx * A_x * A_y + bid * A_y + tid] = tmp >> 7;
            al[bidx * A_x * A_y + bid * A_y + tid] = tmp & (0x7f);
        }
    //}
}

/*__global__ void init_matrix_a(poly* a,int8_t *ah,int8_t *al){
    int bid=blockIdx.x;
    int tid=threadIdx.x;

    for(int num=0;num<BATCH_SIZE;num++){
        int16_t tmp = a[num].coeffs[postive_mod(bid - tid , N)];
        if (bid != 0 && tid - bid >= 0){
            tmp += a[num].coeffs[postive_mod(N - 1 - (tid -bid) , N)];
        }
        ah[num * A_x * A_y + bid * A_y + tid] = tmp >> 7;
        al[num * A_x * A_y + bid * A_y + tid] = tmp & (0x7f);
    }
}*/


__global__ void init_matrix_a_v2(poly* a,int8_t *ah,int8_t *al){
    int bid=blockIdx.x;
    int tidx=threadIdx.x;
    int tidy=threadIdx.y;

    printf("%d,%d,%d,%d\n",bid,tidx,tidy,threadIdx.z);
    
    int16_t tmp = a[bid].coeffs[postive_mod(tidy - tidx , N)];
    if (tidy != 0 && tidx - tidy >= 0){
        tmp += a[bid].coeffs[postive_mod(N - 1 - (tidx -tidy) , N)];
    }
    ah[bid * A_x * A_y + tidy * A_y + tidx] = tmp >> 7;
    al[bid * A_x * A_y + tidy * A_y + tidx] = tmp & (0x7f);

}


/*2024-6-26:
将多项式b转化为**列主序**矩阵B
线程组织形式<<<BATCH_SIZE,653>>>*/
__global__ void init_matrix_b(poly * b,int8_t * B){
    int bid=blockIdx.x;//对应着多项式的下标
    int tid=threadIdx.x;
    B[bid * B_x * B_y +tid] = b[bid].coeffs[tid];
}


/*2024-6-26:
将结果的低高位组合得到相应的结果*/
__global__ void merge_hl(int32_t * r_h, int32_t * r_l , poly * p){
    int bid=blockIdx.x; //对应具体多项式
    int tid=threadIdx.x;

    int32_t res = (r_h[bid * A_x * B_y + tid] << 7) + r_l[bid * A_x * B_y+ tid]; //todo:改为64

    //r[tid] = barrett_reduce_int32_t(res);
    p[bid].coeffs[tid] = fq_freeze(res);
}

//#define looka
/*2024-6-26:
输入:多项式数组a、b、c，其中a对应着大系数多项式,b对应着小系数多项式
输出:c=a*b*/
void tensor_poly_mul(poly *c, poly *a,poly *b,cudaStream_t stream){
    int32_t *C_l,*C_h;
    int8_t *A_h,*A_l,*B;

    HANDLE_ERROR(cudaMalloc((void**)&C_h, A_x * B_y * sizeof(int32_t) * BATCH_SIZE));
    HANDLE_ERROR(cudaMalloc((void**)&C_l, A_x * B_y * sizeof(int32_t) * BATCH_SIZE));

    HANDLE_ERROR(cudaMalloc((void**)&A_h, A_x * A_y * sizeof(int8_t) * BATCH_SIZE));
    HANDLE_ERROR(cudaMalloc((void**)&A_l, A_x * A_y * sizeof(int8_t) * BATCH_SIZE));

    HANDLE_ERROR(cudaMalloc((void**)&B, B_x * B_y * sizeof(int8_t) * BATCH_SIZE));

    init_matrix_a<<<FPTRU_N,FPTRU_N,0,stream>>>(a,A_h,A_l);

    //init_matrix_a_v3<<<BATCH_SIZE,FPTRU_N,0,stream>>>(a,A_h,A_l);

    //dim3 threadsPerBlock(FPTRU_N, FPTRU_N);

    //init_matrix_a_v2<<<BATCH_SIZE,threadsPerBlock,0,stream>>>(a,A_h,A_l);
    //cudaDeviceSynchronize();


#ifdef lookc
    int8_t h_al[A_x*A_y*BATCH_SIZE];
    int8_t h_ah[A_x*A_y*BATCH_SIZE];
    cudaMemcpy(h_al,A_l, A_x * A_y * sizeof(int8_t) * BATCH_SIZE,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ah,A_h, A_x * A_y * sizeof(int8_t) * BATCH_SIZE,cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    printf("looka\n");
    for(int num=0;num < BATCH_SIZE;num++){
        printf("num = %d\n",num);
        for(int i=0;i<A_x;i++){
        for(int j=0;j<A_y;j++){
            printf("(%d,%d)",h_ah[num * A_x * A_y + i*A_y + j] + h_al[num * A_x * A_y + i*A_y + j]);
        }
        printf("\n\n");
    }
    }
#endif

    init_matrix_b<<<BATCH_SIZE,653,0,stream>>>(b,B);


#ifdef lookb
    int8_t bb[B_x*B_y*BATCH_SIZE];
    
    cudaMemcpy(bb,B, B_x * B_y * sizeof(int8_t) * BATCH_SIZE,cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    printf("lookB\n");
    for(int num=0;num < BATCH_SIZE;num++){
        printf("num = %d\n",num);
        for(int i=0;i<A_x;i++){
        for(int j=0;j<B_y;j++){
            printf("%d,",bb[num * B_x * B_y + j*B_x + i]);
        }
        printf("\n\n");
    }
    }
#endif

    int blocks = 41 * (BATCH_SIZE / 8);
    int threads = 32 * 8;

    wmma_ker_padding<<<blocks,threads,0,stream>>>(A_h,B,C_h);
    wmma_ker_padding<<<blocks,threads,0,stream>>>(A_l,B,C_l);

#ifdef looka
    int32_t c_l[A_x*B_y*BATCH_SIZE];
    int32_t c_h[A_x*B_y*BATCH_SIZE];
    cudaMemcpy(c_l,C_l, A_x * B_y * sizeof(int32_t) * BATCH_SIZE,cudaMemcpyDeviceToHost);
    cudaMemcpy(c_h,C_h, A_x * B_y * sizeof(int32_t) * BATCH_SIZE,cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    printf("lookc\n");
    for(int num=0;num < BATCH_SIZE;num++){
        printf("num = %d\n",num);
        for(int i=0;i<A_x;i++){
        for(int j=0;j<B_y;j++){
            printf("(%d,%d)",c_h[num * A_x * B_y + j*A_x + i] , c_l[num * A_x * B_y + j*A_x + i]);
        }
        printf("\n\n");
    }
    }
#endif


    merge_hl<<<BATCH_SIZE,FPTRU_N,0,stream>>>(C_h,C_l,c);

    HANDLE_ERROR(cudaFree(C_l));
    HANDLE_ERROR(cudaFree(C_h));
    HANDLE_ERROR(cudaFree(A_l));
    HANDLE_ERROR(cudaFree(A_h));
    HANDLE_ERROR(cudaFree(B));

}


