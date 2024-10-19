#include <iostream>
#include <cuda_runtime.h>
__device__ void dd(int x,uint16_t * a,uint16_t *b, uint16_t *c, uint16_t *d){
    uint16_t R2[1277];//uint16_t R2[(len + 1) / 2];
    uint16_t M2[1277];//uint16_t M2[(len + 1) / 2];
    uint16_t bottomr[1277];//uint16_t bottomr[len / 2];
    uint16_t bottomt[1277];//uint32_t bottomt[len / 2];
    a[0] = 10;
    if(x>0){
        printf("%d\n",x);
        dd(x-1,R2,M2,bottomr,bottomt);
    }
}
__global__ void digui(){
    uint16_t R2[1277];//uint16_t R2[(len + 1) / 2];
    uint16_t M2[1277];//uint16_t M2[(len + 1) / 2];
    uint16_t bottomr[1277];//uint16_t bottomr[len / 2];
    uint16_t bottomt[1277];//uint32_t bottomt[len / 2];
    printf("hi\n");
    dd(12,R2,M2,bottomr,bottomt);
}
int main(){
    digui<<<1,1>>>();
    cudaDeviceSynchronize();
}