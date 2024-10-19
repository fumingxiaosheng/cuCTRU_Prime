#include "params.h"
#include "kernel.h"
#include "cbd.h"
#include "reduce.h"
#include "poly.h"
#include "coding.h"

// #if (FPTRU_N == 653)
#include "./poly_mul_n653q/n653.h"
// #endif

#include <iostream>


__global__ void poly_sample(poly *array_a,unsigned char * array_coins){
    //printf("[%d %d]poly_sample start\n",blockIdx.x,threadIdx.x);
#if (FPTRU_N == 653) //FPTRU_N / 8 + 1个线程
    cbd3(&array_a[blockIdx.x], &array_coins[blockIdx.x * FPTRU_COIN_BYTES],0,0);
#elif (FPTRU_N == 761) //FPTRU_N / 4 个线程
    cbd2(&array_a[blockIdx.x], &array_coins[blockIdx.x * FPTRU_COIN_BYTES],0,0);
#elif (FPTRU_N == 1277) //FPTRU_N / 4 个线程
    cbd2(&array_a[blockIdx.x], &array_coins[blockIdx.x * FPTRU_COIN_BYTES],0,0);
#endif


}

__global__ void poly_double(poly *array_b,poly *array_a){
    if(threadIdx.x < FPTRU_N){
      //(array_b + blockIdx.x * sizeof(poly))->coeffs[threadIdx.x] = (array_a + blockIdx.x * sizeof(poly) )->coeffs[threadIdx.x] + (array_a + blockIdx.x * sizeof(poly))->coeffs[threadIdx.x];//注意这里得偏移一下多项式的位置

      array_b[blockIdx.x].coeffs[threadIdx.x]=array_a[blockIdx.x].coeffs[threadIdx.x] + array_a[blockIdx.x].coeffs[threadIdx.x];
    }
}

/*2024-4-25:
输入:array_a:全局内存,多项式数组，用于存储最终的sample结果
    array_cions:全局内存,种子数组,用于最为采样的随机数
    is_double:是否需要系数乘以2
    is_add1:是否需要第0个系数加上1

处理流程:在cbd的过程中，引入double的操作

思考:
    1.实验发现，在gpu代码上，额外的循环操作往往会引起时耗的增加，比如，对于cions的拷贝
    2.数据先分块，再处理，不要分块、处理，再分块处理，比如在cbd完了之后再进行double
    
时效:
    before:0.013312 ms
    after: 0.008192 ms*/
__global__ void poly_sample_and_double_v2(poly *array_a,unsigned char * array_coins,uint16_t is_double,uint16_t is_add1,int interval){
    //__shared__ poly a; //共享内存作为写入的地址
    /*
    unsigned char coins[FPTRU_COIN_BYTES];
    
    for(int i=0;i<FPTRU_COIN_BYTES;i++){ //TODO:首先读取到寄存器中
        coins[i]=array_coins[blockIdx.x * FPTRU_COIN_BYTES + i];
    }*/
#if (FPTRU_N == 653) //FPTRU_N / 8 + 1个线程
    cbd3(&array_a[blockIdx.x], &array_coins[blockIdx.x * interval],is_double,is_add1);
#elif (FPTRU_N == 761) //FPTRU_N / 4 个线程
    cbd2(&array_a[blockIdx.x],&array_coins[blockIdx.x * interval],is_double,is_add1);
    //cbd2_761(&array_a[blockIdx.x],&array_coins[blockIdx.x * interval],is_double,is_add1);
#elif (FPTRU_N == 1277) //FPTRU_N / 4 个线程
    cbd2(&array_a[blockIdx.x],&array_coins[blockIdx.x * interval],is_double,is_add1);
#endif
    //__syncthreads();//共享内存同步一下
}

/*2024-4-25:
输入:array_a:全局内存，多项式数组
    a:共享内存,多项式

处理流程:把计算得到的多项式a拷贝到全局内存array_a中

评价:单独拎出来会极大地减慢速度
*/
__device__ void poly_double_v2(poly *array_a,poly a){
#if(FPTRU_N == 653) //除以4个线程
    if(threadIdx.x == FPTRU_N/4){
        array_a[blockIdx.x].coeffs[threadIdx.x * 4]= a.coeffs[threadIdx.x * 4] + a.coeffs[threadIdx.x * 4];
    }
    else{
        array_a[blockIdx.x].coeffs[threadIdx.x * 4]= a.coeffs[threadIdx.x * 4] + a.coeffs[threadIdx.x * 4];

        array_a[blockIdx.x].coeffs[threadIdx.x * 4 + 1]= a.coeffs[threadIdx.x * 4 + 1] + a.coeffs[threadIdx.x * 4 + 1];

        array_a[blockIdx.x].coeffs[threadIdx.x * 4 + 2]= a.coeffs[threadIdx.x * 4 + 2] + a.coeffs[threadIdx.x * 4 + 2];

        array_a[blockIdx.x].coeffs[threadIdx.x * 4 + 3]= a.coeffs[threadIdx.x * 4 + 3] + a.coeffs[threadIdx.x * 4 + 3];
    }
#elif(FPTRU_N == 761) //除以8个线程
    if(threadIdx.x == FPTRU_N/8){
        array_a[blockIdx.x].coeffs[threadIdx.x * 8]= a.coeffs[threadIdx.x * 8] + a.coeffs[threadIdx.x * 8];
    }
    else{
        array_a[blockIdx.x].coeffs[threadIdx.x * 8]= a.coeffs[threadIdx.x * 8] + a.coeffs[threadIdx.x * 8];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 1]= a.coeffs[threadIdx.x * 8 + 1] + a.coeffs[threadIdx.x * 8 + 1];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 2]= a.coeffs[threadIdx.x * 8 + 2] + a.coeffs[threadIdx.x * 8 + 2];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 3]= a.coeffs[threadIdx.x * 8 + 3] + a.coeffs[threadIdx.x * 8 + 3];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 4]= a.coeffs[threadIdx.x * 8 + 4] + a.coeffs[threadIdx.x * 8 + 4];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 5]= a.coeffs[threadIdx.x * 8 + 5] + a.coeffs[threadIdx.x * 8 + 5];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 6]= a.coeffs[threadIdx.x * 8 + 6] + a.coeffs[threadIdx.x * 8 + 6];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 7]= a.coeffs[threadIdx.x * 8 + 7] + a.coeffs[threadIdx.x * 8 + 7];
    }

#elif(FPTRU_N == 1277)
    if(threadIdx.x == FPTRU_N/8){
        array_a[blockIdx.x].coeffs[threadIdx.x * 8]= a.coeffs[threadIdx.x * 8] + a.coeffs[threadIdx.x * 8];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 1]= a.coeffs[threadIdx.x * 8 + 1] + a.coeffs[threadIdx.x * 8 + 1];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 2]= a.coeffs[threadIdx.x * 8 + 2] + a.coeffs[threadIdx.x * 8 + 2];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 3]= a.coeffs[threadIdx.x * 8 + 3] + a.coeffs[threadIdx.x * 8 + 3];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 4]= a.coeffs[threadIdx.x * 8 + 4] + a.coeffs[threadIdx.x * 8 + 4];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 5]= a.coeffs[threadIdx.x * 8 + 5] + a.coeffs[threadIdx.x * 8 + 5];
    }
    else{
        array_a[blockIdx.x].coeffs[threadIdx.x * 8]= a.coeffs[threadIdx.x * 8] + a.coeffs[threadIdx.x * 8];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 1]= a.coeffs[threadIdx.x * 8 + 1] + a.coeffs[threadIdx.x * 8 + 1];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 2]= a.coeffs[threadIdx.x * 8 + 2] + a.coeffs[threadIdx.x * 8 + 2];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 3]= a.coeffs[threadIdx.x * 8 + 3] + a.coeffs[threadIdx.x * 8 + 3];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 4]= a.coeffs[threadIdx.x * 8 + 4] + a.coeffs[threadIdx.x * 8 + 4];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 5]= a.coeffs[threadIdx.x * 8 + 5] + a.coeffs[threadIdx.x * 8 + 5];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 6]= a.coeffs[threadIdx.x * 8 + 6] + a.coeffs[threadIdx.x * 8 + 6];

        array_a[blockIdx.x].coeffs[threadIdx.x * 8 + 7]= a.coeffs[threadIdx.x * 8 + 7] + a.coeffs[threadIdx.x * 8 + 7];
    }
#endif
}

__global__ void poly_copy(poly *dst, poly *src){
    if(threadIdx.x < FPTRU_N){
      dst->coeffs[threadIdx.x] = src->coeffs[threadIdx.x];
    }
}

/*2024-4-19:
输入:array_a:待处理的多项式数组

输出:存储到array_a中

处理流程:调整多项a的每个系数位于[0,FPTRU_Q-1]

线程组织形式:<<<BATCH_SIZE,1>>>
*/
__global__ void poly_fqcsubq_batch(poly *array_a){ 
    for (int i = 0; i < FPTRU_N; ++i){
        array_a[blockIdx.x].coeffs[i]=fqcsubq(array_a[blockIdx.x].coeffs[i]);
    }

}

/*2024-4-29:
输入:array_a:待处理的多项式数组

输出:存储到array_a中

线程组织形式:<<<BATCH_SIZE,FPTRU_N>>>*/
__global__ void poly_fqcsubq_batch_v2(poly *array_a){
    array_a[blockIdx.x].coeffs[threadIdx.x]=fqcsubq(array_a[blockIdx.x].coeffs[threadIdx.x]);
}

__global__ void poly_fqcsubq_batch_1277(poly *array_a){
    if(threadIdx.x == FPTRU_N / 4){
        array_a[blockIdx.x].coeffs[4 * threadIdx.x]=fqcsubq(array_a[blockIdx.x].coeffs[4 * threadIdx.x]);
    }
    else{
        array_a[blockIdx.x].coeffs[4 * threadIdx.x]=fqcsubq(array_a[blockIdx.x].coeffs[4 * threadIdx.x]);
        array_a[blockIdx.x].coeffs[4 * threadIdx.x + 1]=fqcsubq(array_a[blockIdx.x].coeffs[4 * threadIdx.x + 1]);
        array_a[blockIdx.x].coeffs[4 * threadIdx.x + 2]=fqcsubq(array_a[blockIdx.x].coeffs[4 * threadIdx.x + 2]);
        array_a[blockIdx.x].coeffs[4 * threadIdx.x + 3]=fqcsubq(array_a[blockIdx.x].coeffs[4 * threadIdx.x + 3]);
    }
}

/*2024-4-28:
线程组织形式:<<<1,FPTRU_N + 1>>>*/
__global__ void poly_inverse_once(poly * finv, poly * f){
    //printf("In poly_inverse_once\n");
    poly_inverse(finv,f);
}

/*2024-6-24:
线程组织形式:<<<1,FPTRU_N + 1>>>
*/
__global__ void poly_inv(poly * finv, poly * f){
    poly_inverse(&finv[blockIdx.x],&f[blockIdx.x]);
}


/*
输入:array_finv:设备端变量,用于存储计算得到的逆
     array_f:设备端变量,待计算逆的多项式数组
输出:存储到array_finv上

处理流程:使用蒙哥马利求逆的技巧,来同时快速计算多个多项式的逆

*/
void poly_inverse_batch_0(poly * array_finv,poly * array_f,cudaStream_t stream){//TODO:加上stream
    poly *fmul;
    poly *fmul_inv;
    cudaEvent_t start, stop;
    
    HANDLE_ERROR(cudaMalloc(&fmul,sizeof(poly)));
    HANDLE_ERROR(cudaMalloc(&fmul_inv,sizeof(poly)));


    //BEFORE_SPEED
    poly_copy<<<1,FPTRU_N>>>(fmul,&array_f[0]);

    //printf("[in poly_inverse]poly_copy done\n");
    for(int i = 1;i < BATCH_SIZE; i++ ){ //N个多项式乘法 TODO:这里能否有优化的点
        //poly_mul_q1_batch(fmul,fmul,&array_f[i],stream,1);//TODO:这里能否加上多个stream呢？这里是无法并行的，因为这是一个线性的结构->或者用之前简单的乘法的例子，但是此时ntt的算法结构就需要发生改变
        //printf("[in poly_inverse]poly_mul_q1_batch done\n");
        poly_mul_653_batch_q1_v2<<<1,N_N653/2,0,stream>>>(fmul,fmul,&array_f[i]);
    }

    //AFTER_SPEED("N_poly_MUL")

    //BEFORE_SPEED
    poly_inverse_once<<<1,1,0,stream>>>(fmul_inv,fmul);
    //AFTER_SPEED("poly_inverse_once v1")




    //BEFORE_SPEED
    for(int i=BATCH_SIZE-1;i>=2;i--){
        //poly_mul_q1_batch(&array_finv[i],fmul_inv,&array_f[i-1],stream,1); //TODO:poly_mul_q1应该直接写成函数的形式

        poly_mul_653_batch_q1_v2<<<1,N_N653/2,0,stream>>>(&array_finv[i],fmul_inv,&array_f[i-1]);
        for(int j= i-2; j >= 0; j--){
            //poly_mul_q1_batch(&array_finv[i],&array_finv[i],&array_f[j],stream,1);
            poly_mul_653_batch_q1_v2<<<1,N_N653/2,0,stream>>>(&array_finv[i],&array_finv[i],&array_f[j]);
        }
        //poly_mul_q1_batch(fmul_inv,fmul_inv,&array_f[i],stream,1);
        poly_mul_653_batch_q1_v2<<<1,N_N653/2,0,stream>>>(fmul_inv,fmul_inv,&array_f[i]);
    }

    //poly_mul_q1_batch(&array_finv[1],fmul_inv,&array_f[0],stream,1);
    poly_mul_653_batch_q1_v2<<<1,N_N653/2,0,stream>>>(&array_finv[1],fmul_inv,&array_f[0]);
    //poly_mul_q1_batch(&array_finv[0],fmul_inv,&array_f[1],stream,1);
    poly_mul_653_batch_q1_v2<<<1,N_N653/2,0,stream>>>(&array_finv[0],fmul_inv,&array_f[1]);

    //AFTER_SPEED("N_inverse_recover v1")
    cudaStreamSynchronize(stream);//等待执行完成，再free掉申请的指针TODO:这里会没有并行性，导致CPU等待运行结束
    cudaFree(fmul);
    cudaFree(fmul_inv);

}

__device__ void device_function2(int * a){
    *a = 1;
}

__device__ void device_function3(int  a){
    a = 1;
}

__global__ void kernel2(){
    __shared__ int a;
    __shared__ int b;
    a=0;
    b=0;
    device_function2(&a);
    device_function3(b);

    printf("%d %d\n",a,b);
}

void poly_inverse_batch_0_v2(poly * array_finv,poly * array_f,cudaStream_t stream){
    poly *fmul;
    poly *fmul_inv;
    cudaEvent_t start, stop;

    
    HANDLE_ERROR(cudaMalloc(&fmul,sizeof(poly)));
    HANDLE_ERROR(cudaMalloc(&fmul_inv,sizeof(poly)));

    //printf("use kernel2 for test\n");
    //kernel2<<<1,1,0,stream>>>(); //DONE:测试表明，能够将共享内存作为参数传递

    //printf("N_poly_mul 2\n");
    //BEFORE_SPEED
    N_poly_mul<<<1,N_N653/2,0,stream>>>(fmul,array_f);//正确性验证
    //AFTER_SPEED("N_poly_mul")
    
    /*printf("N_poly_mul\n\n");
    poly *fmul_h;

    HANDLE_ERROR(cudaHostAlloc((void**)&fmul_h, sizeof(poly),cudaHostAllocDefault));


    HANDLE_ERROR(cudaMemcpyAsync(fmul_h, fmul, sizeof(poly) , cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);
    //cudaDeviceSynchronize();
    for(int i=0;i<FPTRU_N;i++){
        printf("%d,",fmul_h->coeffs[i]);
    }
    printf("\n");*/

    //BEFORE_SPEED
    //printf("poly_inverse_once\n");
    poly_inverse_once<<<1,FPTRU_N + 1,0,stream>>>(fmul_inv,fmul);//正确性得以验证
    //AFTER_SPEED("poly_inverse_once v2")

    /*HANDLE_ERROR(cudaMemcpyAsync(fmul_h, fmul_inv, sizeof(poly) , cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);
    //cudaDeviceSynchronize();
    for(int i=0;i<FPTRU_N;i++){
        printf("%d,",fmul_h->coeffs[i]);
    }
    printf("\n");*/


    //BEFORE_SPEED
    //printf("N_inverse_recover\n");
    N_inverse_recover<<<1,N_N653/2,0,stream>>>(fmul_inv,array_f,array_finv);
    //AFTER_SPEED("N_inverse_recover v2")


    cudaStreamSynchronize(stream);//等待执行完成，再free掉申请的指针TODO:这里会没有并行性，导致CPU等待运行结束
    cudaFree(fmul);
    cudaFree(fmul_inv);
}

/*2024-6-23:
*/
void poly_inverse_batch_0_v3(poly * array_finv,poly * array_f,cudaStream_t stream){
    poly *fmul;
    poly *fmul_inv;
    cudaEvent_t start, stop;

    
    HANDLE_ERROR(cudaMalloc(&fmul,sizeof(poly)));
    HANDLE_ERROR(cudaMalloc(&fmul_inv,sizeof(poly)));

    //printf("use kernel2 for test\n");
    //kernel2<<<1,1,0,stream>>>(); //DONE:测试表明，能够将共享内存作为参数传递

    //printf("N_poly_mul 2\n");
    //BEFORE_SPEED
    N_poly_mul_v3<<<1,168,0,stream>>>(fmul,array_f);//正确性验证
    //AFTER_SPEED("N_poly_mul")
    
    /*printf("N_poly_mul\n\n");
    poly *fmul_h;

    HANDLE_ERROR(cudaHostAlloc((void**)&fmul_h, sizeof(poly),cudaHostAllocDefault));


    HANDLE_ERROR(cudaMemcpyAsync(fmul_h, fmul, sizeof(poly) , cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);
    //cudaDeviceSynchronize();
    for(int i=0;i<FPTRU_N;i++){
        printf("%d,",fmul_h->coeffs[i]);
    }
    printf("\n");*/

    //BEFORE_SPEED
    //printf("poly_inverse_once\n");
    poly_inverse_once<<<1,FPTRU_N + 1,0,stream>>>(fmul_inv,fmul);//正确性得以验证
    //AFTER_SPEED("poly_inverse_once v2")

    /*HANDLE_ERROR(cudaMemcpyAsync(fmul_h, fmul_inv, sizeof(poly) , cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);
    //cudaDeviceSynchronize();
    for(int i=0;i<FPTRU_N;i++){
        printf("%d,",fmul_h->coeffs[i]);
    }
    printf("\n");*/


    //BEFORE_SPEED
    //printf("N_inverse_recover\n");
    N_inverse_recover_v3<<<1,168,0,stream>>>(fmul_inv,array_f,array_finv);
    //AFTER_SPEED("N_inverse_recover v2")


    cudaStreamSynchronize(stream);//等待执行完成，再free掉申请的指针TODO:这里会没有并行性，导致CPU等待运行结束
    cudaFree(fmul);
    cudaFree(fmul_inv);
}

/*2024-4-27:
输入:fmul:设备端变量，用于存储最终的相乘结果
    array_f:设备端变量,含BATCH_SIZE个多项式

处理流程:将BATCH_SIZE个多项式相乘的结果存储到fmul中

线程组织形式:<<<1,N_N653/2>>>*/
__global__ void N_poly_mul(poly * res_fmul, poly * array_f)
{   
    //printf("in N_poly_mul\n"); //TODO:这里为什么没有输出呢！
    __shared__ poly fmul;
    //__shared__ poly array_f[BATCH_SIZE];//考虑到对于全局内存的合并内存访问,这里还是将其设置为共享内存

    /*if(threadIdx.x < BATCH_SIZE){//BATCH_SIZE往往为32的整数倍，符合warp的形式，不会引起太多的时间浪费
        array_f[threadIdx.x] =src_array_f[threadIdx.x]; //DONE:这里是指针的拷贝还是完全的拷贝->这里应该是完全拷贝
    }

    __syncthreads();*///nvlink error   : Entry function '_Z10N_poly_mulP4polyS0_' uses too much shared data (0x1265a bytes, 0xc000 max) 共享内存数量限制，故删除

    /*if(threadIdx.x == 0){ //DONE:这里的array_f是正确的
        for(int j=0;j<BATCH_SIZE;j++){
            for(int i=0;i<FPTRU_N;i++){
                printf("%d,",array_f[j].coeffs[i]);
            }
            printf("\n");
        }
    }*/

    /*if(threadIdx.x == 0){
        for(int i=0;i<FPTRU_N;i++){
            printf("%d,",0);
        }
        printf("\n");
    }*/
    if(threadIdx.x < FPTRU_N){
        fmul.coeffs[threadIdx.x] = array_f[0].coeffs[threadIdx.x];
        //res_fmul->coeffs[threadIdx.x] = array_f[0].coeffs[threadIdx.x];
    }
    
    /*fmul.coeffs[2 * threadIdx.x] = array_f[0].coeffs[2 * threadIdx.x];
    fmul.coeffs[2 * threadIdx.x + 1] = array_f[0].coeffs[2 * threadIdx.x + 1];*/ //这里存在越界访问

    //TODO:改进为以下并进行速度的测量
    /*
    fmul.coeffs[threadIdx.x] = array_f[0].coeffs[threadIdx.x];
    fmul.coeffs[threadIdx.x + N_N653/2] = array_f[0].coeffs[threadIdx.x + N_N653/2];*/

    __syncthreads();
    /*if(threadIdx.x == 0){
        for(int i=0;i<FPTRU_N;i++){
            printf("%d,",fmul.coeffs[i]);
        }
        printf("\n");

        for(int i=0;i<FPTRU_N;i++){
            printf("%d,",array_f[0].coeffs[i]);
        }
        printf("\n");
    }*/

    for(int i = 1; i < BATCH_SIZE; i++){
        //poly_mul_653_batch_q1_v2_device(&fmul,&fmul,&array_f[i]);//DONE:是什么引起了共享内存的错误->共享内存取址会引发错误
        //poly_mul_653_batch_q1_v2_device(fmul,fmul,&array_f[i]);//应该是这个函数的问题

        /*if(threadIdx.x == 0){//TODO:看一下是不是poly_mul_653_batch_q1_v2_device函数改变了输入
            for(int i=0;i<FPTRU_N;i++){
                printf("%d,",array_f[i].coeffs[i]);
            }
            printf("\n");
        }*/

        /*if(threadIdx.x == 0){
            //printf("++++++++++%d++++++++++++\n",i);
            //printf("input\n");
            for(int j=0;j<FPTRU_N;j++){
                printf("%d,",fmul.coeffs[j]);
                //printf("%d,",res_fmul->coeffs[j]);
            }
            printf("\n");

            for(int j=0;j<FPTRU_N;j++){
                printf("%d,",array_f[i].coeffs[j]);
            }
            printf("\n");
        }*/

        poly_mul_653_batch_q1_v2_device(&fmul,&fmul,&array_f[i]);//TODO:两个共享内存同时做输入和输出是否存在问题，返回值赋给fmul会导致内存的访问冲突，但是此处用于验证是否是共享内存导致的问题
        //poly_mul_653_batch_q1_v2_device(res_fmul,res_fmul,&array_f[i]);
        
        __syncthreads();
        /*if(threadIdx.x == 0){
            //printf("output\n");
            for(int j=0;j<FPTRU_N;j++){
                printf("%d,",fmul.coeffs[j]);
            }
            printf("\n");
        }*/
    }

    /*if(threadIdx.x == 0){
        printf("++++++end+++++\n");
    }*/
    
    if(threadIdx.x < FPTRU_N){
        res_fmul->coeffs[threadIdx.x] = fmul.coeffs[threadIdx.x];
    }
    /*
    res_fmul->coeffs[threadIdx.x] = fmul.coeffs[threadIdx.x];
    res_fmul->coeffs[threadIdx.x + N_N653/2 ] = fmul.coeffs[threadIdx.x + N_N653/2]; //存在越界访问
    printf("[%d]%d [%d]%d\n",threadIdx.x,threadIdx.x + N_N653/2,res_fmul->coeffs[threadIdx.x],res_fmul->coeffs[threadIdx.x + N_N653/2 ]);*/
}


/*2024-6-22:
输入:fmul:设备端变量，用于存储最终的相乘结果
    array_f:设备端变量,含BATCH_SIZE个多项式

处理流程:将BATCH_SIZE个多项式相乘的结果存储到fmul中

线程组织形式:<<<1,168>>>*/
__global__ void N_poly_mul_v3(poly * res_fmul, poly * array_f)
{   

    __shared__ poly fmul;
    int tid = threadIdx.x;
    
    if(threadIdx.x == 0){
        fmul = array_f[0];
    }
    

    __syncthreads();

    for(int i = 1; i < BATCH_SIZE; i++){
        poly_mul_653_batch_q1_v3_device(&fmul,&fmul,&array_f[i]);//TODO:两个共享内存同时做输
        
        __syncthreads();
    }
    if(threadIdx.x == 0){
        res_fmul[0] = fmul;
    }
}

/*2024-4-28:
输入:src_fmul_inv:先前计算的BATCH_SIZE个多项式相乘的结果，全局变量
     array_f:BATCH_SIZE个多项式，全局变量
     array_finv:BATCH_SIZE个多项式，用于存储每个多项式求逆的结果

输出:存储到array_f中

线程组织形式:<<<1,N_N653/2>>>

处理流程:将fmul_inv和finv设置为共享内存，每计算一个，就将其写入共享内存中
*/
__global__ void N_inverse_recover(poly * src_fmul_inv,poly * array_f ,poly * array_finv){
    //printf("in N_inverse_recover\n");
    __shared__ poly fmul_inv;
    __shared__ poly finv;
    fmul_inv = src_fmul_inv[0];

    for(int i=BATCH_SIZE-1;i>=2;i--){
        poly_mul_653_batch_q1_v2_device(&finv,&fmul_inv,&array_f[i-1]);
        __syncthreads();
        for(int j=i-2;j >= 0; j--){
            poly_mul_653_batch_q1_v2_device(&finv,&finv,&array_f[j]);
            __syncthreads();
        }

        //将计算的结果赋值给全局内存
        if(threadIdx.x == 0 ){
            array_finv[i]=finv;
        }
        
        //更正fmul_inv的值
        poly_mul_653_batch_q1_v2_device(&fmul_inv,&fmul_inv,&array_f[i]);
        __syncthreads();
    }

    poly_mul_653_batch_q1_v2_device(&finv,&fmul_inv,&array_f[0]);
    __syncthreads();

    if(threadIdx.x == 0 ){
        array_finv[1]=finv;
    }

    __syncthreads();

    poly_mul_653_batch_q1_v2_device(&finv,&fmul_inv,&array_f[1]);
    __syncthreads();

    if(threadIdx.x == 0 ){
        array_finv[0]=finv;
    }
    __syncthreads();
}


/*2024-6-24:
输入:src_fmul_inv:先前计算的BATCH_SIZE个多项式相乘的结果，全局变量
     array_f:BATCH_SIZE个多项式，全局变量
     array_finv:BATCH_SIZE个多项式，用于存储每个多项式求逆的结果

输出:存储到array_f中

线程组织形式:<<<1,168>>>

处理流程:将fmul_inv和finv设置为共享内存，每计算一个，就将其写入共享内存中
*/
__global__ void N_inverse_recover_v3(poly * src_fmul_inv,poly * array_f ,poly * array_finv){
    //printf("in N_inverse_recover\n");
    __shared__ poly fmul_inv;
    __shared__ poly finv;
    if(threadIdx.x == 0){
        fmul_inv = src_fmul_inv[0];
    }
    

    for(int i=BATCH_SIZE-1;i>=2;i--){
        poly_mul_653_batch_q1_v3_device(&finv,&fmul_inv,&array_f[i-1]);
        __syncthreads();
        for(int j=i-2;j >= 0; j--){
            poly_mul_653_batch_q1_v3_device(&finv,&finv,&array_f[j]);
            __syncthreads();
        }

        //将计算的结果赋值给全局内存
        if(threadIdx.x == 0 ){
            array_finv[i]=finv;
        }
        
        //更正fmul_inv的值
        poly_mul_653_batch_q1_v3_device(&fmul_inv,&fmul_inv,&array_f[i]);
        __syncthreads();
    }

    poly_mul_653_batch_q1_v3_device(&finv,&fmul_inv,&array_f[0]);
    __syncthreads();

    if(threadIdx.x == 0 ){
        array_finv[1]=finv;
    }

    __syncthreads();

    poly_mul_653_batch_q1_v3_device(&finv,&fmul_inv,&array_f[1]);
    __syncthreads();

    if(threadIdx.x == 0 ){
        array_finv[0]=finv;
    }
    __syncthreads();
}


/*
输入:array_finv:设备端变量,用于存储计算得到的逆
     array_f:设备端变量,待计算逆的多项式数组
输出:存储到array_finv上

处理流程:每个线程处理一个poly_inverse
*/
__global__ void poly_inverse_batch_1(poly * array_finv,poly * array_f){
    poly_inverse(&array_finv[threadIdx.x * sizeof(poly)],&array_f[threadIdx.x * sizeof(poly)]);
}

/*TODO:
1.poly_inverse内部是什么样的呢?
2.poly_mul_q1应该如何进行分装
*/

__device__ void poly_inverse(poly *b, const poly *a)
{
    //printf("in poly_inverse\n");
// #if(NEW == 0)
//   rq_inverse(b->coeffs, a->coeffs);
// #endif

#if(NEW == 1)
  rq_inverse_clean(b->coeffs,a->coeffs);
  /*printf("v2 result\n");
  for(int i=0 ;i <FPTRU_N;i++){
    printf("%d,",b->coeffs[i]);
  }
  printf("\n");
  rq_inverse(b->coeffs, a->coeffs);
  printf("v1 result\n");
  for(int i=0 ;i <FPTRU_N;i++){
    printf("%d,",b->coeffs[i]);
  }
  printf("\n");*/
#endif
}

/*2024-4-8:
输入:y:存储x/m
    r:存储x mod m
    x:无符号被除数
    m:无符号除数
输出:将x/m保存在y中,将x mod m保存在r中
处理流程:将输入的32位整数 x 除以14位整数 m，并将结果保存在 y 中，余数保存在 r 中*/
__device__ void uint32_divmod_uint14(uint32_t *y, uint16_t *r, uint32_t x, uint16_t m)
{
    uint32_t w = 0x80000000;
    uint32_t qpart;
    uint32_t mask;

    w /= m;

    *y = 0;
    qpart = (x * (uint64_t)w) >> 31;
    x -= qpart * m;
    *y += qpart;

    qpart = (x * (uint64_t)w) >> 31;
    x -= qpart * m;
    *y += qpart;

    x -= m;
    *y += 1;
    mask = -(x >> 31);
    x += mask & (uint32_t)m;
    *y += mask;

    *r = x;
}

void uint32_divmod_uint14_cpu(uint32_t *y, uint16_t *r, uint32_t x, uint16_t m)
{
    uint32_t w = 0x80000000;
    uint32_t qpart;
    uint32_t mask;

    w /= m;

    *y = 0;
    qpart = (x * (uint64_t)w) >> 31;
    x -= qpart * m;
    *y += qpart;

    qpart = (x * (uint64_t)w) >> 31;
    x -= qpart * m;
    *y += qpart;

    x -= m;
    *y += 1;
    mask = -(x >> 31);
    x += mask & (uint32_t)m;
    *y += mask;

    *r = x;
}

/*2024-4-8:
输入:y:存储x/m
    r:存储x mod m（要求r一定是正数）
    x:有符号被除数
    m:无符号除数
输出:将x/m保存在y中,将x mod m保存在r中，例如x=-31,m=6,y=-6,r=5
处理流程:将输入的32位整数 x 除以14位整数 m，并将结果保存在 y 中，余数保存在 r 中*/
__device__ void int32_divmod_uint14(int32_t *y, uint16_t *r, int32_t x, uint16_t m)
{
    uint32_t uq, uq2;
    uint16_t ur, ur2;
    uint32_t mask;

    uint32_divmod_uint14(&uq, &ur, 0x80000000 + (uint32_t)x, m);
    uint32_divmod_uint14(&uq2, &ur2, 0x80000000, m);

    ur -= ur2;
    uq -= uq2;

    mask = -(uint32_t)(ur >> 15);
    ur += mask & m;
    uq += mask;
    *r = ur;
    *y = uq;
}

/*2024-4-8:
输入:x:32bit的无符号数
    m:32bit的无符号数
输出:x mod m
处理流程:调用uint32_divmod_uint14来计算x/m和x mod m,并返回结果x mod m
*/
__device__ uint16_t uint32_mod_uint14(uint32_t x, uint16_t m)
{
    uint32_t q;
    uint16_t r;
    uint32_divmod_uint14(&q, &r, x, m);
    return r;
}

uint16_t uint32_mod_uint14_cpu(uint32_t x, uint16_t m)
{
    uint32_t q;
    uint16_t r;
    uint32_divmod_uint14_cpu(&q, &r, x, m);
    return r;
}

/*2024-4-8:
输入:x:32bit的有符号数
    m:32bit的无符号数
输出:x mod m
处理流程:调用int32_divmod_uint14来计算x/m和x mod m,并返回结果x mod m*/
__device__ uint16_t int32_mod_uint14(int32_t x, uint16_t m)
{
    int32_t y;
    uint16_t r;
    int32_divmod_uint14(&y, &r, x, m);
    return r;
}
/*2024-4-8:TODO:如何通过half_q来控制相应的范围
输入:x:
输出:x mod FPTRU_Q，且控制了mod的范围*/
__device__ int16_t fq_freeze(int32_t x)
{

    const int16_t half_q = (FPTRU_Q - 1) >> 1;
    return int32_mod_uint14(x + half_q, FPTRU_Q) - half_q;
}

__device__ int int16_nonzero_mask(int16_t x)
{
    uint16_t u = x;
    uint32_t w = u;
    w = -w;
    w >>= 31;
    return -w;
}

__device__ int int16_negative_mask(int16_t x)
{
    uint16_t u = x;
    u >>= 15;
    return -(int)u;
}

#if (FPTRU_Q == 4091)

__device__ int16_t fq_inverse(int16_t a0)
{
    int16_t a = a0;
    int16_t t = a0;

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    return t;
}

#elif (FPTRU_Q == 4591)

__device__ int16_t fq_inverse(int16_t a0)
{
    int16_t a = a0;
    int16_t t = a0;

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    return t;
}

#elif (FPTRU_Q == 4621)

__device__ int16_t fq_inverse(int16_t a0)
{
    int16_t a = a0;
    int16_t t = a0;

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    return t;
}
#elif (FPTRU_Q == 7879)

__device__ int16_t fq_inverse(int16_t a0)
{
    int16_t a = a0;
    int16_t t = a0;

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    a = fq_freeze(a * (int32_t)a);
    t = fq_freeze(t * (int32_t)a);

    return t;
}
#endif

__device__ int rq_inverse(int16_t *finv, const int16_t *f)
{
    int16_t Phi[FPTRU_N + 1], F[FPTRU_N + 1], V[FPTRU_N + 1], S[FPTRU_N + 1];
    int i, loop, Delta, swap, t;
    int32_t Phi0, F0;
    int16_t scale;

    for (i = 0; i < FPTRU_N; ++i)
        Phi[i] = 0;
    Phi[0] = 1;
    Phi[FPTRU_N - 1] = Phi[FPTRU_N] = -1;

    for (i = 0; i < FPTRU_N; ++i)
        F[FPTRU_N - 1 - i] = f[i];//i会越界访问
    F[FPTRU_N] = 0;

    Delta = 1;

    for (i = 0; i < FPTRU_N + 1; ++i)
        V[i] = 0;

    for (i = 0; i < FPTRU_N + 1; ++i)
        S[i] = 0;
    S[0] = 1;

    for (loop = 0; loop < 2 * FPTRU_N - 1; ++loop)
    {
        for (i = FPTRU_N; i > 0; --i)
            V[i] = V[i - 1];
        V[0] = 0;

        swap = int16_negative_mask(-Delta) & int16_nonzero_mask(F[0]);

        for (i = 0; i < FPTRU_N + 1; ++i)
        {
            t = swap & (Phi[i] ^ F[i]);
            Phi[i] ^= t;
            F[i] ^= t;
            t = swap & (V[i] ^ S[i]);
            V[i] ^= t;
            S[i] ^= t;
        }

        Delta ^= swap & (Delta ^ -Delta);
        Delta++;

        Phi0 = Phi[0];
        F0 = F[0];

        for (i = 0; i < FPTRU_N + 1; ++i)
            F[i] = fq_freeze(Phi0 * F[i] - F0 * Phi[i]);
        for (i = 0; i < FPTRU_N; ++i)
            F[i] = F[i + 1];
        F[FPTRU_N] = 0;

        for (i = 0; i < FPTRU_N + 1; ++i)
            S[i] = fq_freeze(Phi0 * S[i] - F0 * V[i]);
    }

    scale = fq_inverse(Phi[0]);
    for (i = 0; i < FPTRU_N; ++i)
        finv[i] = fq_freeze(scale * (int32_t)V[FPTRU_N - 1 - i]);

    return int16_nonzero_mask(Delta);
}

/*2024-4-28:
输入:finv:FPTRU_N个int16_t组成的数组
    f:FPTRU_N个int16_t组成的数组

输出:存储到finv中

线程组织形式:<<<1,FPTRU_N + 1>>> 
*/
__device__ void rq_inverse_v2(int16_t *finv,const int16_t *f){
    __shared__ int16_t Phi[FPTRU_N + 1], F[FPTRU_N + 1], V[FPTRU_N + 1], S[FPTRU_N + 1];

    __shared__ int Delta;

    __shared__ int swap;
    int  loop, t;
    __shared__ int32_t Phi0, F0;
    __shared__ int16_t scale;

    int16_t tmp;

    //TODO:输入是否需要变为共享内存
    Phi[threadIdx.x] = ((!(int16_t)threadIdx.x) & 1 ) - ((!((int16_t)threadIdx.x - FPTRU_N)) & (1)) - ((!((int16_t)threadIdx.x - FPTRU_N + 1)) & (1));


    if(threadIdx.x < FPTRU_N){//存在越界的问题，需要使用条件判断语句
        F[FPTRU_N - 1 - threadIdx.x] = f[threadIdx.x];
    }
    else {
        F[threadIdx.x] = 0;
        Delta = 1;
    }


    V[threadIdx.x] = 0;
    S[threadIdx.x] = (!threadIdx.x) & 1;//线程为0时，对应的值为1

    __syncthreads();

    //下面进入循环
    for(loop = 0;loop < 2 * FPTRU_N - 1; ++loop){ //主要的问题在于循环的次数过多
        if(threadIdx.x){ //存在越界问题，需要使用条件判断语句
            tmp = V[threadIdx.x - 1];
        }
        else{
            tmp = 0;
        }
        __syncthreads();//防止读写冲突
        V[threadIdx.x] = tmp;


        if(!threadIdx.x){
            swap = int16_negative_mask(-Delta) & int16_nonzero_mask(F[0]);
        }

        __syncthreads();//等待线程0计算完swap再向后继续

        t = swap & (Phi[threadIdx.x] ^ F[threadIdx.x]);
        Phi[threadIdx.x] ^= t;
        F[threadIdx.x] ^= t;
        t = swap & (V[threadIdx.x] ^ S[threadIdx.x]);
        V[threadIdx.x] ^= t;
        S[threadIdx.x] ^= t;
        
        if(!threadIdx.x){//对Phi[0]和F[0]的选择都是由线程编号为0的线程来执行的，因此此处不需要线程的同步
            Delta ^= swap & (Delta ^ -Delta);
            Delta++;

            Phi0 = Phi[0];//这里更新了吗
            F0 = F[0];
        }

        __syncthreads();//等待线程0计算结束

        F[threadIdx.x] = fq_freeze(Phi0 * F[threadIdx.x] - F0 * Phi[threadIdx.x]);

        __syncthreads();//等待F更新结束

        if(threadIdx.x < FPTRU_N){
            tmp = F[threadIdx.x + 1];
        }
        else{
            tmp = 0;
        }

        __syncthreads();
        F[threadIdx.x] = tmp;

        S[threadIdx.x] = fq_freeze(Phi0 * S[threadIdx.x] - F0 * V[threadIdx.x]);
    }

    if(!threadIdx.x){
        scale = fq_inverse(Phi[0]);
    }

    __syncthreads();//等待线程0计算结束
    
    if(threadIdx.x < FPTRU_N){
        finv[threadIdx.x] = fq_freeze(scale * (int32_t)V[FPTRU_N - 1 - threadIdx.x]);
    }


}


/*2024-7-21:
过大的循环体导致核函数的较为耗时，可能是由于错误的内存访问模式:
循环判断的开销等造成的;
输入:finv:FPTRU_N个int16_t组成的数组
    f:FPTRU_N个int16_t组成的数组

输出:存储到finv中

线程组织形式:<<<1,FPTRU_N + 1>>> 
*/
__device__ void rq_inverse_v3(int16_t *finv,const int16_t *f){
    __shared__ int16_t Phi[FPTRU_N + 1], F[FPTRU_N + 1], V[FPTRU_N + 1], S[FPTRU_N + 1];//这里应该页不是很多吧

    __shared__ int Delta;

    //int Delta;
    __shared__ int swap;
    int  loop, t;
    __shared__ int32_t Phi0, F0;
    __shared__ int16_t scale;

    int16_t tmp;

    //TODO:输入是否需要变为共享内存
    Phi[threadIdx.x] = ((!(int16_t)threadIdx.x) & 1 ) - ((!((int16_t)threadIdx.x - FPTRU_N)) & (1)) - ((!((int16_t)threadIdx.x - FPTRU_N + 1)) & (1));


    if(threadIdx.x < FPTRU_N){//存在越界的问题，需要使用条件判断语句
        F[FPTRU_N - 1 - threadIdx.x] = f[threadIdx.x];
    }
    else{
        F[threadIdx.x] = 0;
        Delta = 1;
    }
    
    

    V[threadIdx.x] = 0;
    S[threadIdx.x] = (!threadIdx.x) & 1;//线程为0时，对应的值为1

    __syncthreads();


    //TODO:删掉
    __syncthreads();
    if(threadIdx.x == 1){
        printf("\n\nPhi\n");
        
        for(int i=0;i<=FPTRU_N;i++){
            printf("%d,",Phi[i]);
        }

        printf("\n\nS\n");
        for(int i=0;i<=FPTRU_N;i++){
            printf("%d,",S[i]);
        }

        printf("\n\nF\n");
        for(int i=0;i<=FPTRU_N;i++){
            printf("%d,",F[i]);
        }

        printf("\n\nV\n");
        for(int i=0;i<=FPTRU_N;i++){
            printf("%d,",V[i]);
        }
    } //以上构建的输入的正确性已经得到验证
    
    //下面进入循环
#pragma unroll//循环展开
    for(loop = 0;loop < 2 * FPTRU_N - 1; ++loop){ //主要的问题在于循环的次数过多
        if(threadIdx.x){ //存在越界问题，需要使用条件判断语句
            tmp = V[threadIdx.x - 1];
        }
        else{
            tmp = 0;
        }
        __syncthreads();//防止读写冲突
        V[threadIdx.x] = tmp;


        if(!threadIdx.x){
            swap = int16_negative_mask(-Delta) & int16_nonzero_mask(F[0]);
        }

        __syncthreads();//等待线程0计算完swap再向后继续
        if(threadIdx.x == 0){
            printf("ff0=%d,vv0=%d,Phi0=%d,swap=%d,S0=%d\n",F[0],V[0],Phi[0],swap,S[0]);
        }
        t = swap & (Phi[threadIdx.x] ^ F[threadIdx.x]);
        Phi[threadIdx.x] ^= t;
        F[threadIdx.x] ^= t;
        t = swap & (V[threadIdx.x] ^ S[threadIdx.x]);
        V[threadIdx.x] ^= t;
        S[threadIdx.x] ^= t;

        /*+++++++++++++++++++做正确性测试使用++++++++++++++++++++++++++++*/
        __syncthreads();
        if(threadIdx.x == 1){
            printf("\n\nPhi\n");
            
            for(int i=0;i<=FPTRU_N;i++){
                printf("%d,",Phi[i]);
            }

            printf("\n\nS\n");
            for(int i=0;i<=FPTRU_N;i++){
                printf("%d,",S[i]);
            }

            printf("\n\nF\n");
            for(int i=0;i<=FPTRU_N;i++){
                printf("%d,",F[i]);
            }

            printf("\n\nV\n");
            for(int i=0;i<=FPTRU_N;i++){
                printf("%d,",V[i]);
            }
        } //以上构建的输入的正确性已经得到验证

        /*+++++++++++++++++++做正确性测试使用++++++++++++++++++++++++++++*/
        break;
    
        
        if(!threadIdx.x){//对Phi[0]和F[0]的选择都是由线程编号为0的线程来执行的，因此此处不需要线程的同步
            Delta ^= swap & (Delta ^ -Delta);
            Delta++;

            Phi0 = Phi[0];
            F0 = F[0];
        }

        __syncthreads();//等待线程0计算结束

        F[threadIdx.x] = fq_freeze(Phi0 * F[threadIdx.x] - F0 * Phi[threadIdx.x]);

        __syncthreads();//等待F更新结束

        if(threadIdx.x < FPTRU_N){
            tmp = F[threadIdx.x + 1];
        }
        else{
            tmp = 0;
        }

        __syncthreads();
        F[threadIdx.x] = tmp;

        S[threadIdx.x] = fq_freeze(Phi0 * S[threadIdx.x] - F0 * V[threadIdx.x]);


        /*+++++++++++++++++++做正确性测试使用++++++++++++++++++++++++++++*/
        __syncthreads();
        if(threadIdx.x == 1){
            printf("\n\nPhi\n");
            
            for(int i=0;i<=FPTRU_N;i++){
                printf("%d,",Phi[i]);
            }

            printf("\n\nS\n");
            for(int i=0;i<=FPTRU_N;i++){
                printf("%d,",S[i]);
            }

            printf("\n\nF\n");
            for(int i=0;i<=FPTRU_N;i++){
                printf("%d,",F[i]);
            }

            printf("\n\nV\n");
            for(int i=0;i<=FPTRU_N;i++){
                printf("%d,",V[i]);
            }
        } //以上构建的输入的正确性已经得到验证

    /*+++++++++++++++++++做正确性测试使用++++++++++++++++++++++++++++*/
    }

    if(!threadIdx.x){
        scale = fq_inverse(Phi[0]);
    }

    __syncthreads();//等待线程0计算结束
    
    if(threadIdx.x < FPTRU_N){
        finv[threadIdx.x] = fq_freeze(scale * (int32_t)V[FPTRU_N - 1 - threadIdx.x]);
    }


}

//使用FPTRU_N个线程：都使用共享内存的话，会导致频繁地syncthreads,所以要使用寄存器,都维护起来F0和Phi0。每个线程负责维护threadIdx.x+1的位置，同时自己维护下标为0的
//线程组织形式为<<<BATCH_SIZE,FPTRU_N>>>
/*2024-7-23:完成正确性检验*/
__device__ void rq_inverse_v4_test(int16_t *finv,const int16_t *f){
    __shared__ int16_t V[FPTRU_N + 1],F[FPTRU_N + 1];
    int idx = threadIdx.x + 1;//向后移动1个,每一个都计算一边F0
    int16_t Phi,S,Phi0,S0;

    int Delta,swap,t;
    int16_t scale;
    int16_t tmp,ff0,vv0;

    int32_t phi0,f0;


    Phi = ((!(int16_t)idx) & 1 ) - ((!((int16_t)idx - FPTRU_N)) & (1)) - ((!((int16_t)idx - FPTRU_N + 1)) & (1));
    Phi0 = 1;

    F[FPTRU_N - 1 - threadIdx.x] = f[threadIdx.x];//TODO：F[FPTRU_N]是否要重新赋值为0
    
    //printf("f[%d]=%d,",threadIdx.x,f[threadIdx.x]);
    //根据实验结果,共享内存区域并不是初始就赋值为0,因此需要进行0赋值
    
    V[idx] = 0;

    //实验结果:分支执行比存储体冲突带来的开销更大
    /*if(idx == 1){
        F[FPTRU_N] = 0;
        V[0] = 0;
    }*/
    
    F[FPTRU_N] = 0;
    V[0] = 0;

    __syncthreads();//确保第一次关于V的读取是正确的->加了之后没有效果 TODO:正确之后把这个给删了

    //++++++++++++++++++++++++++++++++

    /*if(idx == 1) printf("F[FPTRU_N] = %d\n,V[0]=%d",F[FPTRU_N],V[0]);
    printf("V[%d]=%d\n",idx,V[idx]);*/

    Delta = 1;

    //TODO:是否需要为V赋值为0
    S = 0;
    S0 = 1;
    

    /*+++++++++++++++++++做正确性测试使用++++++++++++++++++++++++++++*/
    //__shared__ int16_t test_phi[FPTRU_N + 1],test_s[FPTRU_N + 1];
    /*test_phi[idx] = Phi;
    test_s[idx] = S;

    __syncthreads();
    if(idx == 1){
        printf("\n\nPhi\n");
        printf("%d,",Phi0);
        for(int i=1;i<=FPTRU_N;i++){
            printf("%d,",test_phi[i]);
        }

        printf("\n\nS\n");
        printf("%d,",S0);
        for(int i=1;i<=FPTRU_N;i++){
            printf("%d,",test_s[i]);
        }

        printf("\n\nF\n");
        for(int i=0;i<=FPTRU_N;i++){
            printf("%d,",F[i]);
        }

        printf("\n\nV\n");
        for(int i=0;i<=FPTRU_N;i++){
            printf("%d,",V[i]);
        }
    } //以上构建的输入的正确性已经得到验证*/

    /*+++++++++++++++++++做正确性测试使用++++++++++++++++++++++++++++*/

    //下面进入循环
#pragma unroll//循环展开
    for(int loop = 0;loop < 2 * FPTRU_N - 1; ++loop){ //主要的问题在于循环的次数过多
        
        tmp = V[idx-1];
        __syncthreads();//防止读写冲突
        V[idx] = tmp;
        V[0] = 0; //TODO:V[0]的值没有发生过改变 是否会造成冲突
        //__syncthreads();
        
        swap = int16_negative_mask(-Delta) & int16_nonzero_mask(F[0]);
       

        t = swap & (Phi ^ F[idx]);
        Phi ^= t;
        F[idx] ^= t;
        t = swap & (V[idx] ^ S);
        V[idx] ^= t;
        S ^= t;
        
        ff0 = F[0];
        vv0 = V[0];
        __syncthreads();

        //printf("ff0=%d,vv0=%d,Phi0=%d,swap=%d,S0=%d\n",ff0,vv0,Phi0,swap,S0);//输入是完全一样的

        t = swap & (Phi0 ^ ff0); 
        Phi0 ^= t;
        //__syncthreads();
        ff0 ^= t; //是否会造成存储体冲突
        t = swap & (vv0 ^ S0);
        //__syncthreads();
        vv0 ^= t;
        S0 ^= t; //TODO:删掉t0

        F[0] = ff0;
        V[0] = vv0;

        //TODO:先看到这里的正确性,F和V的第一个有问题

        /*+++++++++++++++++++做正确性测试使用++++++++++++++++++++++++++++*/
        /*test_phi[idx] = Phi;
        test_s[idx] = S;

        __syncthreads();
        if(idx == 1){
            printf("\n正确性测试结果 loop = %d\n",loop);
            printf("\n\nPhi\n");
            printf("%d,",Phi0);
            for(int i=1;i<=FPTRU_N;i++){
                printf("%d,",test_phi[i]);
            }

            printf("\n\nS\n");
            printf("%d,",S0);
            for(int i=1;i<=FPTRU_N;i++){
                printf("%d,",test_s[i]);
            }

            printf("\n\nF\n");
            for(int i=0;i<=FPTRU_N;i++){
                printf("%d,",F[i]);
            }

            printf("\n\nV\n");
            for(int i=0;i<=FPTRU_N;i++){
                printf("%d,",V[i]);
            }
        } //以上构建的输入的正确性已经得到验证*/
        /*+++++++++++++++++++做正确性测试使用++++++++++++++++++++++++++++*/

        
        Delta ^= swap & (Delta ^ -Delta);
        Delta++;

        
        phi0 = Phi0;
        f0 = F[0];
        ff0 = F[0];

        __syncthreads();//读了F[0],为了防止其余线程把这个给覆盖掉

        F[idx] = fq_freeze(phi0 * F[idx] - f0 * Phi);
        //F[0] = fq_freeze(phi0 * F[0] - f0 * Phi0); //这里会出问题，因为有些数值算了F[0]，然后又迭代进去了
        F[0] = fq_freeze(phi0 * ff0 - f0 * Phi0);
        //__syncthreads();
        tmp = F[idx];
        __syncthreads();
        F[threadIdx.x] = tmp;
        F[FPTRU_N] = 0; //TODO:存储体冲突
        

        S = fq_freeze(phi0 * S - f0 * V[idx]);
        S0 = fq_freeze(phi0 * S0 - f0 * V[0]); //S0可能是由于F的不同导致的V[0]的不同导致的

        /*+++++++++++++++++++做正确性测试使用++++++++++++++++++++++++++++*/
        /*test_phi[idx] = Phi;
        test_s[idx] = S;

        __syncthreads();
        if(idx == 1){
            printf("\n正确性测试结果 loop = %d\n",loop);
            printf("\n\nPhi\n");
            printf("%d,",Phi0);
            for(int i=1;i<=FPTRU_N;i++){
                printf("%d,",test_phi[i]);
            }

            printf("\n\nS\n");
            printf("%d,",S0);
            for(int i=1;i<=FPTRU_N;i++){
                printf("%d,",test_s[i]);
            }

            printf("\n\nF\n");
            for(int i=0;i<=FPTRU_N;i++){
                printf("%d,",F[i]);
            }

            printf("\n\nV\n");
            for(int i=0;i<=FPTRU_N;i++){
                printf("%d,",V[i]);
            }
        } //以上构建的输入的正确性已经得到验证
        break;*/
        /*+++++++++++++++++++做正确性测试使用++++++++++++++++++++++++++++*/
        
    }

    scale = fq_inverse(Phi0);

    finv[threadIdx.x] = fq_freeze(scale * (int32_t)V[FPTRU_N - 1 - threadIdx.x]);

}

__device__ void rq_inverse_v4(int16_t *finv,const int16_t *f){
    __shared__ int16_t V[FPTRU_N + 1],F[FPTRU_N + 1];
    int idx = threadIdx.x + 1;//向后移动1个,每一个都计算一边F0
    int16_t Phi,S,Phi0,S0,tmpV,tmpF;

    int Delta,swap,t;
    int16_t scale;
    int16_t tmp,ff0,vv0;

    int32_t phi0,f0;


    Phi = ((!(int16_t)idx) & 1 ) - ((!((int16_t)idx - FPTRU_N)) & (1)) - ((!((int16_t)idx - FPTRU_N + 1)) & (1));
    Phi0 = 1;

    F[FPTRU_N - 1 - threadIdx.x] = f[threadIdx.x];
    
    
    V[idx] = 0;    
    F[FPTRU_N] = 0;
    V[0] = 0;

    __syncthreads();

    Delta = 1;

    S = 0;
    S0 = 1;
    

    //下面进入循环
#pragma unroll//循环展开
    //v5版本,速度为900.74
    for(int loop = 0;loop < 2 * FPTRU_N - 1; ++loop){
        
        tmpV = V[idx-1];
        //__syncthreads();//防止读写冲突
        //V[idx] = tmp;
        //V[0] = 0;
        tmpF = F[idx];
        ff0 = F[0];
        swap = int16_negative_mask(-Delta) & int16_nonzero_mask(ff0);
       
        t = swap & (Phi ^ tmpF);
        Phi ^= t;
        tmpF ^= t;
        t = swap & (tmpV ^ S);//读共享内存
        tmpV ^= t;//写共享内存
        S ^= t;
        

        //vv0 = V[0];
        vv0 = 0;//V】
        //__syncthreads();


        t = swap & (Phi0 ^ ff0); 
        Phi0 ^= t;
        ff0 ^= t;
        t = swap & (vv0 ^ S0);
        vv0 ^= t;
        S0 ^= t;

        //F[0] = ff0;
        
        
        Delta ^= swap & (Delta ^ -Delta);
        Delta++;

        
        phi0 = Phi0;
        f0 = ff0;//F[0];
        //ff0 = F[0];

        //__syncthreads();

        tmpF = fq_freeze(phi0 * tmpF - f0 * Phi);
        //F[0] = fq_freeze(phi0 * ff0 - f0 * Phi0);
        
        //tmp = F[idx];
        S = fq_freeze(phi0 * S - f0 * tmpV);
        S0 = fq_freeze(phi0 * S0 - f0 * vv0);

        __syncthreads();
        F[threadIdx.x] = tmpF;
        F[FPTRU_N] = 0; //TODO:存储体冲突

        V[idx] = tmpV;
        V[0] = vv0;
        __syncthreads();
    }

    
    /*//v4版本速度为1257.343071
    for(int loop = 0;loop < 2 * FPTRU_N - 1; ++loop){
        
        tmp = V[idx-1];
        __syncthreads();//防止读写冲突
        V[idx] = tmp;
        V[0] = 0;
        
        swap = int16_negative_mask(-Delta) & int16_nonzero_mask(F[0]);
       
        t = swap & (Phi ^ F[idx]);
        Phi ^= t;
        F[idx] ^= t;
        t = swap & (V[idx] ^ S);//读共享内存
        V[idx] ^= t;//写共享内存
        S ^= t;
        
        ff0 = F[0];
        vv0 = V[0];
        __syncthreads();


        t = swap & (Phi0 ^ ff0); 
        Phi0 ^= t;
        ff0 ^= t;
        t = swap & (vv0 ^ S0);
        vv0 ^= t;
        S0 ^= t;

        F[0] = ff0;
        V[0] = vv0;
        
        Delta ^= swap & (Delta ^ -Delta);
        Delta++;

        
        phi0 = Phi0;
        f0 = F[0];
        ff0 = F[0];

        __syncthreads();

        F[idx] = fq_freeze(phi0 * F[idx] - f0 * Phi);
        F[0] = fq_freeze(phi0 * ff0 - f0 * Phi0);
        
        tmp = F[idx];
        __syncthreads();
        F[threadIdx.x] = tmp;
        F[FPTRU_N] = 0; //TODO:存储体冲突
        

        S = fq_freeze(phi0 * S - f0 * V[idx]);
        S0 = fq_freeze(phi0 * S0 - f0 * V[0]);
    }*/

    scale = fq_inverse(Phi0);

    finv[threadIdx.x] = fq_freeze(scale * (int32_t)V[FPTRU_N - 1 - threadIdx.x]);

}

/*2024-7-24:
已经验证了正确性
TODO:为什么防止内存栅栏会存在区别*/
__device__ void rq_inverse_clean(int16_t *finv,const int16_t *f){
    __shared__ int16_t V[FPTRU_N + 1],F[FPTRU_N + 1];
    int idx = threadIdx.x + 1;//向后移动1个,每一个都计算一边F0
    int16_t Phi,S,Phi0,S0,tmpV,tmpF;

    int Delta,swap,t;
    int16_t scale;
    int16_t tmp,ff0,vv0;

    int32_t phi0,f0;


    Phi = ((!(int16_t)idx) & 1 ) - ((!((int16_t)idx - FPTRU_N)) & (1)) - ((!((int16_t)idx - FPTRU_N + 1)) & (1));
    Phi0 = 1;

    F[FPTRU_N - 1 - threadIdx.x] = f[threadIdx.x];
    
    
    V[idx] = 0;    
    F[FPTRU_N] = 0;
    V[0] = 0;

    __syncthreads();

    Delta = 1;

    S = 0;
    S0 = 1;
    

    //下面进入循环
#pragma unroll//循环展开
    for(int loop = 0;loop < 2 * FPTRU_N - 1; ++loop){
        
        tmpV = V[idx-1];
        tmpF = F[idx];
        ff0 = F[0];
        
        //__syncthreads();//全读完才能写 这有什么样的区别呢

        swap = int16_negative_mask(-Delta) & int16_nonzero_mask(ff0);
       
        t = swap & (Phi ^ tmpF);
        Phi ^= t;
        tmpF ^= t;
        t = swap & (tmpV ^ S);//读共享内存
        tmpV ^= t;//写共享内存
        S ^= t;
        
        vv0 = 0;

        t = swap & (Phi0 ^ ff0); 
        Phi0 ^= t;
        ff0 ^= t;
        t = swap & (vv0 ^ S0);
        vv0 ^= t;
        S0 ^= t;

        Delta ^= swap & (Delta ^ -Delta);
        Delta++;

        
        phi0 = Phi0;
        f0 = ff0;

        tmpF = fq_freeze(phi0 * tmpF - f0 * Phi);
        
        S = fq_freeze(phi0 * S - f0 * tmpV);
        S0 = fq_freeze(phi0 * S0 - f0 * vv0);
        __syncthreads();//全读完才能写
        F[threadIdx.x] = tmpF;
        F[FPTRU_N] = 0; //TODO:存储体冲突

        V[idx] = tmpV;
        V[0] = vv0;
        __syncthreads();//全写完才能读
        
    }


    scale = fq_inverse(Phi0);

    finv[threadIdx.x] = fq_freeze(scale * (int32_t)V[FPTRU_N - 1 - threadIdx.x]);

}

/*2024-10-15:
线程组织形式:<<<BATCH_SIZE,639>>>*/
__device__ void rq_inverse_clean_1277(int16_t *finv,const int16_t *f){
    __shared__ int16_t V[FPTRU_N + 1],F[FPTRU_N + 1],S[FPTRU_N + 1],Phi[FPTRU_N + 1];//n太大，因此可以尽情使用共享内存
    int16_t tmp1,tmp2;

    int i, loop, Delta, swap, t;
    int32_t Phi0, F0;
    int16_t scale;

    int idx = threadIdx.x;

    
    Delta = 1;
    Phi[idx * 2] = 0;
    Phi[idx * 2 + 1] = 0;

    V[idx * 2] = 0;
    V[idx * 2 + 1] = 0;

    S[idx * 2] = 0;
    S[idx * 2 + 1] = 0;

    if(idx == FPTRU_N / 2){ //0-638   
        F[FPTRU_N - 1 - idx * 2] = f[idx * 2];
    }
    else{
        F[FPTRU_N - 1 - idx * 2] = f[idx * 2];
        F[FPTRU_N - 1 - (idx * 2 + 1)] = f[idx * 2 + 1];
    }

    if(idx == FPTRU_N / 2){
        Phi[0] = 1;
        Phi[FPTRU_N - 1] = Phi[FPTRU_N] = -1;
        S[0] = 1;
        F[FPTRU_N] = 0;
    }
    
    //初始赋值无问题
   

    for(loop = 0;loop < 2* FPTRU_N - 1; ++loop){
        tmp1 = V[idx * 2];
        tmp2 = V[(idx * 2 + 1) * (idx != FPTRU_N/2)];
        __syncthreads();
        V[idx * 2 + 1] = tmp1;
        V[(idx * 2 + 2) * (idx != FPTRU_N/2)] = tmp2 * (idx != FPTRU_N/2); //减少分支语句
        __syncthreads();

        // if(idx == FPTRU_N / 2 && (loop == 1 || loop == 2)){
        //     printf("in kernel\n");
        //     for(int i =0;i<=FPTRU_N;i++) printf("%d ",V[i]);
        //     printf("\n");
        // }

        // __syncthreads();

        swap = int16_negative_mask(-Delta) & int16_nonzero_mask(F[0]);
        __syncthreads();//防止对于F[0]的读写

        t = swap & (Phi[idx * 2] ^ F[idx * 2]);
        Phi[idx * 2] ^= t;
        F[idx * 2] ^= t;
        t = swap & (V[idx * 2] ^ S[idx * 2]);
        V[idx * 2] ^= t;
        S[idx * 2] ^= t;

        t = swap & (Phi[idx * 2 + 1] ^ F[idx * 2 + 1]);
        Phi[idx * 2 + 1] ^= t;
        F[idx * 2 + 1] ^= t;
        t = swap & (V[idx * 2 + 1] ^ S[idx * 2 + 1]);
        V[idx * 2 + 1] ^= t;
        S[idx * 2 + 1] ^= t;

        Delta ^= swap & (Delta ^ -Delta);
        Delta++;

        __syncthreads();
        Phi0 = Phi[0];
        F0 = F[0];
        __syncthreads();
        F[idx * 2] = fq_freeze(Phi0 * F[idx * 2] - F0 * Phi[idx * 2]);
        F[idx * 2 + 1] = fq_freeze(Phi0 * F[idx * 2 + 1] - F0 * Phi[idx * 2 + 1]);

        S[idx * 2] = fq_freeze(Phi0 * S[idx * 2] - F0 * V[idx * 2]);
        S[idx * 2 + 1] = fq_freeze(Phi0 * S[idx * 2 + 1] - F0 * V[idx * 2 + 1]);

        __syncthreads();

        tmp1 = F[idx * 2 + 1];
        tmp2 = F[(idx * 2 + 2) * (idx != FPTRU_N/2)];

        __syncthreads();
        F[idx * 2] = tmp1;
        F[idx * 2 + 1] = tmp2 * (idx != FPTRU_N/2);
    }

    scale = fq_inverse(Phi[0]);
    finv[idx * 2] = fq_freeze(scale * (int32_t)V[FPTRU_N - 1 - idx * 2]);
    if(idx != FPTRU_N/2) finv[idx * 2 + 1] = fq_freeze(scale * (int32_t)V[FPTRU_N - 1 - (idx * 2 + 1)]);
    
}



__device__ void poly_inverse_1277(poly *b, const poly *a)
{
  rq_inverse_clean_1277(b->coeffs,a->coeffs);
}

__global__ void poly_inv_1277(poly * finv, poly * f){
    poly_inverse_1277(&finv[blockIdx.x],&f[blockIdx.x]);
}

/*2024-7-8
输入:c:待输出的多项式
    sigma:待加上的多项式
    msg:待encode的明文
输出:多项式c

概述:完成一次poly_encode_compress操作,线程组织形式为<<<1,FPTRU_N>>>*/
__device__ void poly_encode_compress_1(poly * c, poly * sigma, unsigned char * msg){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if(tid >= 16 * FPTRU_MSGBYTES){
        int16_t t = (int32_t)((fqcsubq(sigma->coeffs[tid]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
        c->coeffs[tid] = t & (FPTRU_Q2 - 1);
    }
    else{
        int i = tid >> 3;//tid / 8;
        int j = tid & 7 ;//tid % 8;
        int k = i >> 1;//i / 2;

        uint8_t tmp = (msg[k] >> ((i %2 ) << 2)) & 0xF;
        uint8_t mh = encode_e8(tmp);

        int16_t mask = -(int16_t)((mh >> j) & 1);
        int16_t t = (int32_t)((fqcsubq(sigma->coeffs[tid]) << FPTRU_LOGQ2) + (FPTRU_Q >> 1)) / FPTRU_Q;
        t = t + (mask & (FPTRU_Q2 >> 1));//使得大于16 * FPTRU_MSGBYTES 为0
        c->coeffs[tid] = t & (FPTRU_Q2 - 1);
    }
}

__global__ void poly_fqcsubq_encode_compress_batch(poly * c, poly * sigma, unsigned char * msg, int interval){
    poly_encode_compress_1(&c[blockIdx.x],&sigma[blockIdx.x],&msg[interval * blockIdx.x]);
}

__device__ int32_t sqr(int32_t x)
{
    return x * x;
}

__device__ uint32_t abs_q2(uint32_t x)
{
    uint32_t mask = -(((FPTRU_Q2 - (x << 1)) >> 31) & 1);
    return (mask & (FPTRU_Q2 - x)) | ((~mask) & x);
}

__device__ uint32_t const_abs(int32_t x)
{
    uint32_t mask = x >> 31;
    return (x ^ mask) - mask;
}

__device__ uint8_t decode_d8_00(uint32_t *cost, uint32_t tmp_cost[8][2])
{
    uint8_t m[2] = {0, 0}, xor_sum = 0, res;
    uint32_t min_diff = ~0U >> 2, r;
    uint32_t min_i = 0;
    int i;

    *cost = 0;
    for (i = 0; i < 4; i++)
    {
        uint32_t c[2];
        c[0] = tmp_cost[i << 1][0] + tmp_cost[i << 1 | 1][0];
        c[1] = tmp_cost[i << 1][1] + tmp_cost[i << 1 | 1][1];
        r = ((c[1] - c[0]) >> 31) & 1;
        m[0] |= r << i;
        xor_sum ^= r;

        uint32_t tmp = ((-(r ^ 1)) & (uint32_t)c[0]) ^ ((-(r & 1)) & (uint32_t)c[1]);
        uint32_t tmp_xor = ((-(r & 1)) & (uint32_t)c[0]) ^ ((-(r ^ 1)) & (uint32_t)c[1]);

        *cost += tmp;
        uint32_t diff = tmp_xor - tmp;

        r = ((diff - min_diff) >> 31) & 1;
        min_diff = ((-r) & diff) | ((-(r ^ 1)) & min_diff);
        min_i = ((-r) & i) | ((-(r ^ 1)) & min_i);
    }
    m[1] = m[0] ^ (1 << min_i);
    res = ((-(xor_sum ^ 1)) & (uint32_t)m[0]) ^ ((-(xor_sum & 1)) & (uint32_t)m[1]);
    *cost += (-xor_sum) & min_diff;
    return res;
}

__device__ uint8_t decode_d8_10(uint32_t *cost, uint32_t tmp_cost[8][2])
{
    uint8_t m[2] = {0, 0}, xor_sum = 0, res;
    uint32_t min_diff = ~0U >> 2, r;
    uint32_t min_i = 0;
    int i;

    *cost = 0;
    for (i = 0; i < 4; i++)
    {
        uint32_t c[2];
        c[0] = tmp_cost[i << 1][1] + tmp_cost[i << 1 | 1][0];
        c[1] = tmp_cost[i << 1][0] + tmp_cost[i << 1 | 1][1];
        r = ((c[1] - c[0]) >> 31) & 1;
        m[0] |= r << i;
        xor_sum ^= r;

        uint32_t tmp = ((-(r ^ 1)) & (uint32_t)c[0]) ^ ((-(r & 1)) & (uint32_t)c[1]);
        uint32_t tmp_xor = ((-(r & 1)) & (uint32_t)c[0]) ^ ((-(r ^ 1)) & (uint32_t)c[1]);

        *cost += tmp;
        uint32_t diff = tmp_xor - tmp;
        r = ((diff - min_diff) >> 31) & 1;
        min_diff = ((-r) & diff) | ((-(r ^ 1)) & min_diff);
        min_i = ((-r) & i) | ((-(r ^ 1)) & min_i);
    }
    m[1] = m[0] ^ (1 << min_i);
    res = ((-(xor_sum ^ 1)) & (uint32_t)m[0]) ^ ((-(xor_sum & 1)) & (uint32_t)m[1]);
    *cost += (-xor_sum) & min_diff;
    return res;
}

__device__ uint8_t decode_e8(uint32_t vec[8])
{
    uint32_t cost[2], r;
    uint8_t m[2], res;
    uint32_t tmp_cost[8][2];
    int i;

    for (i = 0; i < 8; i++)
    {
        tmp_cost[i][0] = sqr(abs_q2(vec[i]));
        tmp_cost[i][1] = sqr(const_abs(vec[i] - (FPTRU_Q2 >> 1)));
    }

    m[0] = decode_d8_00(cost + 0, tmp_cost);
    m[1] = decode_d8_10(cost + 1, tmp_cost);
    r = ((cost[1] - cost[0]) >> 31) & 1;

    res = ((-(r ^ 1)) & (uint32_t)m[0]) ^ ((-(r & 1)) & (uint32_t)m[1]);
    res = ((((res ^ (res << 1)) & 0x3) | ((res >> 1) & 4)) << 1) | r;

    return res;
}

__device__ void poly_decode(unsigned char * m, poly *mp){
    int idx = threadIdx.x;
    uint32_t tmp_mp[8];
    
    uint8_t msg_l,msg_h;

    tmp_mp[0] = (uint32_t)mp->coeffs[16 * idx];
    tmp_mp[1] = (uint32_t)mp->coeffs[16 * idx + 1];
    tmp_mp[2] = (uint32_t)mp->coeffs[16 * idx + 2];
    tmp_mp[3] = (uint32_t)mp->coeffs[16 * idx + 3];
    tmp_mp[4] = (uint32_t)mp->coeffs[16 * idx + 4];
    tmp_mp[5] = (uint32_t)mp->coeffs[16 * idx + 5];
    tmp_mp[6] = (uint32_t)mp->coeffs[16 * idx + 6];
    tmp_mp[7] = (uint32_t)mp->coeffs[16 * idx + 7];

    msg_l = decode_e8(tmp_mp);

    tmp_mp[0] = (uint32_t)mp->coeffs[16 * idx + 8];
    tmp_mp[1] = (uint32_t)mp->coeffs[16 * idx + 9];
    tmp_mp[2] = (uint32_t)mp->coeffs[16 * idx + 10];
    tmp_mp[3] = (uint32_t)mp->coeffs[16 * idx + 11];
    tmp_mp[4] = (uint32_t)mp->coeffs[16 * idx + 12];
    tmp_mp[5] = (uint32_t)mp->coeffs[16 * idx + 13];
    tmp_mp[6] = (uint32_t)mp->coeffs[16 * idx + 14];
    tmp_mp[7] = (uint32_t)mp->coeffs[16 * idx + 15];

    msg_h = decode_e8(tmp_mp);

    m[threadIdx.x] = (msg_h << 4) + msg_l;
}

__global__ void poly_decode_batch(unsigned char *m, poly *mp){
    poly_decode(&m[blockIdx.x * (FPTRU_PREFIXHASHBYTES + FPTRU_MSGBYTES)],&mp[blockIdx.x]);
}


