#include "n653.h"

#include "iostream"


__device__ int32_t zetas_n653[N_N653 / 7] = {
    10676304, 16761025, 7394872, 16761025, 11121377, 10949150, 7394872, 16761025, 3850100, 7874710, 7862122, 664151, 11121377, 10949150, 7394872, 16761025, 7912042, 14765008, 12738274, 10067325, 16116200, 1189525, 9719104, 16499514, 3850100, 7874710, 7862122, 664151, 11121377, 10949150, 7394872, 16761025, 98522, 4692110, 1288047, 3170012, 16116200, 7862122, 11436071, 12085043, 16212669, 2963283, 9719104, 664151, 10655357, 3627434, 10377718, 13813870, 7912042, 3850100, 1620141, 9051474, 14325862, 12901574, 14765008, 12927053, 15588451, 7583861, 8878623, 290849, 12738274, 7874710, 1572208, 9193292, 4692110, 1126178, 7862122, 10949150, 2963283, 12075328, 3627434, 15650975, 3850100, 11121377, 12901574, 14329275, 7583861, 8673499, 7874710, 5655776, 1126178, 13522707, 12075328, 10649318, 11121377, 7394872, 8673499, 3254446, 13522707, 2596593, 7394872, 16761025, 2596593, 2580465, 2580465, 14180560, 2981589, 15393923, 12412334, 12519308, 14110054, 1590746, 3731610, 15588906, 11857296, 10676304, 864588, 6965437, 11091924, 9920313, 15605542, 15532409, 6191036, 7435780, 600298, 9622180, 9021882, 10966002, 12884094, 1918092, 16647180, 10403679, 10533652, 11973559, 12998707, 1025148, 4383065, 13802193, 9419128, 14587874, 1361894, 3551173, 4768624, 14519764, 9751140, 10588057, 6370682, 12559778, 13982193, 11735271, 14530231, 14772227, 9387670, 11392596, 98522, 1288047, 1189525, 16116200, 11436071, 12097024, 16212669, 9719104, 10283588, 10655357, 10377718, 16499514, 7912042, 1620141, 10485252, 14325862, 14765008, 439146, 15588451, 8878623, 10067325, 12738274, 1572208, 5611087, 4692110, 7862122, 3170012, 2963283, 3627434, 664151, 3850100, 12901574, 9051474, 7583861, 7874710, 290849, 1126178, 12075328, 10949150, 11121377, 8673499, 14329275, 13522707, 7394872, 10649318, 2596593, 2580465, 16761025};

__device__ int32_t zetas_inv_n653[N_N653 / 7] = {
    5384557, 7389483, 2004926, 2246922, 5041882, 2794960, 4217375, 10406471, 6189096, 7026013, 2257389, 12008529, 13225980, 15415259, 2189279, 7358025, 2974960, 12394088, 15752005, 3778446, 4803594, 6243501, 6373474, 129973, 14859061, 3893059, 5811151, 7755271, 7154973, 16176855, 9341373, 10586117, 1244744, 1171611, 6856840, 5685229, 9811716, 15912565, 6100849, 4919857, 1188247, 13045543, 15186407, 2667099, 4257845, 4364819, 1383230, 13795564, 11166066, 15204945, 4038879, 6709828, 7898530, 1188702, 16338007, 2012145, 2451291, 6291901, 15157012, 8865111, 277639, 6399435, 6121796, 6493565, 7058049, 564484, 4680129, 5341082, 660953, 15587628, 15489106, 16678631, 16486304, 8902443, 9193292, 7725679, 3875579, 12927053, 16113002, 13149719, 13813870, 13607141, 8915031, 12085043, 2447878, 8103654, 5655776, 5828003, 4701825, 15650975, 6127835, 9382281, 3254446, 14196688, 14180560, 16761025, 11166066, 16486304, 15204945, 9193292, 6709828, 8902443, 7898530, 290849, 16338007, 7725679, 2012145, 12927053, 6291901, 3875579, 15157012, 9051474, 277639, 16113002, 6399435, 13813870, 6493565, 13149719, 7058049, 664151, 4680129, 13607141, 5341082, 12085043, 15587628, 8915031, 15489106, 3170012, 16486304, 2447878, 8902443, 5655776, 7725679, 8103654, 3875579, 14329275, 16113002, 5828003, 13149719, 15650975, 13607141, 4701825, 8915031, 10949150, 2447878, 6127835, 8103654, 3254446, 5828003, 9382281, 4701825, 10649318, 6127835, 14196688, 9382281, 16761025, 14196688, 14180560, 14180560, 2580465, 277639, 7058049, 15587628, 660953, 6709828, 4038879, 2012145, 8865111, 16113002, 8915031, 8902443, 12927053, 5828003, 5655776, 9382281, 16761025, 16113002, 8915031, 8902443, 12927053, 5828003, 5655776, 9382281, 16761025, 5828003, 5655776, 9382281, 16761025, 9382281, 16761025, 16761025};


    //6层基2,1层基3
__device__ int32_t zetas_n653_v2[N_N653 / 7] = {10676304,16761025, 7394872, 16761025, 11121377, 10949150, 7394872, 16761025, 3850100, 7874710, 7862122, 664151, 11121377, 10949150, 7394872, 16761025, 7912042, 14765008, 12738274, 10067325, 16116200, 1189525, 9719104, 16499514, 3850100, 7874710, 7862122, 664151, 11121377, 10949150, 7394872, 16761025, 16647180, 12998707, 14587874, 9419128, 13982193, 9387670, 14519764, 12559778, 3731610, 864588, 15393923, 1590746, 15532409, 15605542, 9622180, 1918092, 7912042, 14765008, 12738274, 10067325, 16116200, 1189525, 9719104, 16499514, 3850100, 7874710, 7862122, 664151, 11121377, 10949150, 7394872, 16761025, 2981589, 98522, 15393923, 1189525, 12519308, 1288047, 14110054, 16678631, 3731610, 16116200, 15588906, 12097024, 10676304, 11436071, 864588, 660953, 11091924, 16212669, 9920313, 10283588, 15532409, 9719104, 6191036, 564484, 600298, 10655357, 9622180, 16499514, 10966002, 10377718, 12884094, 6121796, 16647180, 7912042, 10403679, 10485252, 11973559, 1620141, 12998707, 8865111, 4383065, 14325862, 13802193, 439146, 14587874, 14765008, 1361894, 2451291, 4768624, 15588451, 14519764, 10067325, 10588057, 8878623, 6370682, 1188702, 13982193, 12738274, 11735271, 5611087, 14772227, 1572208, 9387670, 4038879, 98522, 4692110, 1288047, 3170012, 16116200, 7862122, 11436071, 12085043, 16212669, 2963283, 9719104, 664151, 10655357, 3627434, 10377718, 13813870, 7912042, 3850100, 1620141, 9051474, 14325862, 12901574, 14765008, 12927053, 15588451, 7583861, 8878623, 290849, 12738274, 7874710, 1572208, 9193292, 4692110, 1126178, 7862122, 10949150, 2963283, 12075328, 3627434, 15650975, 3850100, 11121377, 12901574, 14329275, 7583861, 8673499, 7874710, 5655776, 1126178, 13522707, 12075328, 10649318, 11121377, 7394872, 8673499, 3254446, 13522707, 2596593, 7394872, 16761025, 2596593, 2580465, 2580465, 14180560};


__device__ int32_t zetas_inv_n653_v2[N_N653 / 7] = {5384557, 11166066, 7389483, 4038879, 2246922, 15204945, 5041882, 5611087, 4217375, 6709828, 10406471, 1188702, 7026013, 7898530, 2257389, 10067325, 13225980, 16338007, 15415259, 2451291, 7358025, 2012145, 2974960, 439146, 15752005, 6291901, 3778446, 8865111, 6243501, 15157012, 6373474, 10485252, 14859061, 277639, 3893059, 6121796, 7755271, 6399435, 7154973, 16499514, 9341373, 6493565, 10586117, 564484, 1171611, 7058049, 6856840, 10283588, 9811716, 4680129, 15912565, 660953, 4919857, 5341082, 1188247, 12097024, 15186407, 15587628, 2667099, 16678631, 4364819, 15489106, 1383230, 1189525, 11166066, 16486304, 15204945, 9193292, 6709828, 8902443, 7898530, 290849, 16338007, 7725679, 2012145, 12927053, 6291901, 3875579, 15157012, 9051474, 277639, 16113002, 6399435, 13813870, 6493565, 13149719, 7058049, 664151, 4680129, 13607141, 5341082, 12085043, 15587628, 8915031, 15489106, 3170012, 16486304, 2447878, 8902443, 5655776, 7725679, 8103654, 3875579, 14329275, 16113002, 5828003, 13149719, 15650975, 13607141, 4701825, 8915031, 10949150, 2447878, 6127835, 8103654, 3254446, 5828003, 9382281, 4701825, 10649318, 6127835, 14196688, 9382281, 16761025, 14196688, 14180560, 14180560, 2580465, 14859061, 7154973, 1171611, 1244744, 15186407, 1383230, 15912565, 13045543, 4217375, 2257389, 7389483, 2794960, 7358025, 2189279, 3778446, 129973, 277639, 7058049, 15587628, 660953, 6709828, 4038879, 2012145, 8865111, 16113002, 8915031, 8902443, 12927053, 5828003, 5655776, 9382281, 16761025, 277639, 7058049, 15587628, 660953, 6709828, 4038879, 2012145, 8865111, 16113002, 8915031, 8902443, 12927053, 5828003, 5655776, 9382281, 16761025, 16113002, 8915031, 8902443, 12927053, 5828003, 5655776, 9382281, 16761025, 5828003, 5655776, 9382281, 16761025, 9382281, 16761025, 16761025};

__device__ int32_t root3_n653[3] = {16128, 2580465, 14180560};



__device__ int32_t montgomery_reduce_n653_cuda(int64_t a)
{
    int32_t t;
    t = (int32_t)a * NUMBER_ACONFI;//t是低32位置
    t = (a - (int64_t)t * Q_N653) >> 32;
    return t;
}

/*2024-7-11
蒙格玛丽约减法的ptx版本:
*/
__device__ int32_t montgomery_reduce_n653_ptx(int64_t a){
    int64_t res;
    asm(
        "{\n\t"
        //".reg .s64 b;\n\t"
        "mul.lo.s64 %0,%1,%2;\n\t"
        //"and.b64 %0,%0,0xffffffff;\n\t"
        "shr.s64 %0,%0,32;\n\t"
        "mul.lo.s64 %0,%0,%3;\n\t"
        "sub.s64 %0,%1,%0;\n\t"
        "shr.s64 %0,%0,32;\n\t" 
        //"add.s64 %0,%0,%3;\n\t"       
        "}"
    :"=l"(res):"l"(a),"n"(-4610542247161626624),"n"(Q_N653));  //-4610542247161626624 = NUMBER_ACONFI << 32
    return (int32_t)res;
}

__device__ int32_t pseudomersenne_reduce_single_n653_cuda(int32_t a)
{
    int32_t t0, t1;

    t0 = a & 0xffffff;//a的低24bit
    t1 = a >> 24;//右移24bit后的结果
    t0 = (t1 << 6) + t0 - t1 - Q_N653;

    return t0;
}

__global__ void ntt_big_n653_kernel(int32_t a[N_N653]){
    __shared__ int32_t poly_a[N_N653];//shared memory
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x,idx;//idx指向线程tid处理的数
    unsigned int len;
    unsigned int zeta_start=1;

    int32_t t;
    int32_t zeta;


    if(tid < N_N653/2){
        poly_a[tid] = a[tid];
        poly_a[tid+N_N653/2]=a[tid+N_N653/2];

        //第一层基2NTT
        __syncthreads();//控制从全局内存到共享内存的读写是完成的->用于控制线程块内的同步
        len = N_N653 / 2;
        idx = tid;
        t = poly_a[idx + len];//访问bank的一行
        poly_a[idx + len]=poly_a[idx] + t;//写入bank的一行
        poly_a[idx] = poly_a[idx] - t;//写入bank的一行


        //四层基2NTT

        for(len = N_N653 / 4 ; len >= 42;len >>= 1 ){
            zeta_start = zeta_start << 1;
            __syncthreads();

            zeta = zetas_n653[zeta_start + ( tid / len )];//不同的tid可能访问相同的idx TODO:这里能不能改成常量内存
            //printf("%d\n",zeta);
            //这里存在分支执行,能不能把不做乘法的放到一个warp里面->如果实现不了，那为了避免空转，都用蒙哥马利乘法
            idx = ( tid / len ) * len * 2 + tid % len;//TODO:这里的取模操作是不是可以优化一下,调用inverse.cpp里面已经实现的代码
            //printf("%d %d\n",tid,idx);
            /*
            if(zeta == 16761025){
                t = poly_a[idx + len];//访问bank的一行
                poly_a[idx + len]=poly_a[idx] + t;//写入bank的一行
                poly_a[idx] = poly_a[idx] - t;//写入bank的一行
            }
            else{*/
                t = montgomery_reduce_n653_cuda((int64_t)zeta *poly_a[idx + len]);//访问bank的一行
                poly_a[idx + len]=poly_a[idx] - t;//写入bank的一行
                poly_a[idx] = poly_a[idx] + t;//写入bank的一行
            //}
            
        }

        //一层基3NTT
        int32_t tb, tc, tpho, zeta1, zeta2;
        zeta_start = zeta_start << 1;
        //if (tid == 0 ) printf("%d\n",zeta_start);
        len = 14;

        int32_t a_value;
        __syncthreads();
        if(tid < N_N653 / 3){
            zeta1 = zetas_n653[zeta_start + (tid / len) * 2];//TODO:注意考虑这里是否会发生bank的冲突
            zeta2 = zetas_n653[zeta_start + (tid / len) * 2 + 1];
    
            idx = ( tid / len ) * len * 3 + tid % len;
            
            tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * poly_a[idx + len]);
            tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * poly_a[idx + len + len]);
            tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));
            
            a_value = poly_a[idx];
            poly_a[idx + len + len] = (a_value - tb -tpho);//TODO:这里是否能够将a[idx]存储到寄存器，这样减少访问的开销
            poly_a[idx + len] = (a_value - tc + tpho);
            poly_a[idx] = (a_value + tb + tc);
        }


        //最后一层基2NTT
        len = 7 ;
        __syncthreads();
        zeta_start += 64;//基3NTT共32组，每组使用2个zeta
        //printf("my pid is %d %d %d\n",tid, zeta_start, zeta_start + ( tid / len ));
        zeta = zetas_n653[zeta_start + ( tid / len )];
        idx = ( tid / len ) * len * 2 + tid % len;//TODO:这里的取模操作是不是可以优化一下,调用inverse.cpp里面已经实现的代码
        /*if(zeta == 16761025){
            t = poly_a[idx + len];//访问bank的一行
            poly_a[idx + len]=poly_a[idx] + t;//写入bank的一行
            poly_a[idx] = poly_a[idx] - t;//写入bank的一行
        }
        else{*/
            t = montgomery_reduce_n653_cuda((int64_t)zeta *poly_a[idx + len]);
            poly_a[idx + len]=pseudomersenne_reduce_single_n653_cuda( poly_a[idx] - t );//TODO:这里是否会发生bank的冲突
            poly_a[idx] = pseudomersenne_reduce_single_n653_cuda(poly_a[idx] + t);
        //}
        
        
        //将数据写回全局内存
        __syncthreads();
        a[tid] = poly_a[tid];
        a[tid+N_N653/2]=poly_a[tid+N_N653/2];
    }
    
}

/*2024-4-19:
输入:
    c:设备端多项式，用于存储相乘的结果
    a:设备端多项式
    b:设备端多项式
    
处理流程:先将素阶数域上的多项式转化到伪梅森数环上，进行NTT操作后，做点乘，然后再做逆NTT操作，最后返回到素阶数域上
*/
void poly_mul_653_batch_q1(poly *array_c,poly * array_a,poly *array_b,cudaStream_t stream,int batch_size){
    //printf("in poly_mul_653_batch_q1\n");
    nttpoly_n653 * array_ntta;
    nttpoly_n653 * array_nttb;
    nttpoly_n653 * array_nttc;


    HANDLE_ERROR(cudaMalloc((void **)&array_ntta,sizeof(nttpoly_n653)*batch_size));
    HANDLE_ERROR(cudaMalloc((void **)&array_nttb,sizeof(nttpoly_n653)*batch_size));
    HANDLE_ERROR(cudaMalloc((void **)&array_nttc,sizeof(nttpoly_n653)*batch_size));

    
    poly_extend_n653_batch<<<batch_size,1,0,stream>>>(array_ntta,array_a);

    poly_extend_n653_batch<<<batch_size,1,0,stream>>>(array_nttb,array_b);

    poly_ntt_big_n653_batch<<<batch_size,N_N653/2,0,stream>>>(array_ntta);
    poly_ntt_small_n653_batch<<<batch_size,N_N653/2,0,stream>>>(array_nttb);

    poly_basemul_n653_batch<<<batch_size,1,0,stream>>>(array_nttc,array_ntta,array_nttb);

    poly_invntt_n653_batch<<<batch_size,N_N653/2,0,stream>>>(array_nttc);

    poly_extract_n653_q1_batch<<<batch_size,1,0,stream>>>(array_c,array_nttc);

    cudaStreamSynchronize(stream);//等待kernel完成再释放内存
    
    HANDLE_ERROR(cudaFree(array_ntta));
    HANDLE_ERROR(cudaFree(array_nttb));
    HANDLE_ERROR(cudaFree(array_nttc));


}

/*2024-5-4:
用于测试cuda版本的下述功能为什么错误*/
__device__ void poly_extract_n653_q1_test(poly *b, const nttpoly_n653 *a)
{
    b->coeffs[0] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (a->coeffs[0] + a->coeffs[FPTRU_N])));
    for (int i = 1; i <= FPTRU_N - 2; i++)
        b->coeffs[i] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (a->coeffs[i] + a->coeffs[i + FPTRU_N - 1] + a->coeffs[i + FPTRU_N])));
    b->coeffs[FPTRU_N - 1] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (a->coeffs[FPTRU_N - 1] + a->coeffs[2 * FPTRU_N - 2])));
}

/*2024-4-30:
输入:array_c:全局内存
    array_a:全局内存
    array_b:全局内存
<<<BATCH_SIZE,N_N653/2>>>

增加了进行调试的接口*/
__global__ void poly_mul_653_batch_q1_v2_test(poly *array_c,poly * array_a,poly *array_b,nttpoly_n653 * array_ntta,nttpoly_n653 * array_nttb){
    //共享内存用作需要传递的数据
    __shared__ nttpoly_n653 ntta;
    __shared__ nttpoly_n653 nttb;
    __shared__ nttpoly_n653 nttc;

    
    poly a;
    poly b;
    int idx = threadIdx.x;
    int32_t zeta;

    //只要一个线程来做
    a = array_a[blockIdx.x];
    b = array_b[blockIdx.x];

    //做poly_extend
    ntta.coeffs[idx*2]=0;
    ntta.coeffs[idx*2+1]=0;

    nttb.coeffs[idx*2]=0;
    nttb.coeffs[idx*2+1]=0;
    
    if(idx*2 < FPTRU_N) {
        ntta.coeffs[idx*2]=a.coeffs[idx*2];
        nttb.coeffs[idx*2]=b.coeffs[idx*2];
    }
    if(idx*2 + 1< FPTRU_N){
        ntta.coeffs[idx*2 + 1]=a.coeffs[idx*2 + 1];
        nttb.coeffs[idx*2 + 1]=b.coeffs[idx*2 + 1];
    }

    __syncthreads();//上述正确性已验证


    ntt_big_n653(ntta.coeffs);
    ntt_small_n653(nttb.coeffs);


    __syncthreads();

    if(idx < N_N653 /14){ //空转的改进一下
        zeta = zetas_n653[N_N653 /14 + idx];
        basemul_n653((&nttc)->coeffs + 14 *idx, (&ntta)->coeffs + 14 * idx, (&nttb)->coeffs + 14 * idx, zeta);

        basemul_n653((&nttc)->coeffs + 14 *idx + 7, (&ntta)->coeffs + 14 * idx + 7, (&nttb)->coeffs + 14 * idx + 7 , -zeta);
    }
    
    __syncthreads();//basemul验证完成


    invntt_tomont_n653(nttc.coeffs);//求逆正确

    /*正确性测试
    array_ntta[blockIdx.x] = nttc; 
    return;*/

    //TODO:为什么这里出现了错误
    //经过验证，应该是代码当下函数关于poly_extract_n653_q1_test的改写出现了问题
    /*
    if(idx == 0){
        //printf("use poly_extract_n653_q1_test\n");
        poly_extract_n653_q1_test(&array_c[blockIdx.x],&nttc);
    }

    return;*/

    //__syncthreads();//会一直变化，可能是线程组织出现了问题,
    /*if(blockIdx.x ==0) {
        printf("start idx=%d\n",idx);
    }*/
    //__syncthreads();
    if(idx == 0){
        //array_c[blockIdx.x].coeffs[0] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[0] + nttc.coeffs[FPTRU_N])));
        /*if(blockIdx.x ==0) {
            printf("in[0] idx=%d\n",idx);
        }*/

        array_c[blockIdx.x].coeffs[0] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[0] + nttc.coeffs[FPTRU_N])));
    }
    if(idx == FPTRU_N-1){
        //array_c[blockIdx.x].coeffs[FPTRU_N - 1] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[FPTRU_N - 1] + nttc.coeffs[2 * FPTRU_N - 2])));
        /*if(blockIdx.x ==0) {
            printf("in[654] idx=%d\n",idx);
        }*/
        array_c[blockIdx.x].coeffs[FPTRU_N - 1] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[FPTRU_N - 1] + nttc.coeffs[2 * FPTRU_N - 2])));
    }
    if( 1<= idx){
        if(idx <=FPTRU_N-2 ){
            /*if(blockIdx.x ==0) {
            printf("in[m] idx=%d\n",idx);//这里为什么能输出in[m] idx=665
        }*/
        //array_c[blockIdx.x].coeffs[idx] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[idx] + nttc.coeffs[idx + FPTRU_N - 1] + nttc.coeffs[idx + FPTRU_N])));

        array_c[blockIdx.x].coeffs[idx] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[idx] + nttc.coeffs[idx + FPTRU_N - 1] + nttc.coeffs[idx + FPTRU_N])));
        }
    }
    /*else{
        if(blockIdx.x ==0) {
            printf("end idx=%d\n",idx);
        }
    }*/
}

/*2024-5-5:
根绝正确的的poly_mul_653_batch_q1_v2改编得到*/
__device__ poly poly_mul_653_batch_q1_v2_device_test(poly * array_a,poly *array_b){
    //共享内存用作需要传递的数据
    __shared__ nttpoly_n653 ntta;
    __shared__ nttpoly_n653 nttb;
    __shared__ nttpoly_n653 nttc;
    __shared__ poly f_res;

    
    poly a;
    poly b;
    int idx = threadIdx.x;
    int32_t zeta;

    //只要一个线程来做
    a = array_a[blockIdx.x];
    b = array_b[blockIdx.x];

    //做poly_extend
    ntta.coeffs[idx*2]=0;
    ntta.coeffs[idx*2+1]=0;

    nttb.coeffs[idx*2]=0;
    nttb.coeffs[idx*2+1]=0;
    
    if(idx*2 < FPTRU_N) {
        ntta.coeffs[idx*2]=a.coeffs[idx*2];
        nttb.coeffs[idx*2]=b.coeffs[idx*2];
    }
    if(idx*2 + 1< FPTRU_N){
        ntta.coeffs[idx*2 + 1]=a.coeffs[idx*2 + 1];
        nttb.coeffs[idx*2 + 1]=b.coeffs[idx*2 + 1];
    }

    __syncthreads();
    ntt_big_n653(ntta.coeffs);
    ntt_small_n653(nttb.coeffs);


    __syncthreads();

    if(idx < N_N653 /14){ //空转的改进一下
        zeta = zetas_n653[N_N653 /14 + idx];
        basemul_n653((&nttc)->coeffs + 14 *idx, (&ntta)->coeffs + 14 * idx, (&nttb)->coeffs + 14 * idx, zeta);

        basemul_n653((&nttc)->coeffs + 14 *idx + 7, (&ntta)->coeffs + 14 * idx + 7, (&nttb)->coeffs + 14 * idx + 7 , -zeta);
    }
    
    __syncthreads();

    invntt_tomont_n653(nttc.coeffs);

    if(idx == 0){
        f_res.coeffs[0] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[0] + nttc.coeffs[FPTRU_N])));
    }
    if(idx == FPTRU_N-1){
        f_res.coeffs[FPTRU_N - 1] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[FPTRU_N - 1] + nttc.coeffs[2 * FPTRU_N - 2])));
    }
    if( 1<= idx){
        if(idx <=FPTRU_N-2 ){
            f_res.coeffs[idx] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[idx] + nttc.coeffs[idx + FPTRU_N - 1] + nttc.coeffs[idx + FPTRU_N])));
        }
    }
    __syncthreads();//TODO:由于是共享内存，会不会提前返回呢？
    return f_res;
}

/*2024-5-4:
根绝正确的的poly_mul_653_batch_q1_v2改编得到*/
__device__ void poly_mul_653_batch_q1_v2_device(poly *array_c,poly * array_a,poly *array_b){
    //共享内存用作需要传递的数据
    __shared__ nttpoly_n653 ntta;
    __shared__ nttpoly_n653 nttb;
    __shared__ nttpoly_n653 nttc;

    __shared__ poly res;

    
    poly a;
    poly b;
    int idx = threadIdx.x;
    int32_t zeta;

    //只要一个线程来做
    a = array_a[blockIdx.x];
    //__syncthreads();//TODO:防止读写冲突->感觉是没有必要的
    b = array_b[blockIdx.x];
    /*if(idx == 0){//已经验证成功
        for(int i=0;i<FPTRU_N;i++){
            printf("%d,",a.coeffs[i]);
        }
        printf("\n");

        for(int i=0;i<FPTRU_N;i++){
            printf("%d,",b.coeffs[i]);
        }
        printf("\n");
    }*/

    //做poly_extend
    ntta.coeffs[idx*2]=0;
    ntta.coeffs[idx*2+1]=0;

    nttb.coeffs[idx*2]=0;
    nttb.coeffs[idx*2+1]=0;
    
    if(idx*2 < FPTRU_N) {
        ntta.coeffs[idx*2]=a.coeffs[idx*2];
        nttb.coeffs[idx*2]=b.coeffs[idx*2];
    }
    if(idx*2 + 1< FPTRU_N){
        ntta.coeffs[idx*2 + 1]=a.coeffs[idx*2 + 1];
        nttb.coeffs[idx*2 + 1]=b.coeffs[idx*2 + 1];
    }

    __syncthreads();
    ntt_big_n653(ntta.coeffs);
    ntt_small_n653(nttb.coeffs);


    __syncthreads();

    if(idx < N_N653 /14){ //空转的改进一下
        zeta = zetas_n653[N_N653 /14 + idx];
        basemul_n653((&nttc)->coeffs + 14 *idx, (&ntta)->coeffs + 14 * idx, (&nttb)->coeffs + 14 * idx, zeta);

        basemul_n653((&nttc)->coeffs + 14 *idx + 7, (&ntta)->coeffs + 14 * idx + 7, (&nttb)->coeffs + 14 * idx + 7 , -zeta);
    }
    
    __syncthreads();

    invntt_tomont_n653(nttc.coeffs);

    /*if(idx == 0){ //输出查看一下
        printf("invntt\n");
        for(int i=0;i<N_N653;i++){
            printf("%d,",nttc.coeffs[i]);
        }
        printf("\n");
    }*/
    __syncthreads();

    if(idx == 0){
        array_c[blockIdx.x].coeffs[0] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[0] + nttc.coeffs[FPTRU_N])));
        //res.coeffs[0] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[0] + nttc.coeffs[FPTRU_N])));
    }
    if(idx == FPTRU_N-1){
        array_c[blockIdx.x].coeffs[FPTRU_N - 1] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[FPTRU_N - 1] + nttc.coeffs[2 * FPTRU_N - 2])));
        //res.coeffs[FPTRU_N - 1] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[FPTRU_N - 1] + nttc.coeffs[2 * FPTRU_N - 2])));

    }
    if( 1<= idx){
        if(idx <=FPTRU_N-2 ){
            array_c[blockIdx.x].coeffs[idx] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[idx] + nttc.coeffs[idx + FPTRU_N - 1] + nttc.coeffs[idx + FPTRU_N])));
            //res.coeffs[idx] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[idx] + nttc.coeffs[idx + FPTRU_N - 1] + nttc.coeffs[idx + FPTRU_N])));
        }
    }
    //__syncthreads();

    /*if(idx == 0){//已经验证成功
        for(int i=0;i<FPTRU_N;i++){
            printf("%d,",res.coeffs[i]);
        }
        printf("\n");
    }*/

    //__syncthreads();//TODO:由于是共享内存，会不会提前返回呢？
}

/*2024-4-24:
输入:array_c:全局内存
    array_a:全局内存
    array_b:全局内存
<<<BATCH_SIZE,N_N653/2>>>

2024-5-4修改:已经验证正确性*/
__global__ void poly_mul_653_batch_q1_v2(poly *array_c,poly * array_a,poly *array_b){
    //共享内存用作需要传递的数据
    __shared__ nttpoly_n653 ntta;
    __shared__ nttpoly_n653 nttb;
    __shared__ nttpoly_n653 nttc;

    
    poly a;
    poly b;
    int idx = threadIdx.x;
    int32_t zeta;

    //只要一个线程来做
    a = array_a[blockIdx.x];
    b = array_b[blockIdx.x];

    //做poly_extend
    ntta.coeffs[idx*2]=0;
    ntta.coeffs[idx*2+1]=0;

    nttb.coeffs[idx*2]=0;
    nttb.coeffs[idx*2+1]=0;
    
    if(idx*2 < FPTRU_N) {
        ntta.coeffs[idx*2]=a.coeffs[idx*2];
        nttb.coeffs[idx*2]=b.coeffs[idx*2];
    }
    if(idx*2 + 1< FPTRU_N){
        ntta.coeffs[idx*2 + 1]=a.coeffs[idx*2 + 1];
        nttb.coeffs[idx*2 + 1]=b.coeffs[idx*2 + 1];
    }

    __syncthreads();
    ntt_big_n653(ntta.coeffs);
    ntt_small_n653(nttb.coeffs);


    __syncthreads();

    if(idx < N_N653 /14){ //空转的改进一下
        zeta = zetas_n653[N_N653 /14 + idx];
        basemul_n653((&nttc)->coeffs + 14 *idx, (&ntta)->coeffs + 14 * idx, (&nttb)->coeffs + 14 * idx, zeta);

        basemul_n653((&nttc)->coeffs + 14 *idx + 7, (&ntta)->coeffs + 14 * idx + 7, (&nttb)->coeffs + 14 * idx + 7 , -zeta);
    }
    
    __syncthreads();

    invntt_tomont_n653(nttc.coeffs);

    //TODO:这里要不要加上线程同步呢？
    __syncthreads();//TODO:device函数不加会错误

    if(idx == 0){
        array_c[blockIdx.x].coeffs[0] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[0] + nttc.coeffs[FPTRU_N])));
    }
    if(idx == FPTRU_N-1){
        array_c[blockIdx.x].coeffs[FPTRU_N - 1] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[FPTRU_N - 1] + nttc.coeffs[2 * FPTRU_N - 2])));
    }
    if( 1<= idx){
        if(idx <=FPTRU_N-2 ){
            array_c[blockIdx.x].coeffs[idx] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[idx] + nttc.coeffs[idx + FPTRU_N - 1] + nttc.coeffs[idx + FPTRU_N])));
        }
    }
    
}

/*2024-5-18:
查看ntt的速度，寻找性能瓶颈*/
__global__ void poly_mul_653_batch_q1_zhc(poly *array_c,poly * array_a,poly *array_b){
    //共享内存用作需要传递的数据
    __shared__ nttpoly_n653 ntta;
    __shared__ nttpoly_n653 nttb;
    __shared__ nttpoly_n653 nttc;

    
    poly a;
    poly b;
    int idx = threadIdx.x;
    int32_t zeta;

    //只要一个线程来做
    a = array_a[blockIdx.x];
    b = array_b[blockIdx.x];

    //做poly_extend
    
    ntta.coeffs[idx*2]=0;
    ntta.coeffs[idx*2+1]=0;

    nttb.coeffs[idx*2]=0;
    nttb.coeffs[idx*2+1]=0;
    
    if(idx*2 < FPTRU_N) {
        ntta.coeffs[idx*2]=a.coeffs[idx*2];
        nttb.coeffs[idx*2]=b.coeffs[idx*2];
    }
    if(idx*2 + 1< FPTRU_N){
        ntta.coeffs[idx*2 + 1]=a.coeffs[idx*2 + 1];
        nttb.coeffs[idx*2 + 1]=b.coeffs[idx*2 + 1];
    }

    __syncthreads();
    ntt_big_n653_v2(nttc.coeffs);
    /*ntt_small_n653(nttb.coeffs);


    __syncthreads();

    if(idx < N_N653 /14){ //空转的改进一下
        zeta = zetas_n653[N_N653 /14 + idx];
        basemul_n653((&nttc)->coeffs + 14 *idx, (&ntta)->coeffs + 14 * idx, (&nttb)->coeffs + 14 * idx, zeta);

        basemul_n653((&nttc)->coeffs + 14 *idx + 7, (&ntta)->coeffs + 14 * idx + 7, (&nttb)->coeffs + 14 * idx + 7 , -zeta);
    }
    
    __syncthreads();

    invntt_tomont_n653(nttc.coeffs);

    //TODO:这里要不要加上线程同步呢？
    __syncthreads();//TODO:device函数不加会错误

    if(idx == 0){
        array_c[blockIdx.x].coeffs[0] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[0] + nttc.coeffs[FPTRU_N])));
    }
    if(idx == FPTRU_N-1){
        array_c[blockIdx.x].coeffs[FPTRU_N - 1] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[FPTRU_N - 1] + nttc.coeffs[2 * FPTRU_N - 2])));
    }
    if( 1<= idx){
        if(idx <=FPTRU_N-2 ){
            array_c[blockIdx.x].coeffs[idx] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[idx] + nttc.coeffs[idx + FPTRU_N - 1] + nttc.coeffs[idx + FPTRU_N])));
        }
    }*/
    

}

/*2024-4-27:
输入:c:共享内存
    a:共享内存
    array_b:全局内存，只含一个多项式

输出:计算fmul*array_b并存储到fmul中*/
__device__ void poly_mul_653_batch_q1_v2_device_v2(poly*array_c, poly* array_a,poly *array_b){
    return;
    /*
    //共享内存用作需要传递的数据
    __shared__ nttpoly_n653 ntta;
    __shared__ nttpoly_n653 nttb;
    __shared__ nttpoly_n653 nttc;

    
    poly a;
    poly b;
    int idx = threadIdx.x;
    int32_t zeta;

    //只要一个线程来做
    a = array_a[blockIdx.x];
    b = array_b[blockIdx.x];

    //做poly_extend
    ntta.coeffs[idx*2]=0;
    ntta.coeffs[idx*2+1]=0;

    nttb.coeffs[idx*2]=0;
    nttb.coeffs[idx*2+1]=0;
    
    if(idx*2 < FPTRU_N) {
        ntta.coeffs[idx*2]=a.coeffs[idx*2];
        nttb.coeffs[idx*2]=b.coeffs[idx*2];
    }
    if(idx*2 + 1< FPTRU_N){
        ntta.coeffs[idx*2 + 1]=a.coeffs[idx*2 + 1];
        nttb.coeffs[idx*2 + 1]=b.coeffs[idx*2 + 1];
    }

    __syncthreads();
    
    ntt_big_n653(ntta.coeffs);
    ntt_small_n653(nttb.coeffs);


    __syncthreads();

    if(idx < N_N653 /14){ //空转的改进一下
        zeta = zetas_n653[N_N653 /14 + idx];
        basemul_n653((&nttc)->coeffs + 14 *idx, (&ntta)->coeffs + 14 * idx, (&nttb)->coeffs + 14 * idx, zeta);

        basemul_n653((&nttc)->coeffs + 14 *idx + 7, (&ntta)->coeffs + 14 * idx + 7, (&nttb)->coeffs + 14 * idx + 7 , -zeta);
    }
    
    __syncthreads();

    invntt_tomont_n653(nttc.coeffs);*/
    /*
    if(idx == 0){
        array_c[blockIdx.x].coeffs[0] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[0] + nttc.coeffs[FPTRU_N])));
    }
    else if(idx == FPTRU_N-1){
        array_c[blockIdx.x].coeffs[FPTRU_N - 1] = (montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[FPTRU_N - 1] + nttc.coeffs[2 * FPTRU_N - 2]))) & (FPTRU_Q2 - 1);
    }
    else if(0<idx<FPTRU_N-1){
        array_c[blockIdx.x].coeffs[idx] = (montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[idx] + nttc.coeffs[idx + FPTRU_N - 1] + nttc.coeffs[idx + FPTRU_N]))) & (FPTRU_Q2 - 1);
    }*/
}
/*
__device__ void poly_mul_653_batch_q1_v2_device(poly array_c, poly array_a,poly *array_b){
    //共享内存用作需要传递的数据
    __shared__ nttpoly_n653 ntta;
    __shared__ nttpoly_n653 nttb;
    __shared__ nttpoly_n653 nttc;

    
    poly a;
    poly b;
    int idx = threadIdx.x;
    int32_t zeta;

    //只要一个线程来做
    a = array_a;
    b = array_b[blockIdx.x];

    //做poly_extend
    ntta.coeffs[idx*2]=0;
    ntta.coeffs[idx*2+1]=0;

    nttb.coeffs[idx*2]=0;
    nttb.coeffs[idx*2+1]=0;
    
    if(idx*2 < FPTRU_N) {
        ntta.coeffs[idx*2]=a.coeffs[idx*2];
        nttb.coeffs[idx*2]=b.coeffs[idx*2];
    }
    if(idx*2 + 1< FPTRU_N){
        ntta.coeffs[idx*2 + 1]=a.coeffs[idx*2 + 1];
        nttb.coeffs[idx*2 + 1]=b.coeffs[idx*2 + 1];
    }

    __syncthreads();
    
    ntt_big_n653(ntta.coeffs);
    ntt_small_n653(nttb.coeffs);


    __syncthreads();

    if(idx < N_N653 /14){ //空转的改进一下
        zeta = zetas_n653[N_N653 /14 + idx];
        basemul_n653((&nttc)->coeffs + 14 *idx, (&ntta)->coeffs + 14 * idx, (&nttb)->coeffs + 14 * idx, zeta);

        basemul_n653((&nttc)->coeffs + 14 *idx + 7, (&ntta)->coeffs + 14 * idx + 7, (&nttb)->coeffs + 14 * idx + 7 , -zeta);
    }
    
    __syncthreads();

    invntt_tomont_n653(nttc.coeffs);

    if(idx == 0){
        array_c.coeffs[0] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[0] + nttc.coeffs[FPTRU_N])));
    }
    else if(idx == FPTRU_N-1){
        array_c.coeffs[FPTRU_N - 1] = (montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[FPTRU_N - 1] + nttc.coeffs[2 * FPTRU_N - 2]))) & (FPTRU_Q2 - 1);
    }
    else if(0<idx<FPTRU_N-1){
        array_c.coeffs[idx] = (montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[idx] + nttc.coeffs[idx + FPTRU_N - 1] + nttc.coeffs[idx + FPTRU_N]))) & (FPTRU_Q2 - 1);
    }
}
*/

/*2024-4-18
输入:array_nttb:设备端多项式数组，用于存储拓展后的结果
    array_a:设备端多项式数组

输出:存储在nttpoly_n653中

线程组织形式:<<<BATCH_SIZE,1>>>

处理流程:进行数据分块，使得每一个poly_extend_n653函数处理一个多项式的拓展
*/
__global__ void poly_extend_n653_batch(nttpoly_n653 * array_nttb,poly * array_a){
    poly_extend_n653(&(array_nttb[blockIdx.x]),&(array_a[blockIdx.x]));
}

/*2024-4-18:
输入:b:设备端多项式,用于存储拓展后的结果
    a:设备端多项式，待处理的多项式

输出:存储在b中

//TODO:这里是否需要变成多个kernel的形式
*/
__device__ void poly_extend_n653(nttpoly_n653 * b,poly * a){
    int i;
    for (i = 0; i < FPTRU_N; i++)
        b->coeffs[i] = a->coeffs[i];
    for (; i < N_N653; i++)
        b->coeffs[i] = 0;
}

/*2024-4-18:
输入:array_a:设备端待进行NTT处理的多项式数组
    
输出:存放在array_a中

线程组织形式:<<<BATCH_SIZE,N_N653/2>>>
*/
__global__ void poly_ntt_big_n653_batch(nttpoly_n653 * array_a){
    ntt_big_n653((array_a[blockIdx.x]).coeffs);//array_a[BATCH_SIZE]
}


/*2024-4-18:
输入:a:设备端待进行NTT处理的多项式，以数组的形式进行表示

输出:存储在输入中

处理流程:N_N653/2个线程协同处理一个大多项式的NTT操作*/
__device__ void ntt_big_n653(int32_t a[N_N653]){
    __shared__ int32_t poly_a[N_N653];//shared memory
    unsigned int tid = threadIdx.x,idx;//idx指向线程tid处理的数
    unsigned int len;
    unsigned int zeta_start=1;

    int32_t t;
    int32_t zeta;


    if(tid < N_N653/2){
        poly_a[tid] = a[tid];
        poly_a[tid+N_N653/2]=a[tid+N_N653/2];

        //第一层基2NTT
        __syncthreads();//控制从全局内存到共享内存的读写是完成的->用于控制线程块内的同步
        len = N_N653 / 2;
        idx = tid;
        t = poly_a[idx + len];//访问bank的一行
        poly_a[idx + len]=poly_a[idx] + t;//写入bank的一行
        poly_a[idx] = poly_a[idx] - t;//写入bank的一行


        //四层基2NTT

        for(len = N_N653 / 4 ; len >= 42;len >>= 1 ){
            zeta_start = zeta_start << 1;
            __syncthreads();

            zeta = zetas_n653[zeta_start + ( tid / len )];//不同的tid可能访问相同的idx TODO:这里能不能改成常量内存
            //printf("%d\n",zeta);
            //这里存在分支执行,能不能把不做乘法的放到一个warp里面->如果实现不了，那为了避免空转，都用蒙哥马利乘法
            idx = ( tid / len ) * len * 2 + tid % len;//TODO:这里的取模操作是不是可以优化一下,调用inverse.cpp里面已经实现的代码
            //printf("%d %d\n",tid,idx);
            
            if(zeta == 16761025){
                t = poly_a[idx + len];//访问bank的一行
                poly_a[idx + len]=poly_a[idx] + t;//写入bank的一行
                poly_a[idx] = poly_a[idx] - t;//写入bank的一行
            }
            else{
                t = montgomery_reduce_n653_cuda((int64_t)zeta *poly_a[idx + len]);//访问bank的一行
                poly_a[idx + len]=poly_a[idx] - t;//写入bank的一行
                poly_a[idx] = poly_a[idx] + t;//写入bank的一行
            }
            
        }

        //一层基3NTT
        int32_t tb, tc, tpho, zeta1, zeta2;
        zeta_start = zeta_start << 1;
        //if (tid == 0 ) printf("%d\n",zeta_start);
        len = 14;

        int32_t a_value;
        __syncthreads();
        if(tid < N_N653 / 3){
            zeta1 = zetas_n653[zeta_start + (tid / len) * 2];//TODO:注意考虑这里是否会发生bank的冲突
            zeta2 = zetas_n653[zeta_start + (tid / len) * 2 + 1];
    
            idx = ( tid / len ) * len * 3 + tid % len;
            
            tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * poly_a[idx + len]);
            tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * poly_a[idx + len + len]);
            tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));
            
            a_value = poly_a[idx];
            poly_a[idx + len + len] = (a_value - tb -tpho);//TODO:这里是否能够将a[idx]存储到寄存器，这样减少访问的开销
            poly_a[idx + len] = (a_value - tc + tpho);
            poly_a[idx] = (a_value + tb + tc);
        }


        //最后一层基2NTT
        len = 7 ;
        __syncthreads();
        zeta_start += 64;//基3NTT共32组，每组使用2个zeta
        //printf("my pid is %d %d %d\n",tid, zeta_start, zeta_start + ( tid / len ));
        zeta = zetas_n653[zeta_start + ( tid / len )];
        idx = ( tid / len ) * len * 2 + tid % len;//TODO:这里的取模操作是不是可以优化一下,调用inverse.cpp里面已经实现的代码
        if(zeta == 16761025){
            t = poly_a[idx + len];//访问bank的一行
            poly_a[idx + len]=poly_a[idx] + t;//写入bank的一行
            poly_a[idx] = poly_a[idx] - t;//写入bank的一行
        }
        else{
            t = montgomery_reduce_n653_cuda((int64_t)zeta *poly_a[idx + len]);
            poly_a[idx + len]=pseudomersenne_reduce_single_n653_cuda( poly_a[idx] - t );//TODO:这里是否会发生bank的冲突
            poly_a[idx] = pseudomersenne_reduce_single_n653_cuda(poly_a[idx] + t);
        }
        
        
        //将数据写回全局内存
        __syncthreads();
        a[tid] = poly_a[tid];
        a[tid+N_N653/2]=poly_a[tid+N_N653/2];
    }
}

__device__ void ntt_big_n653_v2(int32_t poly_a[N_N653]){
    //__shared__ int32_t poly_a[N_N653];//shared memory
    unsigned int tid = threadIdx.x,idx;//idx指向线程tid处理的数
    unsigned int len;
    unsigned int zeta_start=1;

    int32_t t;
    int32_t zeta;


    if(tid < N_N653/2){// 
        /*poly_a[tid] = a[tid];
        poly_a[tid+N_N653/2]=a[tid+N_N653/2];*/

        /*poly_a[tid * 2] = a[tid * 2];
        poly_a[tid * 2 + 1]=a[tid * 2 + 1];*/

        
        //第一层基2NTT
        //__syncthreads();//控制从全局内存到共享内存的读写是完成的->用于控制线程块内的同步
        len = N_N653 / 2;
        idx = tid;
        t = poly_a[idx + len];//访问bank的一行
        poly_a[idx + len]=poly_a[idx] + t;//写入bank的一行
        poly_a[idx] = poly_a[idx] - t;//写入bank的一行


        //四层基2NTT

        for(len = N_N653 / 4 ; len >= 42;len >>= 1 ){
            zeta_start = zeta_start << 1;
            __syncthreads();

            zeta = zetas_n653[zeta_start + ( tid / len )];//不同的tid可能访问相同的idx TODO:这里能不能改成常量内存
            //printf("%d\n",zeta);
            //这里存在分支执行,能不能把不做乘法的放到一个warp里面->如果实现不了，那为了避免空转，都用蒙哥马利乘法
            idx = ( tid / len ) * len * 2 + tid % len;//TODO:这里的取模操作是不是可以优化一下,调用inverse.cpp里面已经实现的代码
            //printf("%d %d\n",tid,idx);
            
            if(zeta == 16761025){
                t = poly_a[idx + len];//访问bank的一行
                poly_a[idx + len]=poly_a[idx] + t;//写入bank的一行
                poly_a[idx] = poly_a[idx] - t;//写入bank的一行
            }
            else{
                t = montgomery_reduce_n653_cuda((int64_t)zeta *poly_a[idx + len]);//访问bank的一行
                poly_a[idx + len]=poly_a[idx] - t;//写入bank的一行
                poly_a[idx] = poly_a[idx] + t;//写入bank的一行
            }
            
        }

        //一层基3NTT
        int32_t tb, tc, tpho, zeta1, zeta2;
        zeta_start = zeta_start << 1;
        //if (tid == 0 ) printf("%d\n",zeta_start);
        len = 14;

        int32_t a_value;
        __syncthreads();
        if(tid < N_N653 / 3){
            zeta1 = zetas_n653[zeta_start + (tid / len) * 2];//TODO:注意考虑这里是否会发生bank的冲突
            zeta2 = zetas_n653[zeta_start + (tid / len) * 2 + 1];
    
            idx = ( tid / len ) * len * 3 + tid % len;
            
            tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * poly_a[idx + len]);
            tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * poly_a[idx + len + len]);
            tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));
            
            a_value = poly_a[idx];
            poly_a[idx + len + len] = (a_value - tb -tpho);//TODO:这里是否能够将a[idx]存储到寄存器，这样减少访问的开销
            poly_a[idx + len] = (a_value - tc + tpho);
            poly_a[idx] = (a_value + tb + tc);
        }


        //最后一层基2NTT
        len = 7 ;
        __syncthreads();
        zeta_start += 64;//基3NTT共32组，每组使用2个zeta
        //printf("my pid is %d %d %d\n",tid, zeta_start, zeta_start + ( tid / len ));
        zeta = zetas_n653[zeta_start + ( tid / len )];
        idx = ( tid / len ) * len * 2 + tid % len;//TODO:这里的取模操作是不是可以优化一下,调用inverse.cpp里面已经实现的代码
        if(zeta == 16761025){
            t = poly_a[idx + len];//访问bank的一行
            poly_a[idx + len]=poly_a[idx] + t;//写入bank的一行
            poly_a[idx] = poly_a[idx] - t;//写入bank的一行
        }
        else{
            t = montgomery_reduce_n653_cuda((int64_t)zeta *poly_a[idx + len]);
            poly_a[idx + len]=pseudomersenne_reduce_single_n653_cuda( poly_a[idx] - t );//TODO:这里是否会发生bank的冲突
            poly_a[idx] = pseudomersenne_reduce_single_n653_cuda(poly_a[idx] + t);
        }
        
        
        //将数据写回全局内存
        //__syncthreads();
        //a[tid] = poly_a[tid];
        //a[tid+N_N653/2]=poly_a[tid+N_N653/2];
    }
}

/*2024-4-18:
输入:array_a:设备端待进行NTT处理的多项式数组
    
输出:存放在array_a中

线程组织形式:<<<BATCH_SIZE,N_N653/2>>>
*/
__global__ void poly_ntt_small_n653_batch(nttpoly_n653 * array_a){
    ntt_small_n653((array_a[blockIdx.x]).coeffs);//array_a[BATCH_SIZE]
}


/*2024-4-18:
输入:a:设备端待进行NTT处理的多项式，以数组的形式进行表示

输出:存储在输入中

处理流程:N_N653/2个线程协同处理一个小多项式的NTT操作*/

__device__ void ntt_small_n653(int32_t a[N_N653]){
    __shared__ int32_t poly_a[N_N653];//shared memory
    unsigned int tid = threadIdx.x,idx;//idx指向线程tid处理的数
    unsigned int len;
    unsigned int zeta_start=1;

    int32_t t;
    int32_t zeta;


    if(tid < N_N653/2){
        poly_a[tid] = a[tid];
        poly_a[tid+N_N653/2]=a[tid+N_N653/2];

        //第一层基2NTT
        __syncthreads();//控制从全局内存到共享内存的读写是完成的->用于控制线程块内的同步
        len = N_N653 / 2;
        idx = tid;
        t = poly_a[idx + len];//访问bank的一行
        poly_a[idx + len]=poly_a[idx] + t;//写入bank的一行
        poly_a[idx] = poly_a[idx] - t;//写入bank的一行

        //第二层基2NTT
        __syncthreads();
        len = N_N653 / 4;
        idx = ( tid / len ) * len * 2 + tid % len;
        // tid /len = 0 ,则成乘上FACTOR_CNSHHUI；否则乘上 -1
        //zeta = tid / len - 1;//tid /len =0 时，全1；
        //zeta = (zeta & FACTOR_CNSHHUI) + (zeta ^ (-1)); //TODO:这里的zeta可能会存在问题
        if(tid / len == 0){
            zeta = FACTOR_CNSHHUI;
        }
        else zeta = -1;

        t = poly_a[idx + len] * zeta;
        poly_a[idx + len] = poly_a[idx] - t;
        poly_a[idx] = poly_a[idx] + t;//DEBUG:这里符号出错

        //3层基2NTT
        zeta_start = zeta_start << 1;
        for(len = N_N653 / 8 ; len >= 42;len >>= 1 ){
            zeta_start = zeta_start << 1;
            __syncthreads();

            zeta = zetas_n653[zeta_start + ( tid / len )];//不同的tid可能访问相同的idx TODO:这里能不能改成常量内存
            //printf("%d\n",zeta);
            //这里存在分支执行,能不能把不做乘法的放到一个warp里面->如果实现不了，那为了避免空转，都用蒙哥马利乘法
            idx = ( tid / len ) * len * 2 + tid % len;//TODO:这里的取模操作是不是可以优化一下,调用inverse.cpp里面已经实现的代码
            //printf("%d %d\n",tid,idx);
            
            if(zeta == 16761025){
                t = poly_a[idx + len];//访问bank的一行
                poly_a[idx + len]=poly_a[idx] + t;//写入bank的一行
                poly_a[idx] = poly_a[idx] - t;//写入bank的一行
            }
            else{
                t = montgomery_reduce_n653_cuda((int64_t)zeta *poly_a[idx + len]);//访问bank的一行
                poly_a[idx + len]=poly_a[idx] - t;//写入bank的一行
                poly_a[idx] = poly_a[idx] + t;//写入bank的一行
            }  
        }

        //一层基3NTT
        int32_t tb, tc, tpho, zeta1, zeta2;
        zeta_start = zeta_start << 1;
        //if (tid == 0 ) printf("%d\n",zeta_start);
        len = 14;

        int32_t a_value;
        __syncthreads();
        if(tid < N_N653 / 3){
            zeta1 = zetas_n653[zeta_start + (tid / len) * 2];//TODO:注意考虑这里是否会发生bank的冲突
            zeta2 = zetas_n653[zeta_start + (tid / len) * 2 + 1];
    
            idx = ( tid / len ) * len * 3 + tid % len;
            
            tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * poly_a[idx + len]);
            tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * poly_a[idx + len + len]);
            tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));
            
            a_value = poly_a[idx];
            poly_a[idx + len + len] = (a_value - tb -tpho);//TODO:这里是否能够将a[idx]存储到寄存器，这样减少访问的开销
            poly_a[idx + len] = (a_value - tc + tpho);
            poly_a[idx] = (a_value + tb + tc);
        }


        //最后一层基2NTT
        len = 7 ;
        __syncthreads();
        zeta_start += 64;//基3NTT共32组，每组使用2个zeta
        //printf("my pid is %d %d %d\n",tid, zeta_start, zeta_start + ( tid / len ));
        zeta = zetas_n653[zeta_start + ( tid / len )];
        idx = ( tid / len ) * len * 2 + tid % len;//TODO:这里的取模操作是不是可以优化一下,调用inverse.cpp里面已经实现的代码
        if(zeta == 16761025){
            t = poly_a[idx + len];//访问bank的一行
            poly_a[idx + len]=poly_a[idx] + t;//写入bank的一行
            poly_a[idx] = poly_a[idx] - t;//写入bank的一行
        }
        else{
            t = montgomery_reduce_n653_cuda((int64_t)zeta *poly_a[idx + len]);
            poly_a[idx + len]=pseudomersenne_reduce_single_n653_cuda( poly_a[idx] - t );//TODO:这里是否会发生bank的冲突
            poly_a[idx] = pseudomersenne_reduce_single_n653_cuda(poly_a[idx] + t);
        }
        
        
        //将数据写回全局内存
        __syncthreads();
        a[tid] = poly_a[tid];
        a[tid+N_N653/2]=poly_a[tid+N_N653/2];
    }
}

/*2024-4-18:
输入:array_c:设备端多项式数组
    array_a:设备端多项式数组
    array_b:设备端多项式数组
    
输出:封装BATCH_SIZE个多项式的乘法,最终使得array_c=array_a*array_b
*/
__global__ void poly_basemul_n653_batch(nttpoly_n653 *array_c, nttpoly_n653 *array_a, nttpoly_n653 *array_b){
    poly_basemul_n653(&(array_c[blockIdx.x]),&(array_a[blockIdx.x]),&(array_b[blockIdx.x]));
}

__device__ void poly_basemul_n653(nttpoly_n653 *c, nttpoly_n653 *a, nttpoly_n653 *b){
    int i;
    int32_t zeta;
    for (i = 0; i < N_N653 / 14; i++)
    {
        zeta = zetas_n653[N_N653 / 14 + i];//TODO:这里为什么还需要使用zetas？->对应了多项式所在的域->位置是如何确定的
        basemul_n653(c->coeffs + 14 * i, a->coeffs + 14 * i, b->coeffs + 14 * i, zeta);
        basemul_n653(c->coeffs + 14 * i + 7, a->coeffs + 14 * i + 7, b->coeffs + 14 * i + 7, -zeta);
    }
}

#define KARA(a, b, x, y, d) ((montgomery_reduce_n653_cuda((int64_t)(a[x] + a[y]) * (b[x] + b[y])) - d[x] - d[y]))
/*2024-4-7:
输入:c:存储计算结果 c=a*b mod x^7-zeta
    a:待处理的多项式a
    b:待处理的多项式b
输出:存储到c上
处理流程:依据原理c[k]=add(a[i]b[j],i+j=k) + zeta * add(a[i]b[j],i+j=k+7),利用karasuba技巧，先计算a[i]b[i],然后依次处理每一个系数*/
__device__ void basemul_n653(int32_t c[7], const int32_t a[7], const int32_t b[7], const int32_t zeta)
{
    int i;
    int32_t d[7];

    for (i = 0; i < 7; i++)
        d[i] = montgomery_reduce_n653_cuda((int64_t)a[i] * b[i]);

    c[0] = (d[0] + montgomery_reduce_n653_cuda((int64_t)zeta * (KARA(a, b, 1, 6, d) + KARA(a, b, 2, 5, d) + KARA(a, b, 3, 4, d))));//感觉这里使用了karasuba技巧
    c[1] = (KARA(a, b, 0, 1, d) + montgomery_reduce_n653_cuda((int64_t)zeta * (KARA(a, b, 2, 6, d) + KARA(a, b, 3, 5, d) + d[4])));
    c[2] = (KARA(a, b, 0, 2, d) + d[1] + montgomery_reduce_n653_cuda((int64_t)zeta * (KARA(a, b, 3, 6, d) + KARA(a, b, 4, 5, d))));
    c[3] = (KARA(a, b, 0, 3, d) + KARA(a, b, 1, 2, d) + montgomery_reduce_n653_cuda((int64_t)zeta * (KARA(a, b, 4, 6, d) + d[5])));
    c[4] = (KARA(a, b, 0, 4, d) + KARA(a, b, 1, 3, d) + d[2] + montgomery_reduce_n653_cuda((int64_t)zeta * (KARA(a, b, 5, 6, d))));
    c[5] = (KARA(a, b, 0, 5, d) + KARA(a, b, 1, 4, d) + KARA(a, b, 2, 3, d) + montgomery_reduce_n653_cuda((int64_t)zeta * d[6]));
    c[6] = (KARA(a, b, 0, 6, d) + KARA(a, b, 1, 5, d) + KARA(a, b, 2, 4, d) + d[3]);
}

/*2024-4-19:
输入:array_a:设备端多项式数组array_a

输出:存储到array_a中

处理流程:数据分块后，调用invntt_tomont_n653对每一个多项式进行数据处理

线程组织形式:<<<BATCH_SIZE,N_N_653/2>>>*/
__global__ void poly_invntt_n653_batch(nttpoly_n653 * array_a){
    invntt_tomont_n653((array_a[blockIdx.x]).coeffs);//TODO:这里很奇怪,array_a是数组指针的形式吗？
}

/*2024-4-19:
输入:a:设备端以数组表示的多项式

输出:存储到a中

处理流程:N_N653/2个线程协同处理一个多项式的逆NTT操作*/
__device__ void invntt_tomont_n653(int32_t a[N_N653]){
    __shared__ int32_t poly_a[N_N653];//shared memory
    unsigned int tid = threadIdx.x,idx;//idx指向线程tid处理的数
    unsigned int len;
    unsigned int zeta_start = 0;

    int32_t t;
    int32_t zeta;


    if(tid < N_N653/2){
        poly_a[tid] = a[tid];
        poly_a[tid+N_N653/2]=a[tid+N_N653/2];

        //倒数1层基2NTT
        __syncthreads();

        len=7;
        zeta=zetas_inv_n653[zeta_start + (tid / len)];
        idx = ( tid / len ) * len * 2 + tid % len;
        if(zeta == NUMBER_AKNCF){ //注意:这里一定要有条件判断，否则会出现错误
            t = poly_a[idx];
            poly_a[idx] = (poly_a[idx + len] + t);
            poly_a[idx + len] = (poly_a[idx + len] -t);
        }
        else{
            t=poly_a[idx];
            poly_a[idx] = (t + poly_a[idx + len]);
            poly_a[idx + len] = (t - poly_a[idx + len]);
            poly_a[idx + len] = montgomery_reduce_n653_cuda((int64_t)zeta * poly_a[idx + len]);
        }
        

        //倒数第2层基3NTT
        int32_t tb, tc, tpho, zeta1, zeta2;
        len =14;
        zeta_start = 96;

        __syncthreads();
        if(tid < N_N653 / 3){
            
            zeta1 = zetas_inv_n653[zeta_start + (tid / len) * 2];//TODO:注意考虑这里是否会发生bank的冲突
            //if(tid == 0 ) printf("%d\n",zeta1);
            zeta2 = zetas_inv_n653[zeta_start + (tid / len) * 2 + 1];
            //if(tid == 0 ) printf("%d\n",zeta2);

            idx = ( tid / len ) * len * 3 + tid % len;

            tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (poly_a[idx + len] - poly_a[idx + len + len]));
            tb = (poly_a[idx] - poly_a[idx + len] - tpho);
            tc = (poly_a[idx] - poly_a[idx + len + len] + tpho);

            /*if(idx == 50){
                printf("[before cuda] tpho=%d tb=%d tc=%d a[j]=%d %d %d idx=%d len=%d\n",tpho,tb,tc,poly_a[idx],poly_a[idx+len],poly_a[idx+len+len],idx,len);
            }*/

            poly_a[idx] = (poly_a[idx] + poly_a[idx + len] + poly_a[idx + len + len]);
            poly_a[idx + len] = montgomery_reduce_n653_cuda((int64_t)zeta1 * tb);
            poly_a[idx + len + len] = montgomery_reduce_n653_cuda((int64_t)zeta2 * tc);

            /*if(idx == 50){
                printf("[after cuda] tpho=%d tb=%d tc=%d a[j]=%d %d %d idx=%d len=%d\n",tpho,tb,tc,poly_a[idx],poly_a[idx+len],poly_a[idx+len+len],idx,len);
            }*/
        }

        
        

        //4层基2NTT
        zeta_start = 160;

        for(len =42; len <= N_N653/4; len <<= 1){
            __syncthreads();
            zeta = zetas_inv_n653[zeta_start + (tid/len)];
            idx = ( tid / len) * len * 2 + tid % len;
            if(zeta == NUMBER_AKNCF){
                t=poly_a[idx];
                poly_a[idx]=(poly_a[idx + len] + t);
                poly_a[idx + len] = (poly_a[idx + len] - t);

            }
            else{
                t=poly_a[idx];
                poly_a[idx] = (t + poly_a[idx + len]);
                poly_a[idx + len] = (t - poly_a[idx + len]);
                poly_a[idx + len] = montgomery_reduce_n653_cuda((int64_t)zeta * poly_a[idx + len]);
            }
            

            zeta_start += N_N653 / (len * 2);

        }


        //最后一层基2NTT
        len = N_N653 / 2;
        __syncthreads();
        zeta = zetas_inv_n653[zeta_start + (tid / len)];
        idx = tid;
        t = poly_a[idx];
        poly_a[idx] = pseudomersenne_reduce_single_n653_cuda(poly_a[idx + len] + t);
        poly_a[idx + len] = pseudomersenne_reduce_single_n653_cuda(poly_a[idx + len] - t);

        //将数据写回全局内存
        __syncthreads();
        a[tid] = poly_a[tid];
        a[tid+N_N653/2]=poly_a[tid+N_N653/2];
    }
}

/*2024-4-19:
输入:array_a:设备端素阶数域上的多项式数组
    array_ntta:设备端伪梅森数缓缓上的多项式数组

输出:存储到array_a中

处理流程:将数据分块后调用poly_extract_n653

线程组织形式:<<<BATCH_SIZE,1>>>
*/
__global__ void poly_extract_n653_q1_batch(poly * array_a ,nttpoly_n653 * array_ntta){
    poly_extract_n653_q1(&(array_a[blockIdx.x]),&(array_ntta[blockIdx.x]));
}

/*2024-4-19:
输入:b:设备端素阶数域多项式
    a:设备端伪梅森环上的多项式

输出:存储到b中

处理流程:TODO:细看一下*/
__device__ void poly_extract_n653_q1(poly *b,nttpoly_n653 *a){
    b->coeffs[0] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (a->coeffs[0] + a->coeffs[FPTRU_N])));
    for (int i = 1; i <= FPTRU_N - 2; i++)
        b->coeffs[i] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (a->coeffs[i] + a->coeffs[i + FPTRU_N - 1] + a->coeffs[i + FPTRU_N])));
    b->coeffs[FPTRU_N - 1] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (a->coeffs[FPTRU_N - 1] + a->coeffs[2 * FPTRU_N - 2])));
}

#define special_CT_a(i,j)  t = a[j];a[j]=(a[i] + t);a[i] = (a[i] - t);
#define special_CT_b(i,j)  t = b[j];b[j]=(b[i] + t);b[i] = (b[i] - t);

#define special_CT_a_pse(i,j)  t = a[i];a[i]=pseudomersenne_reduce_single_n653_cuda(a[j] + t);a[j] = pseudomersenne_reduce_single_n653_cuda(a[j] - t);

#define CT_a(i,j,zeta)  t = montgomery_reduce_n653_cuda((int64_t)zeta * a[j]); a[j] = (a[i] - t); a[i] = (a[i] + t);

#define CT_b(i,j,zeta)  t = montgomery_reduce_n653_cuda((int64_t)zeta * b[j]); b[j] = (b[i] - t); b[i] = (b[i] + t);

#define GS_a(i,j,zeta) t = a[i]; a[i] = (t + a[j]); a[j] = (t - a[j]); a[j] = montgomery_reduce_n653_cuda((int64_t)zeta * a[j]);

/*2024-6-17:
<<<batch_size,168>>>
*/
__global__ void poly_mul_653_batch_q1_v3(poly *array_c,poly * array_a,poly *array_b){
    __shared__ nttpoly_n653 ntta;
    __shared__ nttpoly_n653 nttb;
    __shared__ nttpoly_n653 nttc;

    int32_t a[8] ={0};
    int32_t b[8] ={0};//注意数据类型和nttpoly_n653保持一致
    int idx = threadIdx.x;
    int32_t zeta;
    int32_t t;

    poly pa;
    poly pb;

    /*int c21 = idx / 21;
    int m21 = idx % 21;
    
    int c7 = idx / 7;
    int m7 = idx - c7 * 7;*/
    pa = array_a[blockIdx.x];
    pb = array_b[blockIdx.x];

    //不做extend，直接读8个系数
    a[0] = pa.coeffs[idx];//array_a[blockIdx.x].coeffs[idx];
    a[1] = pa.coeffs[idx + 168];//array_a[blockIdx.x].coeffs[idx + 168];
    a[2] = pa.coeffs[idx + 168 * 2];//array_a[blockIdx.x].coeffs[idx + 168 * 2];
    
    if(idx + 168 * 3 < FPTRU_N){
        a[3] = pa.coeffs[idx + 168 * 3];//array_a[blockIdx.x].coeffs[idx + 168 * 3];
    }
    
    //不做extend，直接读8个系数
    b[0] = pb.coeffs[idx];//array_b[blockIdx.x].coeffs[idx];
    b[1] = pb.coeffs[idx + 168];//array_b[blockIdx.x].coeffs[idx + 168];
    b[2] = pb.coeffs[idx + 168 * 2];//array_b[blockIdx.x].coeffs[idx + 168 * 2];
    
    if(idx + 168 * 3 < FPTRU_N){
        b[3] = pb.coeffs[idx + 168 * 3];//array_b[blockIdx.x].coeffs[idx + 168 * 3];
    }
    
    //一层基2
    //zeta = zetas_n653[1]; //旋转因子为-1
    special_CT_a(0,4);
    special_CT_a(1,5);
    special_CT_a(2,6);
    special_CT_a(3,7);

    special_CT_b(0,4);
    special_CT_b(1,5);
    special_CT_b(2,6);
    special_CT_b(3,7);

    

    //2 基2
    zeta = zetas_n653_v2[2];

    CT_a(0,2,zeta);
    CT_a(1,3,zeta);

    CT_b(0,2,zeta);
    CT_b(1,3,zeta);

    //zeta = zetas_n653[3]; //旋转因子为-1
    special_CT_a(4,6);
    special_CT_a(5,7);

    special_CT_b(4,6);
    special_CT_b(5,7);


    //3 基2
    zeta = zetas_n653_v2[4];
    CT_a(0,1,zeta);
    CT_b(0,1,zeta);

    zeta = zetas_n653_v2[5];
    CT_a(2,3,zeta);
    CT_b(2,3,zeta);

    zeta = zetas_n653_v2[6];
    CT_a(4,5,zeta);
    CT_b(4,5,zeta);

    special_CT_a(6,7);
    special_CT_b(6,7);

    //放回共享内存
    ntta.coeffs[idx] = a[0];
    ntta.coeffs[idx + 168] = a[1];
    ntta.coeffs[idx + 168 * 2] = a[2];
    ntta.coeffs[idx + 168 * 3] = a[3];
    ntta.coeffs[idx + 168 * 4] = a[4];
    ntta.coeffs[idx + 168 * 5] = a[5];
    ntta.coeffs[idx + 168 * 6] = a[6];
    ntta.coeffs[idx + 168 * 7] = a[7];

    nttb.coeffs[idx] = b[0];
    nttb.coeffs[idx + 168] = b[1];
    nttb.coeffs[idx + 168 * 2] = b[2];
    nttb.coeffs[idx + 168 * 3] = b[3];
    nttb.coeffs[idx + 168 * 4] = b[4];
    nttb.coeffs[idx + 168 * 5] = b[5];
    nttb.coeffs[idx + 168 * 6] = b[6];
    nttb.coeffs[idx + 168 * 7] = b[7];

    __syncthreads();

    /*if(idx == 0){
        for(int i=0;i<N_N653;i++){
            printf("%d,",ntta.coeffs[i]);
        }
        printf("\n\n");

        for(int i=0;i<N_N653;i++){
            printf("%d,",nttb.coeffs[i]);
        }
        printf("\n\n");
    }*/

   //4 基2
    //int start = (idx & 7) * 168;
    int group_num = idx / 21;
    int group_idx = idx % 21;
    int start = group_num * 168;
    a[0] = ntta.coeffs[start + group_idx];
    a[1] = ntta.coeffs[start + group_idx + 21];
    a[2] = ntta.coeffs[start + group_idx + 21 * 2];
    a[3] = ntta.coeffs[start + group_idx + 21 * 3];
    a[4] = ntta.coeffs[start + group_idx + 21 * 4];
    a[5] = ntta.coeffs[start + group_idx + 21 * 5];
    a[6] = ntta.coeffs[start + group_idx + 21 * 6];
    a[7] = ntta.coeffs[start + group_idx + 21 * 7];

    b[0] = nttb.coeffs[start + group_idx];
    b[1] = nttb.coeffs[start + group_idx + 21];
    b[2] = nttb.coeffs[start + group_idx + 21 * 2];
    b[3] = nttb.coeffs[start + group_idx + 21 * 3];
    b[4] = nttb.coeffs[start + group_idx + 21 * 4];
    b[5] = nttb.coeffs[start + group_idx + 21 * 5];
    b[6] = nttb.coeffs[start + group_idx + 21 * 6];
    b[7] = nttb.coeffs[start + group_idx + 21 * 7];

    //printf("%d,%d\n",idx,start);
    zeta = zetas_n653_v2[8 + group_num];

    //if(idx == 0){ printf("%d,(%d,%d)\n",zeta,a[0],a[4]);}
    CT_a(0,4,zeta);
    CT_a(1,5,zeta);
    CT_a(2,6,zeta);
    CT_a(3,7,zeta);

    CT_b(0,4,zeta);
    CT_b(1,5,zeta);
    CT_b(2,6,zeta);
    CT_b(3,7,zeta);
    //if(idx == 0){ printf("%d,(%d,%d)\n",zeta,a[0],a[4]);}
    //goto res;

    //5 基2
    zeta = zetas_n653_v2[16 + group_num * 2];
    CT_a(0,2,zeta);
    CT_a(1,3,zeta);

    CT_b(0,2,zeta);
    CT_b(1,3,zeta);


    zeta = zetas_n653_v2[16 + group_num * 2 + 1];
    CT_a(4,6,zeta);
    CT_a(5,7,zeta);

    CT_b(4,6,zeta);
    CT_b(5,7,zeta);

    

    //6 基3
    zeta = zetas_n653_v2[32 + group_num * 4];
    CT_a(0,1,zeta);
    CT_b(0,1,zeta);

    zeta = zetas_n653_v2[32 + group_num * 4 + 1];
    CT_a(2,3,zeta);
    CT_b(2,3,zeta);

    zeta = zetas_n653_v2[32 + group_num * 4 + 2];
    CT_a(4,5,zeta);
    CT_b(4,5,zeta);

    zeta = zetas_n653_v2[32 + group_num * 4 + 3];
    CT_a(6,7,zeta);
    CT_b(6,7,zeta);


    ntta.coeffs[start + group_idx] = a[0];
    ntta.coeffs[start + group_idx + 21] = a[1];
    ntta.coeffs[start + group_idx + 21 * 2] = a[2];
    ntta.coeffs[start + group_idx + 21 * 3] = a[3];
    ntta.coeffs[start + group_idx + 21 * 4] = a[4];
    ntta.coeffs[start + group_idx + 21 * 5] = a[5];
    ntta.coeffs[start + group_idx + 21 * 6] = a[6];
    ntta.coeffs[start + group_idx + 21 * 7] = a[7];

    nttb.coeffs[start + group_idx] = b[0];
    nttb.coeffs[start + group_idx + 21] = b[1];
    nttb.coeffs[start + group_idx + 21 * 2] = b[2];
    nttb.coeffs[start + group_idx + 21 * 3] = b[3];
    nttb.coeffs[start + group_idx + 21 * 4] = b[4];
    nttb.coeffs[start + group_idx + 21 * 5] = b[5];
    nttb.coeffs[start + group_idx + 21 * 6] = b[6];
    nttb.coeffs[start + group_idx + 21 * 7] = b[7];


    __syncthreads();

    /*if(idx == 0){
        for(int i=0;i<N_N653;i++){
            printf("%d,",ntta.coeffs[i]);
        }
        printf("\n\n");

        for(int i=0;i<N_N653;i++){
            printf("%d,",nttb.coeffs[i]);
        }
        printf("\n\n");
    }*/

   //基础 3，共有64组,先做24组，再做24组，再做16组
   //3.1
   group_num = idx / 7;
   group_idx = idx % 7;
   start = group_num * 21;
   
   int32_t zeta1 = zetas_n653_v2[64 + group_num * 2];
   int32_t zeta2 = zetas_n653_v2[64 + group_num * 2 + 1];

   int32_t tb, tc, tpho;

    tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * ntta.coeffs[start + group_idx + 7]);
    tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * ntta.coeffs[start + group_idx + 14]);
    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

    ntta.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tb - tpho);
    ntta.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tc + tpho);
    ntta.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] + tb + tc);

    tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * nttb.coeffs[start + group_idx + 7]);
    tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * nttb.coeffs[start + group_idx + 14]);
    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

    nttb.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tb - tpho);
    nttb.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tc + tpho);
    nttb.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] + tb + tc);

    __syncthreads();
    //goto res;
    //TODO:为什么这里会存在溢出呢？
    //3.2
    group_num = idx / 7 + 24;
    group_idx = idx % 7;
    start = group_num * 21;

    zeta1 = zetas_n653_v2[64 + group_num * 2];
    zeta2 = zetas_n653_v2[64 + group_num * 2 + 1];

    tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * ntta.coeffs[start + group_idx + 7]);
    tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * ntta.coeffs[start + group_idx + 14]);
    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

    ntta.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tb - tpho);
    ntta.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tc + tpho);
    ntta.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] + tb + tc);


    tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * nttb.coeffs[start + group_idx + 7]);
    tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * nttb.coeffs[start + group_idx + 14]);
    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

    nttb.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tb - tpho);
    nttb.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tc + tpho);
    nttb.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] + tb + tc);

    __syncthreads();
    

    //3.3
    group_num = idx / 7 + 48;
    group_idx = idx % 7;
    start = group_num * 21;
    if(group_num < 64){
        zeta1 = zetas_n653_v2[64 + group_num * 2];
        zeta2 = zetas_n653_v2[64 + group_num * 2 + 1];

        tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * ntta.coeffs[start + group_idx + 7]);
        tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * ntta.coeffs[start + group_idx + 14]);
        tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

        ntta.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tb - tpho);
        ntta.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tc + tpho);
        ntta.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] + tb + tc);

        tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * nttb.coeffs[start + group_idx + 7]);
        tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * nttb.coeffs[start + group_idx + 14]);
        tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

        nttb.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tb - tpho);
        nttb.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tc + tpho);
        nttb.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] + tb + tc);

    }
    __syncthreads();
/*
res:
    if(idx == 0){
        for(int i=0;i<N_N653;i++){
            printf("%d,",ntta.coeffs[i]);
        }
        printf("\n\n");

        for(int i=0;i<N_N653;i++){
            printf("%d,",nttb.coeffs[i]);
        }
        printf("\n\n");
    }
*/
    //basemul
    group_num = idx / 3;
    group_idx = idx % 3;

    zeta = zetas_n653_v2[N_N653 / 21 + group_num * 2];
    zeta = montgomery_reduce_n653_cuda((int64_t)zeta * root3_n653[group_idx]);

    basemul_n653(nttc.coeffs + 21 * group_num + 7 * group_idx, ntta.coeffs + 21 * group_num + 7 * group_idx, nttb.coeffs + 21 * group_num + 7 * group_idx, zeta);

    group_num = idx / 3 + 56;
    if(group_num < 64){
        zeta = zetas_n653_v2[N_N653 / 21 + group_num * 2];
        zeta = montgomery_reduce_n653_cuda((int64_t)zeta * root3_n653[group_idx]);

        basemul_n653(nttc.coeffs + 21 * group_num + 7 * group_idx, ntta.coeffs + 21 * group_num + 7 * group_idx, nttb.coeffs + 21 * group_num + 7 * group_idx, zeta);
    }

    __syncthreads();
    /*if(idx == 0){
        for(int i=0;i<N_N653;i++){
                printf("%d,",nttc.coeffs[i]);
            }
        printf("\n\n");
    }*/

   //invntt
   // 1 - radix 3
    group_num = idx / 7;
    group_idx = idx % 7;
    start = group_num * 21;

    zeta1 = zetas_inv_n653_v2[group_num * 2];
    zeta2 = zetas_inv_n653_v2[group_num * 2 + 1];

    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (nttc.coeffs[start + group_idx + 7] - nttc.coeffs[start + group_idx + 14]));
    tb = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 7] - tpho);
    tc = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 14] + tpho);

    nttc.coeffs[start + group_idx] = (nttc.coeffs[start + group_idx] + nttc.coeffs[start + group_idx + 7] + nttc.coeffs[start + group_idx + 14]);
    nttc.coeffs[start + group_idx + 7] = montgomery_reduce_n653_cuda((int64_t)zeta1 * tb);
    nttc.coeffs[start + group_idx + 14] = montgomery_reduce_n653_cuda((int64_t)zeta2 * tc);

    //2 - radix 3
    group_num = idx / 7 + 24;
    group_idx = idx % 7;
    start = group_num * 21;

    zeta1 = zetas_inv_n653_v2[group_num * 2];
    zeta2 = zetas_inv_n653_v2[group_num * 2 + 1];

    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (nttc.coeffs[start + group_idx + 7] - nttc.coeffs[start + group_idx + 14]));
    tb = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 7] - tpho);
    tc = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 14] + tpho);

    nttc.coeffs[start + group_idx] = (nttc.coeffs[start + group_idx] + nttc.coeffs[start + group_idx + 7] + nttc.coeffs[start + group_idx + 14]);
    nttc.coeffs[start + group_idx + 7] = montgomery_reduce_n653_cuda((int64_t)zeta1 * tb);
    nttc.coeffs[start + group_idx + 14] = montgomery_reduce_n653_cuda((int64_t)zeta2 * tc);

    //3 - radix 3
    group_num = idx / 7 + 48;
    group_idx = idx % 7;
    start = group_num * 21;
    if(group_num < 64){
        zeta1 = zetas_inv_n653_v2[group_num * 2];
        zeta2 = zetas_inv_n653_v2[group_num * 2 + 1];

        tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (nttc.coeffs[start + group_idx + 7] - nttc.coeffs[start + group_idx + 14]));
        tb = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 7] - tpho);
        tc = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 14] + tpho);

        nttc.coeffs[start + group_idx] = (nttc.coeffs[start + group_idx] + nttc.coeffs[start + group_idx + 7] + nttc.coeffs[start + group_idx + 14]);
        nttc.coeffs[start + group_idx + 7] = montgomery_reduce_n653_cuda((int64_t)zeta1 * tb);
        nttc.coeffs[start + group_idx + 14] = montgomery_reduce_n653_cuda((int64_t)zeta2 * tc);
    }

    __syncthreads();
    
    //3层- radix 2

    //1
    group_num = idx / 21;
    group_idx = idx % 21;
    start = group_num * 168;
    a[0] = nttc.coeffs[start + group_idx];
    a[1] = nttc.coeffs[start + group_idx + 21];
    a[2] = nttc.coeffs[start + group_idx + 21 * 2];
    a[3] = nttc.coeffs[start + group_idx + 21 * 3];
    a[4] = nttc.coeffs[start + group_idx + 21 * 4];
    a[5] = nttc.coeffs[start + group_idx + 21 * 5];
    a[6] = nttc.coeffs[start + group_idx + 21 * 6];
    a[7] = nttc.coeffs[start + group_idx + 21 * 7];

    zeta = zetas_inv_n653_v2[128 + group_num * 4];
    GS_a(0,1,zeta);
    
    zeta = zetas_inv_n653_v2[128 + group_num * 4 + 1];
    GS_a(2,3,zeta);

    zeta = zetas_inv_n653_v2[128 + group_num * 4 + 2];
    GS_a(4,5,zeta);

    zeta = zetas_inv_n653_v2[128 + group_num * 4 + 3];
    GS_a(6,7,zeta);

    //2
    zeta = zetas_inv_n653_v2[160 + group_num * 2];
    GS_a(0,2,zeta);
    GS_a(1,3,zeta);

    zeta = zetas_inv_n653_v2[160 + group_num * 2 + 1];
    GS_a(4,6,zeta);
    GS_a(5,7,zeta);

    //3
    zeta = zetas_inv_n653_v2[176 + group_num];
    GS_a(0,4,zeta);
    GS_a(1,5,zeta);
    GS_a(2,6,zeta);
    GS_a(3,7,zeta);

    nttc.coeffs[start + group_idx] = a[0];
    nttc.coeffs[start + group_idx + 21] = a[1];
    nttc.coeffs[start + group_idx + 21 * 2] = a[2];
    nttc.coeffs[start + group_idx + 21 * 3] = a[3];
    nttc.coeffs[start + group_idx + 21 * 4] = a[4];
    nttc.coeffs[start + group_idx + 21 * 5] = a[5];
    nttc.coeffs[start + group_idx + 21 * 6] = a[6];
    nttc.coeffs[start + group_idx + 21 * 7] = a[7];

    __syncthreads();

    /*if(idx == 0){
        for(int i=0;i<N_N653;i++){
                printf("%d,",nttc.coeffs[i]);
            }
        printf("\n\n");
    }*/

    a[0] = nttc.coeffs[idx];
    a[1] = nttc.coeffs[idx + 168];
    a[2] = nttc.coeffs[idx + 168 * 2];
    a[3] = nttc.coeffs[idx + 168 * 3];
    a[4] = nttc.coeffs[idx + 168 * 4];
    a[5] = nttc.coeffs[idx + 168 * 5];
    a[6] = nttc.coeffs[idx + 168 * 6];
    a[7] = nttc.coeffs[idx + 168 * 7];
    
    //1
    zeta = zetas_inv_n653_v2[184];
    GS_a(0,1,zeta);
    
    zeta = zetas_inv_n653_v2[185];
    GS_a(2,3,zeta);

    zeta = zetas_inv_n653_v2[186];
    GS_a(4,5,zeta);

    zeta = zetas_inv_n653_v2[187];
    GS_a(6,7,zeta);

    //2
    zeta = zetas_inv_n653_v2[188];
    GS_a(0,2,zeta);
    GS_a(1,3,zeta);

    zeta = zetas_inv_n653_v2[189];
    GS_a(4,6,zeta);
    GS_a(5,7,zeta);

    
    //3
    special_CT_a_pse(0,4);
    special_CT_a_pse(1,5);
    special_CT_a_pse(2,6);
    special_CT_a_pse(3,7);

    nttc.coeffs[idx] = a[0];
    nttc.coeffs[idx + 168] = a[1];
    nttc.coeffs[idx + 168 * 2] = a[2];
    nttc.coeffs[idx + 168 * 3] = a[3];
    nttc.coeffs[idx + 168 * 4] = a[4];
    nttc.coeffs[idx + 168 * 5] = a[5];
    nttc.coeffs[idx + 168 * 6] = a[6];
    nttc.coeffs[idx + 168 * 7] = a[7];

    __syncthreads();
    /*if(idx == 0){
        for(int i=0;i<N_N653;i++){
                printf("%d,",nttc.coeffs[i]);
            }
        printf("\n\n");
    }*/

    //只需要FPTRU_N个线程来完成 除去0和652
    array_c[blockIdx.x].coeffs[1 + idx * 3 ] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[1 + idx * 3 ] + nttc.coeffs[1 + idx * 3  + FPTRU_N - 1] + nttc.coeffs[1 + idx * 3  + FPTRU_N])));

    array_c[blockIdx.x].coeffs[2 + idx * 3  ] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[2 + idx * 3  ] + nttc.coeffs[2 + idx * 3   + FPTRU_N - 1] + nttc.coeffs[2 + idx * 3   + FPTRU_N])));

    array_c[blockIdx.x].coeffs[3 + idx * 3  ] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[3 + idx * 3  ] + nttc.coeffs[3 + idx * 3   + FPTRU_N - 1] + nttc.coeffs[3 + idx * 3   + FPTRU_N])));


    if(idx < 148){
        if(idx == 0){
            array_c[blockIdx.x].coeffs[0] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[0] + nttc.coeffs[FPTRU_N])));
            array_c[blockIdx.x].coeffs[FPTRU_N - 1] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[FPTRU_N - 1] + nttc.coeffs[2 * FPTRU_N - 2])));
        }
        else{
            array_c[blockIdx.x].coeffs[idx + 504] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[idx + 504] + nttc.coeffs[idx + 504 + FPTRU_N - 1] + nttc.coeffs[idx + 504 + FPTRU_N])));
        }
    }


}

/*2024-6-23:
设备端函数，处理一次多项式的乘法*/
__device__ void poly_mul_653_batch_q1_v3_device(poly *array_c,poly * array_a,poly *array_b){
    __shared__ nttpoly_n653 ntta;
    __shared__ nttpoly_n653 nttb;
    __shared__ nttpoly_n653 nttc;

    int32_t a[8] ={0};
    int32_t b[8] ={0};//注意数据类型和nttpoly_n653保持一致
    int idx = threadIdx.x;
    int32_t zeta;
    int32_t t;

    poly pa;
    poly pb;

    /*int c21 = idx / 21;
    int m21 = idx % 21;
    
    int c7 = idx / 7;
    int m7 = idx - c7 * 7;*/
    pa = array_a[blockIdx.x];
    pb = array_b[blockIdx.x];

    //不做extend，直接读8个系数
    a[0] = pa.coeffs[idx];//array_a[blockIdx.x].coeffs[idx];
    a[1] = pa.coeffs[idx + 168];//array_a[blockIdx.x].coeffs[idx + 168];
    a[2] = pa.coeffs[idx + 168 * 2];//array_a[blockIdx.x].coeffs[idx + 168 * 2];
    
    if(idx + 168 * 3 < FPTRU_N){
        a[3] = pa.coeffs[idx + 168 * 3];//array_a[blockIdx.x].coeffs[idx + 168 * 3];
    }
    
    //不做extend，直接读8个系数
    b[0] = pb.coeffs[idx];//array_b[blockIdx.x].coeffs[idx];
    b[1] = pb.coeffs[idx + 168];//array_b[blockIdx.x].coeffs[idx + 168];
    b[2] = pb.coeffs[idx + 168 * 2];//array_b[blockIdx.x].coeffs[idx + 168 * 2];
    
    if(idx + 168 * 3 < FPTRU_N){
        b[3] = pb.coeffs[idx + 168 * 3];//array_b[blockIdx.x].coeffs[idx + 168 * 3];
    }
    
    //一层基2
    //zeta = zetas_n653[1]; //旋转因子为-1
    special_CT_a(0,4);
    special_CT_a(1,5);
    special_CT_a(2,6);
    special_CT_a(3,7);

    special_CT_b(0,4);
    special_CT_b(1,5);
    special_CT_b(2,6);
    special_CT_b(3,7);

    

    //2 基2
    zeta = zetas_n653_v2[2];

    CT_a(0,2,zeta);
    CT_a(1,3,zeta);

    CT_b(0,2,zeta);
    CT_b(1,3,zeta);

    //zeta = zetas_n653[3]; //旋转因子为-1
    special_CT_a(4,6);
    special_CT_a(5,7);

    special_CT_b(4,6);
    special_CT_b(5,7);


    //3 基2
    zeta = zetas_n653_v2[4];
    CT_a(0,1,zeta);
    CT_b(0,1,zeta);

    zeta = zetas_n653_v2[5];
    CT_a(2,3,zeta);
    CT_b(2,3,zeta);

    zeta = zetas_n653_v2[6];
    CT_a(4,5,zeta);
    CT_b(4,5,zeta);

    special_CT_a(6,7);
    special_CT_b(6,7);

    //放回共享内存
    ntta.coeffs[idx] = a[0];
    ntta.coeffs[idx + 168] = a[1];
    ntta.coeffs[idx + 168 * 2] = a[2];
    ntta.coeffs[idx + 168 * 3] = a[3];
    ntta.coeffs[idx + 168 * 4] = a[4];
    ntta.coeffs[idx + 168 * 5] = a[5];
    ntta.coeffs[idx + 168 * 6] = a[6];
    ntta.coeffs[idx + 168 * 7] = a[7];

    nttb.coeffs[idx] = b[0];
    nttb.coeffs[idx + 168] = b[1];
    nttb.coeffs[idx + 168 * 2] = b[2];
    nttb.coeffs[idx + 168 * 3] = b[3];
    nttb.coeffs[idx + 168 * 4] = b[4];
    nttb.coeffs[idx + 168 * 5] = b[5];
    nttb.coeffs[idx + 168 * 6] = b[6];
    nttb.coeffs[idx + 168 * 7] = b[7];

    __syncthreads();

    /*if(idx == 0){
        for(int i=0;i<N_N653;i++){
            printf("%d,",ntta.coeffs[i]);
        }
        printf("\n\n");

        for(int i=0;i<N_N653;i++){
            printf("%d,",nttb.coeffs[i]);
        }
        printf("\n\n");
    }*/

   //4 基2
    //int start = (idx & 7) * 168;
    int group_num = idx / 21;
    int group_idx = idx % 21;
    int start = group_num * 168;
    a[0] = ntta.coeffs[start + group_idx];
    a[1] = ntta.coeffs[start + group_idx + 21];
    a[2] = ntta.coeffs[start + group_idx + 21 * 2];
    a[3] = ntta.coeffs[start + group_idx + 21 * 3];
    a[4] = ntta.coeffs[start + group_idx + 21 * 4];
    a[5] = ntta.coeffs[start + group_idx + 21 * 5];
    a[6] = ntta.coeffs[start + group_idx + 21 * 6];
    a[7] = ntta.coeffs[start + group_idx + 21 * 7];

    b[0] = nttb.coeffs[start + group_idx];
    b[1] = nttb.coeffs[start + group_idx + 21];
    b[2] = nttb.coeffs[start + group_idx + 21 * 2];
    b[3] = nttb.coeffs[start + group_idx + 21 * 3];
    b[4] = nttb.coeffs[start + group_idx + 21 * 4];
    b[5] = nttb.coeffs[start + group_idx + 21 * 5];
    b[6] = nttb.coeffs[start + group_idx + 21 * 6];
    b[7] = nttb.coeffs[start + group_idx + 21 * 7];

    //printf("%d,%d\n",idx,start);
    zeta = zetas_n653_v2[8 + group_num];

    //if(idx == 0){ printf("%d,(%d,%d)\n",zeta,a[0],a[4]);}
    CT_a(0,4,zeta);
    CT_a(1,5,zeta);
    CT_a(2,6,zeta);
    CT_a(3,7,zeta);

    CT_b(0,4,zeta);
    CT_b(1,5,zeta);
    CT_b(2,6,zeta);
    CT_b(3,7,zeta);
    //if(idx == 0){ printf("%d,(%d,%d)\n",zeta,a[0],a[4]);}
    //goto res;

    //5 基2
    zeta = zetas_n653_v2[16 + group_num * 2];
    CT_a(0,2,zeta);
    CT_a(1,3,zeta);

    CT_b(0,2,zeta);
    CT_b(1,3,zeta);


    zeta = zetas_n653_v2[16 + group_num * 2 + 1];
    CT_a(4,6,zeta);
    CT_a(5,7,zeta);

    CT_b(4,6,zeta);
    CT_b(5,7,zeta);

    

    //6 基3
    zeta = zetas_n653_v2[32 + group_num * 4];
    CT_a(0,1,zeta);
    CT_b(0,1,zeta);

    zeta = zetas_n653_v2[32 + group_num * 4 + 1];
    CT_a(2,3,zeta);
    CT_b(2,3,zeta);

    zeta = zetas_n653_v2[32 + group_num * 4 + 2];
    CT_a(4,5,zeta);
    CT_b(4,5,zeta);

    zeta = zetas_n653_v2[32 + group_num * 4 + 3];
    CT_a(6,7,zeta);
    CT_b(6,7,zeta);


    ntta.coeffs[start + group_idx] = a[0];
    ntta.coeffs[start + group_idx + 21] = a[1];
    ntta.coeffs[start + group_idx + 21 * 2] = a[2];
    ntta.coeffs[start + group_idx + 21 * 3] = a[3];
    ntta.coeffs[start + group_idx + 21 * 4] = a[4];
    ntta.coeffs[start + group_idx + 21 * 5] = a[5];
    ntta.coeffs[start + group_idx + 21 * 6] = a[6];
    ntta.coeffs[start + group_idx + 21 * 7] = a[7];

    nttb.coeffs[start + group_idx] = b[0];
    nttb.coeffs[start + group_idx + 21] = b[1];
    nttb.coeffs[start + group_idx + 21 * 2] = b[2];
    nttb.coeffs[start + group_idx + 21 * 3] = b[3];
    nttb.coeffs[start + group_idx + 21 * 4] = b[4];
    nttb.coeffs[start + group_idx + 21 * 5] = b[5];
    nttb.coeffs[start + group_idx + 21 * 6] = b[6];
    nttb.coeffs[start + group_idx + 21 * 7] = b[7];


    __syncthreads();

    /*if(idx == 0){
        for(int i=0;i<N_N653;i++){
            printf("%d,",ntta.coeffs[i]);
        }
        printf("\n\n");

        for(int i=0;i<N_N653;i++){
            printf("%d,",nttb.coeffs[i]);
        }
        printf("\n\n");
    }*/

   //基础 3，共有64组,先做24组，再做24组，再做16组
   //3.1
   group_num = idx / 7;
   group_idx = idx % 7;
   start = group_num * 21;
   
   int32_t zeta1 = zetas_n653_v2[64 + group_num * 2];
   int32_t zeta2 = zetas_n653_v2[64 + group_num * 2 + 1];

   int32_t tb, tc, tpho;

    tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * ntta.coeffs[start + group_idx + 7]);
    tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * ntta.coeffs[start + group_idx + 14]);
    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

    ntta.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tb - tpho);
    ntta.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tc + tpho);
    ntta.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] + tb + tc);

    tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * nttb.coeffs[start + group_idx + 7]);
    tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * nttb.coeffs[start + group_idx + 14]);
    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

    nttb.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tb - tpho);
    nttb.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tc + tpho);
    nttb.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] + tb + tc);

    __syncthreads();
    //goto res;
    //TODO:为什么这里会存在溢出呢？
    //3.2
    group_num = idx / 7 + 24;
    group_idx = idx % 7;
    start = group_num * 21;

    zeta1 = zetas_n653_v2[64 + group_num * 2];
    zeta2 = zetas_n653_v2[64 + group_num * 2 + 1];

    tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * ntta.coeffs[start + group_idx + 7]);
    tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * ntta.coeffs[start + group_idx + 14]);
    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

    ntta.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tb - tpho);
    ntta.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tc + tpho);
    ntta.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] + tb + tc);


    tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * nttb.coeffs[start + group_idx + 7]);
    tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * nttb.coeffs[start + group_idx + 14]);
    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

    nttb.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tb - tpho);
    nttb.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tc + tpho);
    nttb.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] + tb + tc);

    __syncthreads();
    

    //3.3
    group_num = idx / 7 + 48;
    group_idx = idx % 7;
    start = group_num * 21;
    if(group_num < 64){
        zeta1 = zetas_n653_v2[64 + group_num * 2];
        zeta2 = zetas_n653_v2[64 + group_num * 2 + 1];

        tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * ntta.coeffs[start + group_idx + 7]);
        tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * ntta.coeffs[start + group_idx + 14]);
        tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

        ntta.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tb - tpho);
        ntta.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tc + tpho);
        ntta.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] + tb + tc);

        tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * nttb.coeffs[start + group_idx + 7]);
        tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * nttb.coeffs[start + group_idx + 14]);
        tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

        nttb.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tb - tpho);
        nttb.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tc + tpho);
        nttb.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] + tb + tc);

    }
    __syncthreads();
/*
res:
    if(idx == 0){
        for(int i=0;i<N_N653;i++){
            printf("%d,",ntta.coeffs[i]);
        }
        printf("\n\n");

        for(int i=0;i<N_N653;i++){
            printf("%d,",nttb.coeffs[i]);
        }
        printf("\n\n");
    }
*/
    //basemul
    group_num = idx / 3;
    group_idx = idx % 3;

    zeta = zetas_n653_v2[N_N653 / 21 + group_num * 2];
    zeta = montgomery_reduce_n653_cuda((int64_t)zeta * root3_n653[group_idx]);

    basemul_n653(nttc.coeffs + 21 * group_num + 7 * group_idx, ntta.coeffs + 21 * group_num + 7 * group_idx, nttb.coeffs + 21 * group_num + 7 * group_idx, zeta);

    group_num = idx / 3 + 56;
    if(group_num < 64){
        zeta = zetas_n653_v2[N_N653 / 21 + group_num * 2];
        zeta = montgomery_reduce_n653_cuda((int64_t)zeta * root3_n653[group_idx]);

        basemul_n653(nttc.coeffs + 21 * group_num + 7 * group_idx, ntta.coeffs + 21 * group_num + 7 * group_idx, nttb.coeffs + 21 * group_num + 7 * group_idx, zeta);
    }

    __syncthreads();
    /*if(idx == 0){
        for(int i=0;i<N_N653;i++){
                printf("%d,",nttc.coeffs[i]);
            }
        printf("\n\n");
    }*/

   //invntt
   // 1 - radix 3
    group_num = idx / 7;
    group_idx = idx % 7;
    start = group_num * 21;

    zeta1 = zetas_inv_n653_v2[group_num * 2];
    zeta2 = zetas_inv_n653_v2[group_num * 2 + 1];

    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (nttc.coeffs[start + group_idx + 7] - nttc.coeffs[start + group_idx + 14]));
    tb = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 7] - tpho);
    tc = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 14] + tpho);

    nttc.coeffs[start + group_idx] = (nttc.coeffs[start + group_idx] + nttc.coeffs[start + group_idx + 7] + nttc.coeffs[start + group_idx + 14]);
    nttc.coeffs[start + group_idx + 7] = montgomery_reduce_n653_cuda((int64_t)zeta1 * tb);
    nttc.coeffs[start + group_idx + 14] = montgomery_reduce_n653_cuda((int64_t)zeta2 * tc);

    //2 - radix 3
    group_num = idx / 7 + 24;
    group_idx = idx % 7;
    start = group_num * 21;

    zeta1 = zetas_inv_n653_v2[group_num * 2];
    zeta2 = zetas_inv_n653_v2[group_num * 2 + 1];

    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (nttc.coeffs[start + group_idx + 7] - nttc.coeffs[start + group_idx + 14]));
    tb = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 7] - tpho);
    tc = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 14] + tpho);

    nttc.coeffs[start + group_idx] = (nttc.coeffs[start + group_idx] + nttc.coeffs[start + group_idx + 7] + nttc.coeffs[start + group_idx + 14]);
    nttc.coeffs[start + group_idx + 7] = montgomery_reduce_n653_cuda((int64_t)zeta1 * tb);
    nttc.coeffs[start + group_idx + 14] = montgomery_reduce_n653_cuda((int64_t)zeta2 * tc);

    //3 - radix 3
    group_num = idx / 7 + 48;
    group_idx = idx % 7;
    start = group_num * 21;
    if(group_num < 64){
        zeta1 = zetas_inv_n653_v2[group_num * 2];
        zeta2 = zetas_inv_n653_v2[group_num * 2 + 1];

        tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (nttc.coeffs[start + group_idx + 7] - nttc.coeffs[start + group_idx + 14]));
        tb = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 7] - tpho);
        tc = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 14] + tpho);

        nttc.coeffs[start + group_idx] = (nttc.coeffs[start + group_idx] + nttc.coeffs[start + group_idx + 7] + nttc.coeffs[start + group_idx + 14]);
        nttc.coeffs[start + group_idx + 7] = montgomery_reduce_n653_cuda((int64_t)zeta1 * tb);
        nttc.coeffs[start + group_idx + 14] = montgomery_reduce_n653_cuda((int64_t)zeta2 * tc);
    }

    __syncthreads();
    
    //3层- radix 2

    //1
    group_num = idx / 21;
    group_idx = idx % 21;
    start = group_num * 168;
    a[0] = nttc.coeffs[start + group_idx];
    a[1] = nttc.coeffs[start + group_idx + 21];
    a[2] = nttc.coeffs[start + group_idx + 21 * 2];
    a[3] = nttc.coeffs[start + group_idx + 21 * 3];
    a[4] = nttc.coeffs[start + group_idx + 21 * 4];
    a[5] = nttc.coeffs[start + group_idx + 21 * 5];
    a[6] = nttc.coeffs[start + group_idx + 21 * 6];
    a[7] = nttc.coeffs[start + group_idx + 21 * 7];

    zeta = zetas_inv_n653_v2[128 + group_num * 4];
    GS_a(0,1,zeta);
    
    zeta = zetas_inv_n653_v2[128 + group_num * 4 + 1];
    GS_a(2,3,zeta);

    zeta = zetas_inv_n653_v2[128 + group_num * 4 + 2];
    GS_a(4,5,zeta);

    zeta = zetas_inv_n653_v2[128 + group_num * 4 + 3];
    GS_a(6,7,zeta);

    //2
    zeta = zetas_inv_n653_v2[160 + group_num * 2];
    GS_a(0,2,zeta);
    GS_a(1,3,zeta);

    zeta = zetas_inv_n653_v2[160 + group_num * 2 + 1];
    GS_a(4,6,zeta);
    GS_a(5,7,zeta);

    //3
    zeta = zetas_inv_n653_v2[176 + group_num];
    GS_a(0,4,zeta);
    GS_a(1,5,zeta);
    GS_a(2,6,zeta);
    GS_a(3,7,zeta);

    nttc.coeffs[start + group_idx] = a[0];
    nttc.coeffs[start + group_idx + 21] = a[1];
    nttc.coeffs[start + group_idx + 21 * 2] = a[2];
    nttc.coeffs[start + group_idx + 21 * 3] = a[3];
    nttc.coeffs[start + group_idx + 21 * 4] = a[4];
    nttc.coeffs[start + group_idx + 21 * 5] = a[5];
    nttc.coeffs[start + group_idx + 21 * 6] = a[6];
    nttc.coeffs[start + group_idx + 21 * 7] = a[7];

    __syncthreads();

    /*if(idx == 0){
        for(int i=0;i<N_N653;i++){
                printf("%d,",nttc.coeffs[i]);
            }
        printf("\n\n");
    }*/

    a[0] = nttc.coeffs[idx];
    a[1] = nttc.coeffs[idx + 168];
    a[2] = nttc.coeffs[idx + 168 * 2];
    a[3] = nttc.coeffs[idx + 168 * 3];
    a[4] = nttc.coeffs[idx + 168 * 4];
    a[5] = nttc.coeffs[idx + 168 * 5];
    a[6] = nttc.coeffs[idx + 168 * 6];
    a[7] = nttc.coeffs[idx + 168 * 7];
    
    //1
    zeta = zetas_inv_n653_v2[184];
    GS_a(0,1,zeta);
    
    zeta = zetas_inv_n653_v2[185];
    GS_a(2,3,zeta);

    zeta = zetas_inv_n653_v2[186];
    GS_a(4,5,zeta);

    zeta = zetas_inv_n653_v2[187];
    GS_a(6,7,zeta);

    //2
    zeta = zetas_inv_n653_v2[188];
    GS_a(0,2,zeta);
    GS_a(1,3,zeta);

    zeta = zetas_inv_n653_v2[189];
    GS_a(4,6,zeta);
    GS_a(5,7,zeta);

    
    //3
    special_CT_a_pse(0,4);
    special_CT_a_pse(1,5);
    special_CT_a_pse(2,6);
    special_CT_a_pse(3,7);

    nttc.coeffs[idx] = a[0];
    nttc.coeffs[idx + 168] = a[1];
    nttc.coeffs[idx + 168 * 2] = a[2];
    nttc.coeffs[idx + 168 * 3] = a[3];
    nttc.coeffs[idx + 168 * 4] = a[4];
    nttc.coeffs[idx + 168 * 5] = a[5];
    nttc.coeffs[idx + 168 * 6] = a[6];
    nttc.coeffs[idx + 168 * 7] = a[7];

    __syncthreads();
    /*if(idx == 0){
        for(int i=0;i<N_N653;i++){
                printf("%d,",nttc.coeffs[i]);
            }
        printf("\n\n");
    }*/

    //只需要FPTRU_N个线程来完成 除去0和652
    array_c[blockIdx.x].coeffs[1 + idx * 3 ] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[1 + idx * 3 ] + nttc.coeffs[1 + idx * 3  + FPTRU_N - 1] + nttc.coeffs[1 + idx * 3  + FPTRU_N])));

    array_c[blockIdx.x].coeffs[2 + idx * 3  ] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[2 + idx * 3  ] + nttc.coeffs[2 + idx * 3   + FPTRU_N - 1] + nttc.coeffs[2 + idx * 3   + FPTRU_N])));

    array_c[blockIdx.x].coeffs[3 + idx * 3  ] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[3 + idx * 3  ] + nttc.coeffs[3 + idx * 3   + FPTRU_N - 1] + nttc.coeffs[3 + idx * 3   + FPTRU_N])));


    if(idx < 148){
        if(idx == 0){
            array_c[blockIdx.x].coeffs[0] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[0] + nttc.coeffs[FPTRU_N])));
            array_c[blockIdx.x].coeffs[FPTRU_N - 1] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[FPTRU_N - 1] + nttc.coeffs[2 * FPTRU_N - 2])));
        }
        else{
            array_c[blockIdx.x].coeffs[idx + 504] = fq_freeze(montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[idx + 504] + nttc.coeffs[idx + 504 + FPTRU_N - 1] + nttc.coeffs[idx + 504 + FPTRU_N])));
        }
    }


}

/*__device__ void CT_ptx(int32_t* px,int32_t* py,int32_t zeta){
    int32_t x = *px;
    int32_t y = *py;

    asm(
        "{\n\t"
        ""

        }
    );

    *px = x;
    *py = y;
}*/



__global__ void poly_mul_653_batch_q2(poly *array_c,poly * array_a,poly *array_b){
    __shared__ nttpoly_n653 ntta;
    __shared__ nttpoly_n653 nttb;
    __shared__ nttpoly_n653 nttc;

    int32_t a[8] ={0};
    int32_t b[8] ={0};//注意数据类型和nttpoly_n653保持一致
    int idx = threadIdx.x;
    int32_t zeta;
    int32_t t;

    poly pa;
    poly pb;

    /*int c21 = idx / 21;
    int m21 = idx % 21;
    
    int c7 = idx / 7;
    int m7 = idx - c7 * 7;*/
    pa = array_a[blockIdx.x];
    pb = array_b[blockIdx.x];

    //不做extend，直接读8个系数
    a[0] = pa.coeffs[idx];//array_a[blockIdx.x].coeffs[idx];
    a[1] = pa.coeffs[idx + 168];//array_a[blockIdx.x].coeffs[idx + 168];
    a[2] = pa.coeffs[idx + 168 * 2];//array_a[blockIdx.x].coeffs[idx + 168 * 2];
    
    if(idx + 168 * 3 < FPTRU_N){
        a[3] = pa.coeffs[idx + 168 * 3];//array_a[blockIdx.x].coeffs[idx + 168 * 3];
    }
    
    //不做extend，直接读8个系数
    b[0] = pb.coeffs[idx];//array_b[blockIdx.x].coeffs[idx];
    b[1] = pb.coeffs[idx + 168];//array_b[blockIdx.x].coeffs[idx + 168];
    b[2] = pb.coeffs[idx + 168 * 2];//array_b[blockIdx.x].coeffs[idx + 168 * 2];
    
    if(idx + 168 * 3 < FPTRU_N){
        b[3] = pb.coeffs[idx + 168 * 3];//array_b[blockIdx.x].coeffs[idx + 168 * 3];
    }
    
    //一层基2
    //zeta = zetas_n653[1]; //旋转因子为-1
    special_CT_a(0,4);
    special_CT_a(1,5);
    special_CT_a(2,6);
    special_CT_a(3,7);

    special_CT_b(0,4);
    special_CT_b(1,5);
    special_CT_b(2,6);
    special_CT_b(3,7);

    

    //2 基2
    zeta = zetas_n653_v2[2];

    CT_a(0,2,zeta);
    CT_a(1,3,zeta);

    CT_b(0,2,zeta);
    CT_b(1,3,zeta);

    //zeta = zetas_n653[3]; //旋转因子为-1
    special_CT_a(4,6);
    special_CT_a(5,7);

    special_CT_b(4,6);
    special_CT_b(5,7);


    //3 基2
    zeta = zetas_n653_v2[4];
    CT_a(0,1,zeta);
    CT_b(0,1,zeta);

    zeta = zetas_n653_v2[5];
    CT_a(2,3,zeta);
    CT_b(2,3,zeta);

    zeta = zetas_n653_v2[6];
    CT_a(4,5,zeta);
    CT_b(4,5,zeta);

    special_CT_a(6,7);
    special_CT_b(6,7);

    //放回共享内存
    ntta.coeffs[idx] = a[0];
    ntta.coeffs[idx + 168] = a[1];
    ntta.coeffs[idx + 168 * 2] = a[2];
    ntta.coeffs[idx + 168 * 3] = a[3];
    ntta.coeffs[idx + 168 * 4] = a[4];
    ntta.coeffs[idx + 168 * 5] = a[5];
    ntta.coeffs[idx + 168 * 6] = a[6];
    ntta.coeffs[idx + 168 * 7] = a[7];

    nttb.coeffs[idx] = b[0];
    nttb.coeffs[idx + 168] = b[1];
    nttb.coeffs[idx + 168 * 2] = b[2];
    nttb.coeffs[idx + 168 * 3] = b[3];
    nttb.coeffs[idx + 168 * 4] = b[4];
    nttb.coeffs[idx + 168 * 5] = b[5];
    nttb.coeffs[idx + 168 * 6] = b[6];
    nttb.coeffs[idx + 168 * 7] = b[7];

    __syncthreads();

    /*if(idx == 0){
        for(int i=0;i<N_N653;i++){
            printf("%d,",ntta.coeffs[i]);
        }
        printf("\n\n");

        for(int i=0;i<N_N653;i++){
            printf("%d,",nttb.coeffs[i]);
        }
        printf("\n\n");
    }*/

   //4 基2
    //int start = (idx & 7) * 168;
    int group_num = idx / 21;
    int group_idx = idx % 21;
    int start = group_num * 168;
    a[0] = ntta.coeffs[start + group_idx];
    a[1] = ntta.coeffs[start + group_idx + 21];
    a[2] = ntta.coeffs[start + group_idx + 21 * 2];
    a[3] = ntta.coeffs[start + group_idx + 21 * 3];
    a[4] = ntta.coeffs[start + group_idx + 21 * 4];
    a[5] = ntta.coeffs[start + group_idx + 21 * 5];
    a[6] = ntta.coeffs[start + group_idx + 21 * 6];
    a[7] = ntta.coeffs[start + group_idx + 21 * 7];

    b[0] = nttb.coeffs[start + group_idx];
    b[1] = nttb.coeffs[start + group_idx + 21];
    b[2] = nttb.coeffs[start + group_idx + 21 * 2];
    b[3] = nttb.coeffs[start + group_idx + 21 * 3];
    b[4] = nttb.coeffs[start + group_idx + 21 * 4];
    b[5] = nttb.coeffs[start + group_idx + 21 * 5];
    b[6] = nttb.coeffs[start + group_idx + 21 * 6];
    b[7] = nttb.coeffs[start + group_idx + 21 * 7];

    //printf("%d,%d\n",idx,start);
    zeta = zetas_n653_v2[8 + group_num];

    //if(idx == 0){ printf("%d,(%d,%d)\n",zeta,a[0],a[4]);}
    CT_a(0,4,zeta);
    CT_a(1,5,zeta);
    CT_a(2,6,zeta);
    CT_a(3,7,zeta);

    CT_b(0,4,zeta);
    CT_b(1,5,zeta);
    CT_b(2,6,zeta);
    CT_b(3,7,zeta);
    //if(idx == 0){ printf("%d,(%d,%d)\n",zeta,a[0],a[4]);}
    //goto res;

    //5 基2
    zeta = zetas_n653_v2[16 + group_num * 2];
    CT_a(0,2,zeta);
    CT_a(1,3,zeta);

    CT_b(0,2,zeta);
    CT_b(1,3,zeta);


    zeta = zetas_n653_v2[16 + group_num * 2 + 1];
    CT_a(4,6,zeta);
    CT_a(5,7,zeta);

    CT_b(4,6,zeta);
    CT_b(5,7,zeta);

    

    //6 基3
    zeta = zetas_n653_v2[32 + group_num * 4];
    CT_a(0,1,zeta);
    CT_b(0,1,zeta);

    zeta = zetas_n653_v2[32 + group_num * 4 + 1];
    CT_a(2,3,zeta);
    CT_b(2,3,zeta);

    zeta = zetas_n653_v2[32 + group_num * 4 + 2];
    CT_a(4,5,zeta);
    CT_b(4,5,zeta);

    zeta = zetas_n653_v2[32 + group_num * 4 + 3];
    CT_a(6,7,zeta);
    CT_b(6,7,zeta);


    ntta.coeffs[start + group_idx] = a[0];
    ntta.coeffs[start + group_idx + 21] = a[1];
    ntta.coeffs[start + group_idx + 21 * 2] = a[2];
    ntta.coeffs[start + group_idx + 21 * 3] = a[3];
    ntta.coeffs[start + group_idx + 21 * 4] = a[4];
    ntta.coeffs[start + group_idx + 21 * 5] = a[5];
    ntta.coeffs[start + group_idx + 21 * 6] = a[6];
    ntta.coeffs[start + group_idx + 21 * 7] = a[7];

    nttb.coeffs[start + group_idx] = b[0];
    nttb.coeffs[start + group_idx + 21] = b[1];
    nttb.coeffs[start + group_idx + 21 * 2] = b[2];
    nttb.coeffs[start + group_idx + 21 * 3] = b[3];
    nttb.coeffs[start + group_idx + 21 * 4] = b[4];
    nttb.coeffs[start + group_idx + 21 * 5] = b[5];
    nttb.coeffs[start + group_idx + 21 * 6] = b[6];
    nttb.coeffs[start + group_idx + 21 * 7] = b[7];


    __syncthreads();

    /*if(idx == 0){
        for(int i=0;i<N_N653;i++){
            printf("%d,",ntta.coeffs[i]);
        }
        printf("\n\n");

        for(int i=0;i<N_N653;i++){
            printf("%d,",nttb.coeffs[i]);
        }
        printf("\n\n");
    }*/

   //基础 3，共有64组,先做24组，再做24组，再做16组
   //3.1
   group_num = idx / 7;
   group_idx = idx % 7;
   start = group_num * 21;
   
   int32_t zeta1 = zetas_n653_v2[64 + group_num * 2];
   int32_t zeta2 = zetas_n653_v2[64 + group_num * 2 + 1];

   int32_t tb, tc, tpho;

    tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * ntta.coeffs[start + group_idx + 7]);
    tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * ntta.coeffs[start + group_idx + 14]);
    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

    ntta.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tb - tpho);
    ntta.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tc + tpho);
    ntta.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] + tb + tc);

    tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * nttb.coeffs[start + group_idx + 7]);
    tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * nttb.coeffs[start + group_idx + 14]);
    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

    nttb.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tb - tpho);
    nttb.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tc + tpho);
    nttb.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] + tb + tc);

    __syncthreads();
    //goto res;
    //TODO:为什么这里会存在溢出呢？
    //3.2
    group_num = idx / 7 + 24;
    group_idx = idx % 7;
    start = group_num * 21;

    zeta1 = zetas_n653_v2[64 + group_num * 2];
    zeta2 = zetas_n653_v2[64 + group_num * 2 + 1];

    tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * ntta.coeffs[start + group_idx + 7]);
    tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * ntta.coeffs[start + group_idx + 14]);
    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

    ntta.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tb - tpho);
    ntta.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tc + tpho);
    ntta.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] + tb + tc);


    tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * nttb.coeffs[start + group_idx + 7]);
    tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * nttb.coeffs[start + group_idx + 14]);
    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

    nttb.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tb - tpho);
    nttb.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tc + tpho);
    nttb.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] + tb + tc);

    __syncthreads();
    

    //3.3
    group_num = idx / 7 + 48;
    group_idx = idx % 7;
    start = group_num * 21;
    if(group_num < 64){
        zeta1 = zetas_n653_v2[64 + group_num * 2];
        zeta2 = zetas_n653_v2[64 + group_num * 2 + 1];

        tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * ntta.coeffs[start + group_idx + 7]);
        tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * ntta.coeffs[start + group_idx + 14]);
        tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

        ntta.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tb - tpho);
        ntta.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] - tc + tpho);
        ntta.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(ntta.coeffs[start + group_idx] + tb + tc);

        tb = montgomery_reduce_n653_cuda((int64_t)zeta1 * nttb.coeffs[start + group_idx + 7]);
        tc = montgomery_reduce_n653_cuda((int64_t)zeta2 * nttb.coeffs[start + group_idx + 14]);
        tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (tb - tc));

        nttb.coeffs[start + group_idx + 14] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tb - tpho);
        nttb.coeffs[start + group_idx + 7] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] - tc + tpho);
        nttb.coeffs[start + group_idx] = pseudomersenne_reduce_single_n653_cuda(nttb.coeffs[start + group_idx] + tb + tc);

    }
    __syncthreads();
/*
res:
    if(idx == 0){
        for(int i=0;i<N_N653;i++){
            printf("%d,",ntta.coeffs[i]);
        }
        printf("\n\n");

        for(int i=0;i<N_N653;i++){
            printf("%d,",nttb.coeffs[i]);
        }
        printf("\n\n");
    }
*/
    //basemul
    group_num = idx / 3;
    group_idx = idx % 3;

    zeta = zetas_n653_v2[N_N653 / 21 + group_num * 2];
    zeta = montgomery_reduce_n653_cuda((int64_t)zeta * root3_n653[group_idx]);

    basemul_n653(nttc.coeffs + 21 * group_num + 7 * group_idx, ntta.coeffs + 21 * group_num + 7 * group_idx, nttb.coeffs + 21 * group_num + 7 * group_idx, zeta);

    group_num = idx / 3 + 56;
    if(group_num < 64){
        zeta = zetas_n653_v2[N_N653 / 21 + group_num * 2];
        zeta = montgomery_reduce_n653_cuda((int64_t)zeta * root3_n653[group_idx]);

        basemul_n653(nttc.coeffs + 21 * group_num + 7 * group_idx, ntta.coeffs + 21 * group_num + 7 * group_idx, nttb.coeffs + 21 * group_num + 7 * group_idx, zeta);
    }

    __syncthreads();
    /*if(idx == 0){
        for(int i=0;i<N_N653;i++){
                printf("%d,",nttc.coeffs[i]);
            }
        printf("\n\n");
    }*/

   //invntt
   // 1 - radix 3
    group_num = idx / 7;
    group_idx = idx % 7;
    start = group_num * 21;

    zeta1 = zetas_inv_n653_v2[group_num * 2];
    zeta2 = zetas_inv_n653_v2[group_num * 2 + 1];

    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (nttc.coeffs[start + group_idx + 7] - nttc.coeffs[start + group_idx + 14]));
    tb = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 7] - tpho);
    tc = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 14] + tpho);

    nttc.coeffs[start + group_idx] = (nttc.coeffs[start + group_idx] + nttc.coeffs[start + group_idx + 7] + nttc.coeffs[start + group_idx + 14]);
    nttc.coeffs[start + group_idx + 7] = montgomery_reduce_n653_cuda((int64_t)zeta1 * tb);
    nttc.coeffs[start + group_idx + 14] = montgomery_reduce_n653_cuda((int64_t)zeta2 * tc);

    //2 - radix 3
    group_num = idx / 7 + 24;
    group_idx = idx % 7;
    start = group_num * 21;

    zeta1 = zetas_inv_n653_v2[group_num * 2];
    zeta2 = zetas_inv_n653_v2[group_num * 2 + 1];

    tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (nttc.coeffs[start + group_idx + 7] - nttc.coeffs[start + group_idx + 14]));
    tb = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 7] - tpho);
    tc = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 14] + tpho);

    nttc.coeffs[start + group_idx] = (nttc.coeffs[start + group_idx] + nttc.coeffs[start + group_idx + 7] + nttc.coeffs[start + group_idx + 14]);
    nttc.coeffs[start + group_idx + 7] = montgomery_reduce_n653_cuda((int64_t)zeta1 * tb);
    nttc.coeffs[start + group_idx + 14] = montgomery_reduce_n653_cuda((int64_t)zeta2 * tc);

    //3 - radix 3
    group_num = idx / 7 + 48;
    group_idx = idx % 7;
    start = group_num * 21;
    if(group_num < 64){
        zeta1 = zetas_inv_n653_v2[group_num * 2];
        zeta2 = zetas_inv_n653_v2[group_num * 2 + 1];

        tpho = montgomery_reduce_n653_cuda((int64_t)root3_n653[1] * (nttc.coeffs[start + group_idx + 7] - nttc.coeffs[start + group_idx + 14]));
        tb = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 7] - tpho);
        tc = (nttc.coeffs[start + group_idx] - nttc.coeffs[start + group_idx + 14] + tpho);

        nttc.coeffs[start + group_idx] = (nttc.coeffs[start + group_idx] + nttc.coeffs[start + group_idx + 7] + nttc.coeffs[start + group_idx + 14]);
        nttc.coeffs[start + group_idx + 7] = montgomery_reduce_n653_cuda((int64_t)zeta1 * tb);
        nttc.coeffs[start + group_idx + 14] = montgomery_reduce_n653_cuda((int64_t)zeta2 * tc);
    }

    __syncthreads();
    
    //3层- radix 2

    //1
    group_num = idx / 21;
    group_idx = idx % 21;
    start = group_num * 168;
    a[0] = nttc.coeffs[start + group_idx];
    a[1] = nttc.coeffs[start + group_idx + 21];
    a[2] = nttc.coeffs[start + group_idx + 21 * 2];
    a[3] = nttc.coeffs[start + group_idx + 21 * 3];
    a[4] = nttc.coeffs[start + group_idx + 21 * 4];
    a[5] = nttc.coeffs[start + group_idx + 21 * 5];
    a[6] = nttc.coeffs[start + group_idx + 21 * 6];
    a[7] = nttc.coeffs[start + group_idx + 21 * 7];

    zeta = zetas_inv_n653_v2[128 + group_num * 4];
    GS_a(0,1,zeta);
    
    zeta = zetas_inv_n653_v2[128 + group_num * 4 + 1];
    GS_a(2,3,zeta);

    zeta = zetas_inv_n653_v2[128 + group_num * 4 + 2];
    GS_a(4,5,zeta);

    zeta = zetas_inv_n653_v2[128 + group_num * 4 + 3];
    GS_a(6,7,zeta);

    //2
    zeta = zetas_inv_n653_v2[160 + group_num * 2];
    GS_a(0,2,zeta);
    GS_a(1,3,zeta);

    zeta = zetas_inv_n653_v2[160 + group_num * 2 + 1];
    GS_a(4,6,zeta);
    GS_a(5,7,zeta);

    //3
    zeta = zetas_inv_n653_v2[176 + group_num];
    GS_a(0,4,zeta);
    GS_a(1,5,zeta);
    GS_a(2,6,zeta);
    GS_a(3,7,zeta);

    nttc.coeffs[start + group_idx] = a[0];
    nttc.coeffs[start + group_idx + 21] = a[1];
    nttc.coeffs[start + group_idx + 21 * 2] = a[2];
    nttc.coeffs[start + group_idx + 21 * 3] = a[3];
    nttc.coeffs[start + group_idx + 21 * 4] = a[4];
    nttc.coeffs[start + group_idx + 21 * 5] = a[5];
    nttc.coeffs[start + group_idx + 21 * 6] = a[6];
    nttc.coeffs[start + group_idx + 21 * 7] = a[7];

    __syncthreads();

    /*if(idx == 0){
        for(int i=0;i<N_N653;i++){
                printf("%d,",nttc.coeffs[i]);
            }
        printf("\n\n");
    }*/

    a[0] = nttc.coeffs[idx];
    a[1] = nttc.coeffs[idx + 168];
    a[2] = nttc.coeffs[idx + 168 * 2];
    a[3] = nttc.coeffs[idx + 168 * 3];
    a[4] = nttc.coeffs[idx + 168 * 4];
    a[5] = nttc.coeffs[idx + 168 * 5];
    a[6] = nttc.coeffs[idx + 168 * 6];
    a[7] = nttc.coeffs[idx + 168 * 7];
    
    //1
    zeta = zetas_inv_n653_v2[184];
    GS_a(0,1,zeta);
    
    zeta = zetas_inv_n653_v2[185];
    GS_a(2,3,zeta);

    zeta = zetas_inv_n653_v2[186];
    GS_a(4,5,zeta);

    zeta = zetas_inv_n653_v2[187];
    GS_a(6,7,zeta);

    //2
    zeta = zetas_inv_n653_v2[188];
    GS_a(0,2,zeta);
    GS_a(1,3,zeta);

    zeta = zetas_inv_n653_v2[189];
    GS_a(4,6,zeta);
    GS_a(5,7,zeta);

    
    //3
    special_CT_a_pse(0,4);
    special_CT_a_pse(1,5);
    special_CT_a_pse(2,6);
    special_CT_a_pse(3,7);

    nttc.coeffs[idx] = a[0];
    nttc.coeffs[idx + 168] = a[1];
    nttc.coeffs[idx + 168 * 2] = a[2];
    nttc.coeffs[idx + 168 * 3] = a[3];
    nttc.coeffs[idx + 168 * 4] = a[4];
    nttc.coeffs[idx + 168 * 5] = a[5];
    nttc.coeffs[idx + 168 * 6] = a[6];
    nttc.coeffs[idx + 168 * 7] = a[7];

    __syncthreads();
    /*if(idx == 0){
        for(int i=0;i<N_N653;i++){
                printf("%d,",nttc.coeffs[i]);
            }
        printf("\n\n");
    }*/

    //只需要FPTRU_N个线程来完成 除去0和652
    array_c[blockIdx.x].coeffs[1 + idx * 3 ] = (montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[1 + idx * 3 ] + nttc.coeffs[1 + idx * 3  + FPTRU_N - 1] + nttc.coeffs[1 + idx * 3  + FPTRU_N]))) & (FPTRU_Q2 - 1);
    array_c[blockIdx.x].coeffs[2 + idx * 3  ] = (montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[2 + idx * 3 ] + nttc.coeffs[2 + idx * 3  + FPTRU_N - 1] + nttc.coeffs[2 + idx * 3  + FPTRU_N]))) & (FPTRU_Q2 - 1);

    array_c[blockIdx.x].coeffs[3 + idx * 3  ] = (montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[3 + idx * 3 ] + nttc.coeffs[3 + idx * 3 + FPTRU_N - 1] + nttc.coeffs[3 + idx * 3  + FPTRU_N]))) & (FPTRU_Q2 - 1);


    if(idx < 148){
        if(idx == 0){
            array_c[blockIdx.x].coeffs[0] = (montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[0] + nttc.coeffs[FPTRU_N]))) & (FPTRU_Q2 - 1);
            array_c[blockIdx.x].coeffs[FPTRU_N - 1] = (montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[FPTRU_N - 1] + nttc.coeffs[2 * FPTRU_N - 2]))) & (FPTRU_Q2 - 1);
        }
        else{
            array_c[blockIdx.x].coeffs[idx + 504] = (montgomery_reduce_n653_cuda((int64_t)FACTOR_NAKYFHAIRNF * (nttc.coeffs[idx + 504] + nttc.coeffs[idx + 504 + FPTRU_N - 1] + nttc.coeffs[idx + 504 + FPTRU_N]))) & (FPTRU_Q2 - 1);
        }
    }


}
