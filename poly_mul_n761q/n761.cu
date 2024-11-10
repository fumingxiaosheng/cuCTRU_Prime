#include "n761.h"

#include "../cbd.h"

/*2024-10-14
线程组织结构:
<<<BATCH_SIZE,192>>>*/

__device__ int32_t zetas_n761[N_N761 / 3] = {
    5592919, 33026177, 26790957, 33026177, 13499153, 21938565, 26790957, 33026177, 10764447, 19493511, 9265027, 5611999, 13499153, 21938565, 26790957, 33026177,
    31350138, 24124141, 10149685, 3198144, 32900457, 1140056, 24000204, 27372525, 10764447, 19493511, 9265027, 5611999, 13499153, 21938565, 26790957, 33026177,
    32738467, 16657861, 28138468, 501067, 16015129, 21620241, 1529124, 28870577, 18285219, 3050639, 27906268, 21100145, 5343767, 15280732, 12498775, 17354707,
    31350138, 24124141, 10149685, 3198144, 32900457, 1140056, 24000204, 27372525, 10764447, 19493511, 9265027, 5611999, 13499153, 21938565, 26790957, 33026177,
    9795452, 12096975, 33505735, 10865068, 16076627, 23564159, 31258325, 19464215, 24580770, 6067746, 18546326, 875322, 23054283, 15379670, 17941644, 2515533,
    17182522, 9742026, 11427911, 16642093, 24272961, 6983968, 22883012, 17558876, 10658623, 29624690, 1466133, 20016426, 18447657, 8460173, 19793647, 24333065,
    32738467, 16657861, 28138468, 501067, 16015129, 21620241, 1529124, 28870577, 18285219, 3050639, 27906268, 21100145, 5343767, 15280732, 12498775, 17354707,
    31350138, 24124141, 10149685, 3198144, 32900457, 1140056, 24000204, 27372525, 10764447, 19493511, 9265027, 5611999, 13499153, 21938565, 26790957, 33026177,
    10420445, 32038148, 2264, 33265110, 25958325, 11902971, 25622482, 27761690, 28792776, 33361188, 24969419, 12394212, 33193998, 6229980, 8743221, 4457820,
    28382998, 8431289, 1312494, 4681025, 29392120, 295408, 2951574, 9444158, 16620023, 32579155, 31375729, 11964204, 33164555, 16089224, 25578586, 22029388,
    19233249, 12984689, 30813171, 15944111, 13539352, 27143752, 7098522, 30855744, 15029351, 2274807, 31244694, 16958068, 23199393, 17755942, 6583826, 30969403,
    1918505, 22360437, 26047052, 3762725, 10986757, 5860566, 28381447, 3454840, 31955344, 13111549, 10268342, 11767139, 6958801, 13085226, 6345790, 33276057,
    9795452, 12096975, 33505735, 10865068, 16076627, 23564159, 31258325, 19464215, 24580770, 6067746, 18546326, 875322, 23054283, 15379670, 17941644, 2515533,
    17182522, 9742026, 11427911, 16642093, 24272961, 6983968, 22883012, 17558876, 10658623, 29624690, 1466133, 20016426, 18447657, 8460173, 19793647, 24333065,
    32738467, 16657861, 28138468, 501067, 16015129, 21620241, 1529124, 28870577, 18285219, 3050639, 27906268, 21100145, 5343767, 15280732, 12498775, 17354707,
    31350138, 24124141, 10149685, 3198144, 32900457, 1140056, 24000204, 27372525, 10764447, 19493511, 9265027, 5611999, 13499153, 21938565, 26790957, 33026177,
    5592919, 31075518, 21303665, 22682749, 11872764, 2432073, 19322199, 11292275, 23696636, 32659557, 23346277, 31556495, 21579819, 5882189, 6581391, 32328326,
    10696307, 16874833, 31719194, 6911566, 29285890, 23963075, 31256330, 9623777, 31413463, 32254537, 11613809, 24253081, 24356853, 24689701, 17056149, 15482269,
    10167038, 22751501, 18535303, 3968231, 1394219, 26379347, 16811031, 28313524, 16967280, 23729319, 31158593, 2331434, 549280, 18765598, 27900090, 4955113,
    1126043, 7409310, 26996082, 3493835, 5276286, 25397512, 30230071, 26868776, 1340079, 15120857, 24105137, 12241024, 12565260, 33504217, 7283953, 25114323,
    7865111, 32661754, 22892358, 6749048, 9488390, 10088648, 22153949, 5957087, 6543554, 17785943, 9563220, 4840264, 22490456, 6137845, 17318481, 17583519,
    24218356, 23955081, 26738484, 22876568, 3544406, 28057775, 18074286, 26082996, 32554080, 21141507, 15212341, 7347621, 25101513, 16054630, 18249116, 7050722,
    2250785, 7291436, 3750223, 26357349, 10321733, 17918503, 8792503, 33073869, 21274321, 10730693, 6218037, 32966155, 23321992, 20997298, 25154884, 8989918,
    5527986, 7143126, 31109078, 27760189, 9686871, 5400208, 13544313, 4038243, 9254376, 26381630, 18797971, 13377401, 350844, 19639872, 20608463, 28613855,
    10420445, 32038148, 2264, 33265110, 25958325, 11902971, 25622482, 27761690, 28792776, 33361188, 24969419, 12394212, 33193998, 6229980, 8743221, 4457820,
    28382998, 8431289, 1312494, 4681025, 29392120, 295408, 2951574, 9444158, 16620023, 32579155, 31375729, 11964204, 33164555, 16089224, 25578586, 22029388,
    19233249, 12984689, 30813171, 15944111, 13539352, 27143752, 7098522, 30855744, 15029351, 2274807, 31244694, 16958068, 23199393, 17755942, 6583826, 30969403,
    1918505, 22360437, 26047052, 3762725, 10986757, 5860566, 28381447, 3454840, 31955344, 13111549, 10268342, 11767139, 6958801, 13085226, 6345790, 33276057,
    9795452, 12096975, 33505735, 10865068, 16076627, 23564159, 31258325, 19464215, 24580770, 6067746, 18546326, 875322, 23054283, 15379670, 17941644, 2515533,
    17182522, 9742026, 11427911, 16642093, 24272961, 6983968, 22883012, 17558876, 10658623, 29624690, 1466133, 20016426, 18447657, 8460173, 19793647, 24333065,
    32738467, 16657861, 28138468, 501067, 16015129, 21620241, 1529124, 28870577, 18285219, 3050639, 27906268, 21100145, 5343767, 15280732, 12498775, 17354707,
    31350138, 24124141, 10149685, 3198144, 32900457, 1140056, 24000204, 27372525, 10764447, 19493511, 9265027, 5611999, 13499153, 21938565, 26790957, 33026177};

__device__ int32_t zetas_inv_n761[N_N761 / 3] = {
    4936482, 12941874, 13910465, 33199493, 20172936, 14752366, 7168707, 24295961, 29512094, 20006024, 28150129, 23863466, 5790148, 2441259, 26407211, 28022351,
    24560419, 8395453, 12553039, 10228345, 584182, 27332300, 22819644, 12276016, 476468, 24757834, 15631834, 23228604, 7192988, 29800114, 26258901, 31299552,
    26499615, 15301221, 17495707, 8448824, 26202716, 18337996, 12408830, 996257, 7467341, 15476051, 5492562, 30005931, 10673769, 6811853, 9595256, 9331981,
    15966818, 16231856, 27412492, 11059881, 28710073, 23987117, 15764394, 27006783, 27593250, 11396388, 23461689, 24061947, 26801289, 10657979, 888583, 25685226,
    8436014, 26266384, 46120, 20985077, 21309313, 9445200, 18429480, 32210258, 6681561, 3320266, 8152825, 28274051, 30056502, 6554255, 26141027, 32424294,
    28595224, 5650247, 14784739, 33001057, 31218903, 2391744, 9821018, 16583057, 5236813, 16739306, 7170990, 32156118, 29582106, 15015034, 10798836, 23383299,
    18068068, 16494188, 8860636, 9193484, 9297256, 21936528, 1295800, 2136874, 23926560, 2294007, 9587262, 4264447, 26638771, 1831143, 16675504, 22854030,
    1222011, 26968946, 27668148, 11970518, 1993842, 10204060, 890780, 9853701, 22258062, 14228138, 31118264, 21677573, 10867588, 12246672, 2474819, 27957418,
    274280, 27204547, 20465111, 26591536, 21783198, 23281995, 20438788, 1594993, 30095497, 5168890, 27689771, 22563580, 29787612, 7503285, 11189900, 31631832,
    2580934, 26966511, 15794395, 10350944, 16592269, 2305643, 31275530, 18520986, 2694593, 26451815, 6406585, 20010985, 17606226, 2737166, 20565648, 14317088,
    11520949, 7971751, 17461113, 385782, 21586133, 2174608, 971182, 16930314, 24106179, 30598763, 33254929, 4158217, 28869312, 32237843, 25119048, 5167339,
    29092517, 24807116, 27320357, 356339, 21156125, 8580918, 189149, 4757561, 5788647, 7927855, 21647366, 7592012, 285227, 33548073, 1512189, 23129892,
    9217272, 13756690, 25090164, 15102680, 13533911, 32084204, 3925647, 22891714, 15991461, 10667325, 26566369, 9277376, 16908244, 22122426, 23808311, 16367815,
    31034804, 15608693, 18170667, 10496054, 32675015, 15004011, 27482591, 8969567, 14086122, 2292012, 9986178, 17473710, 22685269, 44602, 21453362, 23754885,
    16195630, 21051562, 18269605, 28206570, 12450192, 5644069, 30499698, 15265118, 4679760, 32021213, 11930096, 17535208, 33049270, 5411869, 16892476, 811870,
    6177812, 9550133, 32410281, 649880, 30352193, 23400652, 9426196, 2200199, 27938338, 24285310, 14056826, 22785890, 11611772, 20051184, 6759380, 33026177,
    274280, 27204547, 20465111, 26591536, 21783198, 23281995, 20438788, 1594993, 30095497, 5168890, 27689771, 22563580, 29787612, 7503285, 11189900, 31631832,
    2580934, 26966511, 15794395, 10350944, 16592269, 2305643, 31275530, 18520986, 2694593, 26451815, 6406585, 20010985, 17606226, 2737166, 20565648, 14317088,
    11520949, 7971751, 17461113, 385782, 21586133, 2174608, 971182, 16930314, 24106179, 30598763, 33254929, 4158217, 28869312, 32237843, 25119048, 5167339,
    29092517, 24807116, 27320357, 356339, 21156125, 8580918, 189149, 4757561, 5788647, 7927855, 21647366, 7592012, 285227, 33548073, 1512189, 23129892,
    9217272, 13756690, 25090164, 15102680, 13533911, 32084204, 3925647, 22891714, 15991461, 10667325, 26566369, 9277376, 16908244, 22122426, 23808311, 16367815,
    31034804, 15608693, 18170667, 10496054, 32675015, 15004011, 27482591, 8969567, 14086122, 2292012, 9986178, 17473710, 22685269, 44602, 21453362, 23754885,
    16195630, 21051562, 18269605, 28206570, 12450192, 5644069, 30499698, 15265118, 4679760, 32021213, 11930096, 17535208, 33049270, 5411869, 16892476, 811870,
    6177812, 9550133, 32410281, 649880, 30352193, 23400652, 9426196, 2200199, 27938338, 24285310, 14056826, 22785890, 11611772, 20051184, 6759380, 33026177,
    9217272, 13756690, 25090164, 15102680, 13533911, 32084204, 3925647, 22891714, 15991461, 10667325, 26566369, 9277376, 16908244, 22122426, 23808311, 16367815,
    31034804, 15608693, 18170667, 10496054, 32675015, 15004011, 27482591, 8969567, 14086122, 2292012, 9986178, 17473710, 22685269, 44602, 21453362, 23754885,
    16195630, 21051562, 18269605, 28206570, 12450192, 5644069, 30499698, 15265118, 4679760, 32021213, 11930096, 17535208, 33049270, 5411869, 16892476, 811870,
    6177812, 9550133, 32410281, 649880, 30352193, 23400652, 9426196, 2200199, 27938338, 24285310, 14056826, 22785890, 11611772, 20051184, 6759380, 33026177,
    16195630, 21051562, 18269605, 28206570, 12450192, 5644069, 30499698, 15265118, 4679760, 32021213, 11930096, 17535208, 33049270, 5411869, 16892476, 811870,
    6177812, 9550133, 32410281, 649880, 30352193, 23400652, 9426196, 2200199, 27938338, 24285310, 14056826, 22785890, 11611772, 20051184, 6759380, 33026177,
    6177812, 9550133, 32410281, 649880, 30352193, 23400652, 9426196, 2200199, 27938338, 24285310, 14056826, 22785890, 11611772, 20051184, 6759380, 33026177,
    27938338, 24285310, 14056826, 22785890, 11611772, 20051184, 6759380, 33026177, 11611772, 20051184, 6759380, 33026177, 6759380, 33026177, 33026177};


#define special_CT_a(i,j)  t = a[j];a[j]=(a[i] + t);a[i] = (a[i] - t);
#define special_CT_b(i,j)  t = b[j];b[j]=(b[i] + t);b[i] = (b[i] - t);

#define pse_CT_a(i,j)  t = a[j];a[j]=(a[i] + t);a[i] = pseudomersenne_reduce_single_n761_cuda(a[i] - t);
#define pse_CT_b(i,j)  t = b[j];b[j]=(b[i] + t);b[i] = pseudomersenne_reduce_single_n761_cuda(b[i] - t);

#define CT_a(i,j,zeta)  t = montgomery_reduce_n761_cuda((int64_t)zeta * a[j]); a[j] = (a[i] - t); a[i] = (a[i] + t);

#define CT_b(i,j,zeta)  t = montgomery_reduce_n761_cuda((int64_t)zeta * b[j]); b[j] = (b[i] - t); b[i] = (b[i] + t);

#define GS_a(i,j,zeta) t = a[i]; a[i] = (t + a[j]); a[j] = (t - a[j]); a[j] = montgomery_reduce_n761_cuda((int64_t)zeta * a[j]);

#define KARA(a, b, x, y, d) ((montgomery_reduce_n761_cuda((int64_t)(a[x] + a[y]) * (b[x] + b[y])) - d[x] - d[y]))

__device__ int32_t montgomery_reduce_n761_cuda(int64_t a)
{
    int32_t t;
    t = (int32_t)a * NUMBER_QCMOSSH;
    t = (a - (int64_t)t * Q_N761) >> 32;
    return t;
}

__device__ int32_t pseudomersenne_reduce_single_n761_cuda(int32_t a)
{
    int32_t t0, t1;
    t0 = a & 0x1ffffff;
    t1 = a >> 25;
    t0 = (t1 << 12) + t0 - t1 - Q_N761;
    return t0;
}

__device__ void basemul_n761_cuda(int32_t c[3], const int32_t a[3], const int32_t b[3], const int32_t zeta)
{
    int i;
    int32_t d[3];
    int32_t cc[3];

    for (i = 0; i < 3; i++)
        d[i] = montgomery_reduce_n761_cuda((int64_t)a[i] * b[i]);

    cc[0] = (d[0] + montgomery_reduce_n761_cuda((int64_t)zeta * (KARA(a, b, 1, 2, d))));
    cc[1] = (KARA(a, b, 0, 1, d) + montgomery_reduce_n761_cuda((int64_t)zeta * (d[2])));
    cc[2] = (KARA(a, b, 0, 2, d) + d[1]);

    c[0] = cc[0],c[1]=cc[1],c[2]=cc[2];
}



__global__ void poly_mul_761_batch_q1(poly * array_c,poly * array_a,poly * array_b){
    __shared__ nttpoly_n761 ntta;
    __shared__ nttpoly_n761 nttb;
    __shared__ nttpoly_n761 nttc;

    int32_t a[8]={0};
    int32_t b[8]={0};

    int32_t zeta;
    int32_t t;

    a[0] = array_a[blockIdx.x].coeffs[threadIdx.x];
    a[1] = array_a[blockIdx.x].coeffs[threadIdx.x + blockDim.x];
    a[2] = array_a[blockIdx.x].coeffs[threadIdx.x + blockDim.x * 2];

    if(threadIdx.x + blockDim.x * 3 < FPTRU_N) a[3] = array_a[blockIdx.x].coeffs[threadIdx.x + blockDim.x * 3];


    b[0] = array_b[blockIdx.x].coeffs[threadIdx.x];
    b[1] = array_b[blockIdx.x].coeffs[threadIdx.x + blockDim.x];
    b[2] = array_b[blockIdx.x].coeffs[threadIdx.x + blockDim.x * 2];

    if(threadIdx.x + blockDim.x * 3 < FPTRU_N) b[3] = array_b[blockIdx.x].coeffs[threadIdx.x + blockDim.x * 3];


    //1 基2
    special_CT_a(0,4);
    special_CT_a(1,5);
    special_CT_a(2,6);
    special_CT_a(3,7);

    special_CT_b(0,4);
    special_CT_b(1,5);
    special_CT_b(2,6);
    special_CT_b(3,7);

    //1 基2
    zeta = zetas_n761[2];

    CT_a(0,2,zeta);
    CT_a(1,3,zeta);

    CT_b(0,2,zeta);
    CT_b(1,3,zeta);

    pse_CT_a(4,6);
    pse_CT_a(5,7);

    pse_CT_b(4,6);
    pse_CT_b(5,7);

    //3 基2
    zeta = zetas_n761[4];
    CT_a(0,1,zeta);
    CT_b(0,1,zeta);

    zeta = zetas_n761[5];
    CT_a(2,3,zeta);
    CT_b(2,3,zeta);

    zeta = zetas_n761[6];
    CT_a(4,5,zeta);
    CT_b(4,5,zeta);

    pse_CT_a(6,7);
    pse_CT_b(6,7);

    //放回共享内存
    ntta.coeffs[threadIdx.x] = a[0];
    ntta.coeffs[threadIdx.x + 192] = a[1];
    ntta.coeffs[threadIdx.x + 192 * 2] = a[2];
    ntta.coeffs[threadIdx.x + 192 * 3] = a[3];
    ntta.coeffs[threadIdx.x + 192 * 4] = a[4];
    ntta.coeffs[threadIdx.x + 192 * 5] = a[5];
    ntta.coeffs[threadIdx.x + 192 * 6] = a[6];
    ntta.coeffs[threadIdx.x + 192 * 7] = a[7];

    nttb.coeffs[threadIdx.x] = b[0];
    nttb.coeffs[threadIdx.x + 192] = b[1];
    nttb.coeffs[threadIdx.x + 192 * 2] = b[2];
    nttb.coeffs[threadIdx.x + 192 * 3] = b[3];
    nttb.coeffs[threadIdx.x + 192 * 4] = b[4];
    nttb.coeffs[threadIdx.x + 192 * 5] = b[5];
    nttb.coeffs[threadIdx.x + 192 * 6] = b[6];
    nttb.coeffs[threadIdx.x + 192 * 7] = b[7];

    __syncthreads();

    int group_num = threadIdx.x / 24;
    int group_idx = threadIdx.x % 24;
    int start = group_num * 192;

    a[0] = ntta.coeffs[start + group_idx];
    a[1] = ntta.coeffs[start + group_idx + 24];
    a[2] = ntta.coeffs[start + group_idx + 24 * 2];
    a[3] = ntta.coeffs[start + group_idx + 24 * 3];
    a[4] = ntta.coeffs[start + group_idx + 24 * 4];
    a[5] = ntta.coeffs[start + group_idx + 24 * 5];
    a[6] = ntta.coeffs[start + group_idx + 24 * 6];
    a[7] = ntta.coeffs[start + group_idx + 24 * 7];

    b[0] = nttb.coeffs[start + group_idx];
    b[1] = nttb.coeffs[start + group_idx + 24];
    b[2] = nttb.coeffs[start + group_idx + 24 * 2];
    b[3] = nttb.coeffs[start + group_idx + 24 * 3];
    b[4] = nttb.coeffs[start + group_idx + 24 * 4];
    b[5] = nttb.coeffs[start + group_idx + 24 * 5];
    b[6] = nttb.coeffs[start + group_idx + 24 * 6];
    b[7] = nttb.coeffs[start + group_idx + 24 * 7];


    //4 基2
    zeta = zetas_n761[8 + group_num];
    CT_a(0,4,zeta);
    CT_a(1,5,zeta);
    CT_a(2,6,zeta);
    CT_a(3,7,zeta);

    CT_b(0,4,zeta);
    CT_b(1,5,zeta);
    CT_b(2,6,zeta);
    CT_b(3,7,zeta);

    //5 基2
    zeta = zetas_n761[16 + group_num * 2];
    CT_a(0,2,zeta);
    CT_a(1,3,zeta);

    CT_b(0,2,zeta);
    CT_b(1,3,zeta);

    zeta = zetas_n761[16 + group_num * 2 + 1];
    CT_a(4,6,zeta);
    CT_a(5,7,zeta);

    CT_b(4,6,zeta);
    CT_b(5,7,zeta);

    //6 基2
    zeta = zetas_n761[32 + group_num * 4];
    CT_a(0,1,zeta);
    CT_b(0,1,zeta);

    zeta = zetas_n761[32 + group_num * 4 + 1];
    CT_a(2,3,zeta);
    CT_b(2,3,zeta);

    zeta = zetas_n761[32 + group_num * 4 + 2];
    CT_a(4,5,zeta);
    CT_b(4,5,zeta);

    zeta = zetas_n761[32 + group_num * 4 + 3];
    CT_a(6,7,zeta);
    CT_b(6,7,zeta);

    ntta.coeffs[start + group_idx] = a[0];
    ntta.coeffs[start + group_idx + 24] = a[1];
    ntta.coeffs[start + group_idx + 24 * 2] = a[2];
    ntta.coeffs[start + group_idx + 24 * 3] = a[3];
    ntta.coeffs[start + group_idx + 24 * 4] = a[4];
    ntta.coeffs[start + group_idx + 24 * 5] = a[5];
    ntta.coeffs[start + group_idx + 24 * 6] = a[6];
    ntta.coeffs[start + group_idx + 24 * 7] = a[7];

    nttb.coeffs[start + group_idx] = b[0];
    nttb.coeffs[start + group_idx + 24] = b[1];
    nttb.coeffs[start + group_idx + 24 * 2] = b[2];
    nttb.coeffs[start + group_idx + 24 * 3] = b[3];
    nttb.coeffs[start + group_idx + 24 * 4] = b[4];
    nttb.coeffs[start + group_idx + 24 * 5] = b[5];
    nttb.coeffs[start + group_idx + 24 * 6] = b[6];
    nttb.coeffs[start + group_idx + 24 * 7] = b[7];

    __syncthreads();

    group_num = threadIdx.x / 3;
    group_idx = threadIdx.x % 3;
    start = group_num * 24;

    a[0] = ntta.coeffs[start + group_idx];
    a[1] = ntta.coeffs[start + group_idx + 3];
    a[2] = ntta.coeffs[start + group_idx + 3 * 2];
    a[3] = ntta.coeffs[start + group_idx + 3 * 3];
    a[4] = ntta.coeffs[start + group_idx + 3 * 4];
    a[5] = ntta.coeffs[start + group_idx + 3 * 5];
    a[6] = ntta.coeffs[start + group_idx + 3 * 6];
    a[7] = ntta.coeffs[start + group_idx + 3 * 7];

    b[0] = nttb.coeffs[start + group_idx];
    b[1] = nttb.coeffs[start + group_idx + 3];
    b[2] = nttb.coeffs[start + group_idx + 3 * 2];
    b[3] = nttb.coeffs[start + group_idx + 3 * 3];
    b[4] = nttb.coeffs[start + group_idx + 3 * 4];
    b[5] = nttb.coeffs[start + group_idx + 3 * 5];
    b[6] = nttb.coeffs[start + group_idx + 3 * 6];
    b[7] = nttb.coeffs[start + group_idx + 3 * 7];

    //7 基2
    zeta = zetas_n761[64 + group_num];
    CT_a(0,4,zeta);
    CT_a(1,5,zeta);
    CT_a(2,6,zeta);
    CT_a(3,7,zeta);

    CT_b(0,4,zeta);
    CT_b(1,5,zeta);
    CT_b(2,6,zeta);
    CT_b(3,7,zeta);

    //8 基2
    zeta = zetas_n761[128 + group_num * 2];
    CT_a(0,2,zeta);
    CT_a(1,3,zeta);

    CT_b(0,2,zeta);
    CT_b(1,3,zeta);

    zeta = zetas_n761[128 + group_num * 2 + 1];
    CT_a(4,6,zeta);
    CT_a(5,7,zeta);

    CT_b(4,6,zeta);
    CT_b(5,7,zeta);

    //9 基2
    zeta = zetas_n761[256 + group_num * 4];
    CT_a(0,1,zeta);
    CT_b(0,1,zeta);

    zeta = zetas_n761[256 + group_num * 4 + 1];
    CT_a(2,3,zeta);
    CT_b(2,3,zeta);

    zeta = zetas_n761[256 + group_num * 4 + 2];
    CT_a(4,5,zeta);
    CT_b(4,5,zeta);

    zeta = zetas_n761[256 + group_num * 4 + 3];
    CT_a(6,7,zeta);
    CT_b(6,7,zeta);

    ntta.coeffs[start + group_idx] = a[0];
    ntta.coeffs[start + group_idx + 3] = a[1];
    ntta.coeffs[start + group_idx + 3 * 2] = a[2];
    ntta.coeffs[start + group_idx + 3 * 3] = a[3];
    ntta.coeffs[start + group_idx + 3 * 4] = a[4];
    ntta.coeffs[start + group_idx + 3 * 5] = a[5];
    ntta.coeffs[start + group_idx + 3 * 6] = a[6];
    ntta.coeffs[start + group_idx + 3 * 7] = a[7];

    nttb.coeffs[start + group_idx] = b[0];
    nttb.coeffs[start + group_idx + 3] = b[1];
    nttb.coeffs[start + group_idx + 3 * 2] = b[2];
    nttb.coeffs[start + group_idx + 3 * 3] = b[3];
    nttb.coeffs[start + group_idx + 3 * 4] = b[4];
    nttb.coeffs[start + group_idx + 3 * 5] = b[5];
    nttb.coeffs[start + group_idx + 3 * 6] = b[6];
    nttb.coeffs[start + group_idx + 3 * 7] = b[7];
    
    __syncthreads();
    
    //512 个 basemul

    //0-191个
    zeta = zetas_n761[256 + threadIdx.x / 2];
    if(threadIdx.x & 1) zeta = -zeta;
    basemul_n761_cuda(nttc.coeffs + 3 * threadIdx.x, ntta.coeffs + 3 * threadIdx.x,nttb.coeffs + 3 * threadIdx.x,zeta);

    //192 - 383
    zeta = zetas_n761[256 + threadIdx.x / 2 + 96];
    if(threadIdx.x & 1) zeta = -zeta;
    basemul_n761_cuda(nttc.coeffs + 3 * threadIdx.x + 576, ntta.coeffs + 3 * threadIdx.x + 576, nttb.coeffs + 3 * threadIdx.x + 576, zeta);

    if(threadIdx.x < 128){
        zeta = zetas_n761[256 + threadIdx.x / 2 + 192];
        if(threadIdx.x & 1) zeta = -zeta;
        basemul_n761_cuda(nttc.coeffs + 3 * threadIdx.x + 1152, ntta.coeffs + 3 * threadIdx.x + 1152, nttb.coeffs + 3 * threadIdx.x + 1152, zeta);
    }

    __syncthreads();


    //inv_ntt
    group_num = threadIdx.x / 3;
    group_idx = threadIdx.x % 3;
    start = group_num * 24;

    a[0] = nttc.coeffs[start + group_idx];
    a[1] = nttc.coeffs[start + group_idx + 3];
    a[2] = nttc.coeffs[start + group_idx + 3 * 2];
    a[3] = nttc.coeffs[start + group_idx + 3 * 3];
    a[4] = nttc.coeffs[start + group_idx + 3 * 4];
    a[5] = nttc.coeffs[start + group_idx + 3 * 5];
    a[6] = nttc.coeffs[start + group_idx + 3 * 6];
    a[7] = nttc.coeffs[start + group_idx + 3 * 7];


    //1 基2 256
    zeta = zetas_inv_n761[group_num * 4];
    GS_a(0,1,zeta);

    zeta = zetas_inv_n761[group_num * 4 + 1];
    GS_a(2,3,zeta);

    zeta = zetas_inv_n761[group_num * 4 + 2];
    GS_a(4,5,zeta);

    zeta = zetas_inv_n761[group_num * 4 + 3];
    GS_a(6,7,zeta);

    //2 基2 128
    zeta = zetas_inv_n761[256 + group_num * 2];
    GS_a(0,2,zeta);
    GS_a(1,3,zeta);

    zeta = zetas_inv_n761[256 + group_num * 2 + 1];
    GS_a(4,6,zeta);
    GS_a(5,7,zeta);

    //3 基2 64
    zeta = zetas_inv_n761[384 + group_num];
    GS_a(0,4,zeta);
    GS_a(1,5,zeta);
    GS_a(2,6,zeta);
    GS_a(3,7,zeta);

    nttc.coeffs[start + group_idx] = a[0];
    nttc.coeffs[start + group_idx + 3] = a[1];
    nttc.coeffs[start + group_idx + 3 * 2] = a[2];
    nttc.coeffs[start + group_idx + 3 * 3] = a[3];
    nttc.coeffs[start + group_idx + 3 * 4] = a[4];
    nttc.coeffs[start + group_idx + 3 * 5] = a[5];
    nttc.coeffs[start + group_idx + 3 * 6] = a[6];
    nttc.coeffs[start + group_idx + 3 * 7] = a[7];

    __syncthreads();

    group_num = threadIdx.x / 24;
    group_idx = threadIdx.x % 24;
    start = group_num * 192;

    a[0] = nttc.coeffs[start + group_idx];
    a[1] = nttc.coeffs[start + group_idx + 24];
    a[2] = nttc.coeffs[start + group_idx + 24 * 2];
    a[3] = nttc.coeffs[start + group_idx + 24 * 3];
    a[4] = nttc.coeffs[start + group_idx + 24 * 4];
    a[5] = nttc.coeffs[start + group_idx + 24 * 5];
    a[6] = nttc.coeffs[start + group_idx + 24 * 6];
    a[7] = nttc.coeffs[start + group_idx + 24 * 7];

    //4 基2 32
    zeta = zetas_inv_n761[448 + group_num * 4];
    GS_a(0,1,zeta);

    zeta = zetas_inv_n761[448 + group_num * 4 + 1];
    GS_a(2,3,zeta);

    zeta = zetas_inv_n761[448 + group_num * 4 + 2];
    GS_a(4,5,zeta);

    zeta = zetas_inv_n761[448 + group_num * 4 + 3];
    GS_a(6,7,zeta);

#pragma unroll
    for(int i=0;i<8;i++){
        pseudomersenne_reduce_single_n761_cuda(a[i]);
        pseudomersenne_reduce_single_n761_cuda(b[i]);
    }
    
    //5 基2 16
    zeta = zetas_inv_n761[480 + group_num * 2];
    GS_a(0,2,zeta);
    GS_a(1,3,zeta);

    zeta = zetas_inv_n761[480 + group_num * 2 + 1];
    GS_a(4,6,zeta);
    GS_a(5,7,zeta);

    //6 基2 8
    zeta = zetas_inv_n761[496 + group_num];
    GS_a(0,4,zeta);
    GS_a(1,5,zeta);
    GS_a(2,6,zeta);
    GS_a(3,7,zeta);

    nttc.coeffs[start + group_idx] = a[0];
    nttc.coeffs[start + group_idx + 24] = a[1];
    nttc.coeffs[start + group_idx + 24 * 2] = a[2];
    nttc.coeffs[start + group_idx + 24 * 3] = a[3];
    nttc.coeffs[start + group_idx + 24 * 4] = a[4];
    nttc.coeffs[start + group_idx + 24 * 5] = a[5];
    nttc.coeffs[start + group_idx + 24 * 6] = a[6];
    nttc.coeffs[start + group_idx + 24 * 7] = a[7];

    __syncthreads();
    

    a[0] = nttc.coeffs[threadIdx.x];
    a[1] = nttc.coeffs[threadIdx.x + 192];
    a[2] = nttc.coeffs[threadIdx.x + 192 * 2];
    a[3] = nttc.coeffs[threadIdx.x + 192 * 3];
    a[4] = nttc.coeffs[threadIdx.x + 192 * 4];
    a[5] = nttc.coeffs[threadIdx.x + 192 * 5];
    a[6] = nttc.coeffs[threadIdx.x + 192 * 6];
    a[7] = nttc.coeffs[threadIdx.x + 192 * 7];

    //7 基2 4
    zeta = zetas_inv_n761[504];
    GS_a(0,1,zeta);

    zeta = zetas_inv_n761[504 + 1];
    GS_a(2,3,zeta);

    zeta = zetas_inv_n761[504 + 2];
    GS_a(4,5,zeta);

    zeta = zetas_inv_n761[504 + 3];
    GS_a(6,7,zeta);

    //8 基2 2
    zeta = zetas_inv_n761[508];
    GS_a(0,2,zeta);
    GS_a(1,3,zeta);

    zeta = zetas_inv_n761[508 + 1];
    GS_a(4,6,zeta);
    GS_a(5,7,zeta);

    //9 基2 1
    zeta = zetas_inv_n761[510];
    GS_a(0,4,zeta);
    GS_a(1,5,zeta);
    GS_a(2,6,zeta);
    GS_a(3,7,zeta);

    nttc.coeffs[threadIdx.x] = a[0];
    nttc.coeffs[threadIdx.x + 192] = a[1];
    nttc.coeffs[threadIdx.x + 192 * 2] = a[2];
    nttc.coeffs[threadIdx.x + 192 * 3] = a[3];
    nttc.coeffs[threadIdx.x + 192 * 4] = a[4];
    nttc.coeffs[threadIdx.x + 192 * 5] = a[5];
    nttc.coeffs[threadIdx.x + 192 * 6] = a[6];
    nttc.coeffs[threadIdx.x + 192 * 7] = a[7];

    __syncthreads();
    
    //1 - 192
    array_c[blockIdx.x].coeffs[threadIdx.x + 1] = fq_freeze(montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (nttc.coeffs[threadIdx.x + 1] + nttc.coeffs[threadIdx.x + 1 + FPTRU_N - 1] + nttc.coeffs[threadIdx.x + 1 + FPTRU_N])));
    
    //193 - 384
    array_c[blockIdx.x].coeffs[threadIdx.x + 193] = fq_freeze(montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (nttc.coeffs[threadIdx.x + 193] + nttc.coeffs[threadIdx.x + 193 + FPTRU_N - 1] + nttc.coeffs[threadIdx.x + 193 + FPTRU_N])));

    //385 - 576
    array_c[blockIdx.x].coeffs[threadIdx.x + 385] = fq_freeze(montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (nttc.coeffs[threadIdx.x + 385] + nttc.coeffs[threadIdx.x + 385 + FPTRU_N - 1] + nttc.coeffs[threadIdx.x + 385 + FPTRU_N])));
    
    //0,760, 577-759
    if(threadIdx.x <= 183){
        if(threadIdx.x == 183){
            array_c[blockIdx.x].coeffs[0] = fq_freeze(montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (nttc.coeffs[0] + nttc.coeffs[FPTRU_N])));
            array_c[blockIdx.x].coeffs[FPTRU_N - 1] = fq_freeze(montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (nttc.coeffs[FPTRU_N - 1] + nttc.coeffs[2 * FPTRU_N - 2])));
        }
        else{
            array_c[blockIdx.x].coeffs[threadIdx.x + 577] = fq_freeze(montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (nttc.coeffs[threadIdx.x + 577] + nttc.coeffs[threadIdx.x + 577 + FPTRU_N - 1] + nttc.coeffs[threadIdx.x + 577 + FPTRU_N])));
        }
    }

}


__global__ void poly_mul_761_batch_q2(poly * array_c,poly * array_a,poly * array_b){
    __shared__ nttpoly_n761 ntta;
    __shared__ nttpoly_n761 nttb;
    __shared__ nttpoly_n761 nttc;

    int32_t a[8]={0};
    int32_t b[8]={0};

    int32_t zeta;
    int32_t t;

    a[0] = array_a[blockIdx.x].coeffs[threadIdx.x];
    a[1] = array_a[blockIdx.x].coeffs[threadIdx.x + blockDim.x];
    a[2] = array_a[blockIdx.x].coeffs[threadIdx.x + blockDim.x * 2];

    if(threadIdx.x + blockDim.x * 3 < FPTRU_N) a[3] = array_a[blockIdx.x].coeffs[threadIdx.x + blockDim.x * 3];
    // else a[3] = 0;
    // a[4] = 0;
    // a[5] = 0;
    // a[6] = 0;
    // a[7] = 0;



    b[0] = array_b[blockIdx.x].coeffs[threadIdx.x];
    b[1] = array_b[blockIdx.x].coeffs[threadIdx.x + blockDim.x];
    b[2] = array_b[blockIdx.x].coeffs[threadIdx.x + blockDim.x * 2];

    if(threadIdx.x + blockDim.x * 3 < FPTRU_N) b[3] = array_b[blockIdx.x].coeffs[threadIdx.x + blockDim.x * 3];
    // else b[3] = 0;
    // b[4] = 0;
    // b[5] = 0;
    // b[6] = 0;
    // b[7] = 0;


    //1 基2
    special_CT_a(0,4);
    special_CT_a(1,5);
    special_CT_a(2,6);
    special_CT_a(3,7);

    special_CT_b(0,4);
    special_CT_b(1,5);
    special_CT_b(2,6);
    special_CT_b(3,7);

    //1 基2
    zeta = zetas_n761[2];

    CT_a(0,2,zeta);
    CT_a(1,3,zeta);

    CT_b(0,2,zeta);
    CT_b(1,3,zeta);

    pse_CT_a(4,6);
    pse_CT_a(5,7);

    pse_CT_b(4,6);
    pse_CT_b(5,7);

    //3 基2
    zeta = zetas_n761[4];
    CT_a(0,1,zeta);
    CT_b(0,1,zeta);

    zeta = zetas_n761[5];
    CT_a(2,3,zeta);
    CT_b(2,3,zeta);

    zeta = zetas_n761[6];
    CT_a(4,5,zeta);
    CT_b(4,5,zeta);

    pse_CT_a(6,7);
    pse_CT_b(6,7);

    //放回共享内存
    ntta.coeffs[threadIdx.x] = a[0];
    ntta.coeffs[threadIdx.x + 192] = a[1];
    ntta.coeffs[threadIdx.x + 192 * 2] = a[2];
    ntta.coeffs[threadIdx.x + 192 * 3] = a[3];
    ntta.coeffs[threadIdx.x + 192 * 4] = a[4];
    ntta.coeffs[threadIdx.x + 192 * 5] = a[5];
    ntta.coeffs[threadIdx.x + 192 * 6] = a[6];
    ntta.coeffs[threadIdx.x + 192 * 7] = a[7];

    nttb.coeffs[threadIdx.x] = b[0];
    nttb.coeffs[threadIdx.x + 192] = b[1];
    nttb.coeffs[threadIdx.x + 192 * 2] = b[2];
    nttb.coeffs[threadIdx.x + 192 * 3] = b[3];
    nttb.coeffs[threadIdx.x + 192 * 4] = b[4];
    nttb.coeffs[threadIdx.x + 192 * 5] = b[5];
    nttb.coeffs[threadIdx.x + 192 * 6] = b[6];
    nttb.coeffs[threadIdx.x + 192 * 7] = b[7];

    __syncthreads();

    int group_num = threadIdx.x / 24;
    int group_idx = threadIdx.x % 24;
    int start = group_num * 192;

    a[0] = ntta.coeffs[start + group_idx];
    a[1] = ntta.coeffs[start + group_idx + 24];
    a[2] = ntta.coeffs[start + group_idx + 24 * 2];
    a[3] = ntta.coeffs[start + group_idx + 24 * 3];
    a[4] = ntta.coeffs[start + group_idx + 24 * 4];
    a[5] = ntta.coeffs[start + group_idx + 24 * 5];
    a[6] = ntta.coeffs[start + group_idx + 24 * 6];
    a[7] = ntta.coeffs[start + group_idx + 24 * 7];

    b[0] = nttb.coeffs[start + group_idx];
    b[1] = nttb.coeffs[start + group_idx + 24];
    b[2] = nttb.coeffs[start + group_idx + 24 * 2];
    b[3] = nttb.coeffs[start + group_idx + 24 * 3];
    b[4] = nttb.coeffs[start + group_idx + 24 * 4];
    b[5] = nttb.coeffs[start + group_idx + 24 * 5];
    b[6] = nttb.coeffs[start + group_idx + 24 * 6];
    b[7] = nttb.coeffs[start + group_idx + 24 * 7];


    //4 基2
    zeta = zetas_n761[8 + group_num];
    CT_a(0,4,zeta);
    CT_a(1,5,zeta);
    CT_a(2,6,zeta);
    CT_a(3,7,zeta);

    CT_b(0,4,zeta);
    CT_b(1,5,zeta);
    CT_b(2,6,zeta);
    CT_b(3,7,zeta);

    //5 基2
    zeta = zetas_n761[16 + group_num * 2];
    CT_a(0,2,zeta);
    CT_a(1,3,zeta);

    CT_b(0,2,zeta);
    CT_b(1,3,zeta);

    zeta = zetas_n761[16 + group_num * 2 + 1];
    CT_a(4,6,zeta);
    CT_a(5,7,zeta);

    CT_b(4,6,zeta);
    CT_b(5,7,zeta);

    //6 基2
    zeta = zetas_n761[32 + group_num * 4];
    CT_a(0,1,zeta);
    CT_b(0,1,zeta);

    zeta = zetas_n761[32 + group_num * 4 + 1];
    CT_a(2,3,zeta);
    CT_b(2,3,zeta);

    zeta = zetas_n761[32 + group_num * 4 + 2];
    CT_a(4,5,zeta);
    CT_b(4,5,zeta);

    zeta = zetas_n761[32 + group_num * 4 + 3];
    CT_a(6,7,zeta);
    CT_b(6,7,zeta);

    ntta.coeffs[start + group_idx] = a[0];
    ntta.coeffs[start + group_idx + 24] = a[1];
    ntta.coeffs[start + group_idx + 24 * 2] = a[2];
    ntta.coeffs[start + group_idx + 24 * 3] = a[3];
    ntta.coeffs[start + group_idx + 24 * 4] = a[4];
    ntta.coeffs[start + group_idx + 24 * 5] = a[5];
    ntta.coeffs[start + group_idx + 24 * 6] = a[6];
    ntta.coeffs[start + group_idx + 24 * 7] = a[7];

    nttb.coeffs[start + group_idx] = b[0];
    nttb.coeffs[start + group_idx + 24] = b[1];
    nttb.coeffs[start + group_idx + 24 * 2] = b[2];
    nttb.coeffs[start + group_idx + 24 * 3] = b[3];
    nttb.coeffs[start + group_idx + 24 * 4] = b[4];
    nttb.coeffs[start + group_idx + 24 * 5] = b[5];
    nttb.coeffs[start + group_idx + 24 * 6] = b[6];
    nttb.coeffs[start + group_idx + 24 * 7] = b[7];

    __syncthreads();

    group_num = threadIdx.x / 3;
    group_idx = threadIdx.x % 3;
    start = group_num * 24;

    a[0] = ntta.coeffs[start + group_idx];
    a[1] = ntta.coeffs[start + group_idx + 3];
    a[2] = ntta.coeffs[start + group_idx + 3 * 2];
    a[3] = ntta.coeffs[start + group_idx + 3 * 3];
    a[4] = ntta.coeffs[start + group_idx + 3 * 4];
    a[5] = ntta.coeffs[start + group_idx + 3 * 5];
    a[6] = ntta.coeffs[start + group_idx + 3 * 6];
    a[7] = ntta.coeffs[start + group_idx + 3 * 7];

    b[0] = nttb.coeffs[start + group_idx];
    b[1] = nttb.coeffs[start + group_idx + 3];
    b[2] = nttb.coeffs[start + group_idx + 3 * 2];
    b[3] = nttb.coeffs[start + group_idx + 3 * 3];
    b[4] = nttb.coeffs[start + group_idx + 3 * 4];
    b[5] = nttb.coeffs[start + group_idx + 3 * 5];
    b[6] = nttb.coeffs[start + group_idx + 3 * 6];
    b[7] = nttb.coeffs[start + group_idx + 3 * 7];

    //7 基2
    zeta = zetas_n761[64 + group_num];
    CT_a(0,4,zeta);
    CT_a(1,5,zeta);
    CT_a(2,6,zeta);
    CT_a(3,7,zeta);

    CT_b(0,4,zeta);
    CT_b(1,5,zeta);
    CT_b(2,6,zeta);
    CT_b(3,7,zeta);

    //8 基2
    zeta = zetas_n761[128 + group_num * 2];
    CT_a(0,2,zeta);
    CT_a(1,3,zeta);

    CT_b(0,2,zeta);
    CT_b(1,3,zeta);

    zeta = zetas_n761[128 + group_num * 2 + 1];
    CT_a(4,6,zeta);
    CT_a(5,7,zeta);

    CT_b(4,6,zeta);
    CT_b(5,7,zeta);

    //9 基2
    zeta = zetas_n761[256 + group_num * 4];
    CT_a(0,1,zeta);
    CT_b(0,1,zeta);

    zeta = zetas_n761[256 + group_num * 4 + 1];
    CT_a(2,3,zeta);
    CT_b(2,3,zeta);

    zeta = zetas_n761[256 + group_num * 4 + 2];
    CT_a(4,5,zeta);
    CT_b(4,5,zeta);

    zeta = zetas_n761[256 + group_num * 4 + 3];
    CT_a(6,7,zeta);
    CT_b(6,7,zeta);

    ntta.coeffs[start + group_idx] = a[0];
    ntta.coeffs[start + group_idx + 3] = a[1];
    ntta.coeffs[start + group_idx + 3 * 2] = a[2];
    ntta.coeffs[start + group_idx + 3 * 3] = a[3];
    ntta.coeffs[start + group_idx + 3 * 4] = a[4];
    ntta.coeffs[start + group_idx + 3 * 5] = a[5];
    ntta.coeffs[start + group_idx + 3 * 6] = a[6];
    ntta.coeffs[start + group_idx + 3 * 7] = a[7];

    nttb.coeffs[start + group_idx] = b[0];
    nttb.coeffs[start + group_idx + 3] = b[1];
    nttb.coeffs[start + group_idx + 3 * 2] = b[2];
    nttb.coeffs[start + group_idx + 3 * 3] = b[3];
    nttb.coeffs[start + group_idx + 3 * 4] = b[4];
    nttb.coeffs[start + group_idx + 3 * 5] = b[5];
    nttb.coeffs[start + group_idx + 3 * 6] = b[6];
    nttb.coeffs[start + group_idx + 3 * 7] = b[7];
    
    __syncthreads();
    // printf("NTTA result\n");
    // if(threadIdx.x == 0 && blockIdx.x == 0){
        //printf("NTTA result\n");
        //int i = 1 + 1;不行还是会出错 zhc4.txt
        // for(int i=0;i<N_N761;i++){
        //     printf("%d,",ntta.coeffs[i]);
        // }
        // printf("\n\n");
        // printf("NTTB result\n");
        // for(int i=0;i<N_N761;i++){
        //     printf("%d,",nttb.coeffs[i]);
        // }
        // printf("\n\n");
    //}
    //__syncthreads();
    //512 个 basemul

    //0-191个
    zeta = zetas_n761[256 + threadIdx.x / 2];
    if(threadIdx.x & 1) zeta = -zeta;
    basemul_n761_cuda(nttc.coeffs + 3 * threadIdx.x, ntta.coeffs + 3 * threadIdx.x,nttb.coeffs + 3 * threadIdx.x,zeta);

    //192 - 383
    zeta = zetas_n761[256 + threadIdx.x / 2 + 96];
    if(threadIdx.x & 1) zeta = -zeta;
    basemul_n761_cuda(nttc.coeffs + 3 * threadIdx.x + 576, ntta.coeffs + 3 * threadIdx.x + 576, nttb.coeffs + 3 * threadIdx.x + 576, zeta);

    if(threadIdx.x < 128){
        zeta = zetas_n761[256 + threadIdx.x / 2 + 192];
        if(threadIdx.x & 1) zeta = -zeta;
        basemul_n761_cuda(nttc.coeffs + 3 * threadIdx.x + 1152, ntta.coeffs + 3 * threadIdx.x + 1152, nttb.coeffs + 3 * threadIdx.x + 1152, zeta);
    }

    __syncthreads();


    //inv_ntt
    group_num = threadIdx.x / 3;
    group_idx = threadIdx.x % 3;
    start = group_num * 24;

    a[0] = nttc.coeffs[start + group_idx];
    a[1] = nttc.coeffs[start + group_idx + 3];
    a[2] = nttc.coeffs[start + group_idx + 3 * 2];
    a[3] = nttc.coeffs[start + group_idx + 3 * 3];
    a[4] = nttc.coeffs[start + group_idx + 3 * 4];
    a[5] = nttc.coeffs[start + group_idx + 3 * 5];
    a[6] = nttc.coeffs[start + group_idx + 3 * 6];
    a[7] = nttc.coeffs[start + group_idx + 3 * 7];


    //1 基2 256
    zeta = zetas_inv_n761[group_num * 4];
    GS_a(0,1,zeta);

    zeta = zetas_inv_n761[group_num * 4 + 1];
    GS_a(2,3,zeta);

    zeta = zetas_inv_n761[group_num * 4 + 2];
    GS_a(4,5,zeta);

    zeta = zetas_inv_n761[group_num * 4 + 3];
    GS_a(6,7,zeta);

    //2 基2 128
    zeta = zetas_inv_n761[256 + group_num * 2];
    GS_a(0,2,zeta);
    GS_a(1,3,zeta);

    zeta = zetas_inv_n761[256 + group_num * 2 + 1];
    GS_a(4,6,zeta);
    GS_a(5,7,zeta);

    //3 基2 64
    zeta = zetas_inv_n761[384 + group_num];
    GS_a(0,4,zeta);
    GS_a(1,5,zeta);
    GS_a(2,6,zeta);
    GS_a(3,7,zeta);

    nttc.coeffs[start + group_idx] = a[0];
    nttc.coeffs[start + group_idx + 3] = a[1];
    nttc.coeffs[start + group_idx + 3 * 2] = a[2];
    nttc.coeffs[start + group_idx + 3 * 3] = a[3];
    nttc.coeffs[start + group_idx + 3 * 4] = a[4];
    nttc.coeffs[start + group_idx + 3 * 5] = a[5];
    nttc.coeffs[start + group_idx + 3 * 6] = a[6];
    nttc.coeffs[start + group_idx + 3 * 7] = a[7];

    __syncthreads();

    group_num = threadIdx.x / 24;
    group_idx = threadIdx.x % 24;
    start = group_num * 192;

    a[0] = nttc.coeffs[start + group_idx];
    a[1] = nttc.coeffs[start + group_idx + 24];
    a[2] = nttc.coeffs[start + group_idx + 24 * 2];
    a[3] = nttc.coeffs[start + group_idx + 24 * 3];
    a[4] = nttc.coeffs[start + group_idx + 24 * 4];
    a[5] = nttc.coeffs[start + group_idx + 24 * 5];
    a[6] = nttc.coeffs[start + group_idx + 24 * 6];
    a[7] = nttc.coeffs[start + group_idx + 24 * 7];

    //4 基2 32
    zeta = zetas_inv_n761[448 + group_num * 4];
    GS_a(0,1,zeta);

    zeta = zetas_inv_n761[448 + group_num * 4 + 1];
    GS_a(2,3,zeta);

    zeta = zetas_inv_n761[448 + group_num * 4 + 2];
    GS_a(4,5,zeta);

    zeta = zetas_inv_n761[448 + group_num * 4 + 3];
    GS_a(6,7,zeta);

#pragma unroll
    for(int i=0;i<8;i++){
        pseudomersenne_reduce_single_n761_cuda(a[i]);
        pseudomersenne_reduce_single_n761_cuda(b[i]);
    }
    
    //5 基2 16
    zeta = zetas_inv_n761[480 + group_num * 2];
    GS_a(0,2,zeta);
    GS_a(1,3,zeta);

    zeta = zetas_inv_n761[480 + group_num * 2 + 1];
    GS_a(4,6,zeta);
    GS_a(5,7,zeta);

    //6 基2 8
    zeta = zetas_inv_n761[496 + group_num];
    GS_a(0,4,zeta);
    GS_a(1,5,zeta);
    GS_a(2,6,zeta);
    GS_a(3,7,zeta);

    nttc.coeffs[start + group_idx] = a[0];
    nttc.coeffs[start + group_idx + 24] = a[1];
    nttc.coeffs[start + group_idx + 24 * 2] = a[2];
    nttc.coeffs[start + group_idx + 24 * 3] = a[3];
    nttc.coeffs[start + group_idx + 24 * 4] = a[4];
    nttc.coeffs[start + group_idx + 24 * 5] = a[5];
    nttc.coeffs[start + group_idx + 24 * 6] = a[6];
    nttc.coeffs[start + group_idx + 24 * 7] = a[7];

    __syncthreads();
    

    a[0] = nttc.coeffs[threadIdx.x];
    a[1] = nttc.coeffs[threadIdx.x + 192];
    a[2] = nttc.coeffs[threadIdx.x + 192 * 2];
    a[3] = nttc.coeffs[threadIdx.x + 192 * 3];
    a[4] = nttc.coeffs[threadIdx.x + 192 * 4];
    a[5] = nttc.coeffs[threadIdx.x + 192 * 5];
    a[6] = nttc.coeffs[threadIdx.x + 192 * 6];
    a[7] = nttc.coeffs[threadIdx.x + 192 * 7];

    //7 基2 4
    zeta = zetas_inv_n761[504];
    GS_a(0,1,zeta);

    zeta = zetas_inv_n761[504 + 1];
    GS_a(2,3,zeta);

    zeta = zetas_inv_n761[504 + 2];
    GS_a(4,5,zeta);

    zeta = zetas_inv_n761[504 + 3];
    GS_a(6,7,zeta);

    //8 基2 2
    zeta = zetas_inv_n761[508];
    GS_a(0,2,zeta);
    GS_a(1,3,zeta);

    zeta = zetas_inv_n761[508 + 1];
    GS_a(4,6,zeta);
    GS_a(5,7,zeta);

    //9 基2 1
    zeta = zetas_inv_n761[510];
    GS_a(0,4,zeta);
    GS_a(1,5,zeta);
    GS_a(2,6,zeta);
    GS_a(3,7,zeta);

    nttc.coeffs[threadIdx.x] = a[0];
    nttc.coeffs[threadIdx.x + 192] = a[1];
    nttc.coeffs[threadIdx.x + 192 * 2] = a[2];
    nttc.coeffs[threadIdx.x + 192 * 3] = a[3];
    nttc.coeffs[threadIdx.x + 192 * 4] = a[4];
    nttc.coeffs[threadIdx.x + 192 * 5] = a[5];
    nttc.coeffs[threadIdx.x + 192 * 6] = a[6];
    nttc.coeffs[threadIdx.x + 192 * 7] = a[7];

    __syncthreads();
    
    //1 - 192
    array_c[blockIdx.x].coeffs[threadIdx.x + 1] = (montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (nttc.coeffs[threadIdx.x + 1] + nttc.coeffs[threadIdx.x + 1 + FPTRU_N - 1] + nttc.coeffs[threadIdx.x + 1 + FPTRU_N]))) & (FPTRU_Q2 - 1);
    
    //193 - 384
    array_c[blockIdx.x].coeffs[threadIdx.x + 193] = (montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (nttc.coeffs[threadIdx.x + 193] + nttc.coeffs[threadIdx.x + 193 + FPTRU_N - 1] + nttc.coeffs[threadIdx.x + 193 + FPTRU_N]))) & (FPTRU_Q2 - 1);

    //385 - 576
    array_c[blockIdx.x].coeffs[threadIdx.x + 385] = (montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (nttc.coeffs[threadIdx.x + 385] + nttc.coeffs[threadIdx.x + 385 + FPTRU_N - 1] + nttc.coeffs[threadIdx.x + 385 + FPTRU_N]))) & (FPTRU_Q2 - 1);;
    
    //0,760, 577-759
    if(threadIdx.x <= 183){
        if(threadIdx.x == 183){
            array_c[blockIdx.x].coeffs[0] = (montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (nttc.coeffs[0] + nttc.coeffs[FPTRU_N]))) & (FPTRU_Q2 - 1);
            array_c[blockIdx.x].coeffs[FPTRU_N - 1] = (montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (nttc.coeffs[FPTRU_N - 1] + nttc.coeffs[2 * FPTRU_N - 2]))) & (FPTRU_Q2 - 1);
        }
        else{
            array_c[blockIdx.x].coeffs[threadIdx.x + 577] = (montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (nttc.coeffs[threadIdx.x + 577] + nttc.coeffs[threadIdx.x + 577 + FPTRU_N - 1] + nttc.coeffs[threadIdx.x + 577 + FPTRU_N]))) & (FPTRU_Q2 - 1);;
        }
    }

}




__device__  void d_poly_mul_q1_761(nttpoly_n761 & ntta, nttpoly_n761 & nttb, poly * array_a,poly * array_c){
    int32_t a[8]={0};
    int32_t b[8]={0};

    int32_t zeta;
    int32_t t;

    


    a[0] = array_a[blockIdx.x].coeffs[threadIdx.x];
    a[1] = array_a[blockIdx.x].coeffs[threadIdx.x + blockDim.x];
    a[2] = array_a[blockIdx.x].coeffs[threadIdx.x + blockDim.x * 2];

    if(threadIdx.x + blockDim.x * 3 < FPTRU_N) a[3] = array_a[blockIdx.x].coeffs[threadIdx.x + blockDim.x * 3];

    b[0] = nttb.coeffs[threadIdx.x];
    b[1] = nttb.coeffs[threadIdx.x + blockDim.x];
    b[2] = nttb.coeffs[threadIdx.x + blockDim.x * 2];
    b[3] = nttb.coeffs[threadIdx.x + blockDim.x * 3];
    b[4] = nttb.coeffs[threadIdx.x + blockDim.x * 4];
    b[5] = nttb.coeffs[threadIdx.x + blockDim.x * 5];
    b[6] = nttb.coeffs[threadIdx.x + blockDim.x * 6];
    b[7] = nttb.coeffs[threadIdx.x + blockDim.x * 7];

    //1 基2
    special_CT_a(0,4);
    special_CT_a(1,5);
    special_CT_a(2,6);
    special_CT_a(3,7);

    special_CT_b(0,4);
    special_CT_b(1,5);
    special_CT_b(2,6);
    special_CT_b(3,7);

    //1 基2
    zeta = zetas_n761[2];

    CT_a(0,2,zeta);
    CT_a(1,3,zeta);

    CT_b(0,2,zeta);
    CT_b(1,3,zeta);

    pse_CT_a(4,6);
    pse_CT_a(5,7);

    pse_CT_b(4,6);
    pse_CT_b(5,7);

    //3 基2
    zeta = zetas_n761[4];
    CT_a(0,1,zeta);
    CT_b(0,1,zeta);

    zeta = zetas_n761[5];
    CT_a(2,3,zeta);
    CT_b(2,3,zeta);

    zeta = zetas_n761[6];
    CT_a(4,5,zeta);
    CT_b(4,5,zeta);

    pse_CT_a(6,7);
    pse_CT_b(6,7);

    //放回共享内存
    ntta.coeffs[threadIdx.x] = a[0];
    ntta.coeffs[threadIdx.x + 192] = a[1];
    ntta.coeffs[threadIdx.x + 192 * 2] = a[2];
    ntta.coeffs[threadIdx.x + 192 * 3] = a[3];
    ntta.coeffs[threadIdx.x + 192 * 4] = a[4];
    ntta.coeffs[threadIdx.x + 192 * 5] = a[5];
    ntta.coeffs[threadIdx.x + 192 * 6] = a[6];
    ntta.coeffs[threadIdx.x + 192 * 7] = a[7];

    nttb.coeffs[threadIdx.x] = b[0];
    nttb.coeffs[threadIdx.x + 192] = b[1];
    nttb.coeffs[threadIdx.x + 192 * 2] = b[2];
    nttb.coeffs[threadIdx.x + 192 * 3] = b[3];
    nttb.coeffs[threadIdx.x + 192 * 4] = b[4];
    nttb.coeffs[threadIdx.x + 192 * 5] = b[5];
    nttb.coeffs[threadIdx.x + 192 * 6] = b[6];
    nttb.coeffs[threadIdx.x + 192 * 7] = b[7];

    __syncthreads();

    int group_num = threadIdx.x / 24;
    int group_idx = threadIdx.x % 24;
    int start = group_num * 192;

    a[0] = ntta.coeffs[start + group_idx];
    a[1] = ntta.coeffs[start + group_idx + 24];
    a[2] = ntta.coeffs[start + group_idx + 24 * 2];
    a[3] = ntta.coeffs[start + group_idx + 24 * 3];
    a[4] = ntta.coeffs[start + group_idx + 24 * 4];
    a[5] = ntta.coeffs[start + group_idx + 24 * 5];
    a[6] = ntta.coeffs[start + group_idx + 24 * 6];
    a[7] = ntta.coeffs[start + group_idx + 24 * 7];

    b[0] = nttb.coeffs[start + group_idx];
    b[1] = nttb.coeffs[start + group_idx + 24];
    b[2] = nttb.coeffs[start + group_idx + 24 * 2];
    b[3] = nttb.coeffs[start + group_idx + 24 * 3];
    b[4] = nttb.coeffs[start + group_idx + 24 * 4];
    b[5] = nttb.coeffs[start + group_idx + 24 * 5];
    b[6] = nttb.coeffs[start + group_idx + 24 * 6];
    b[7] = nttb.coeffs[start + group_idx + 24 * 7];


    //4 基2
    zeta = zetas_n761[8 + group_num];
    CT_a(0,4,zeta);
    CT_a(1,5,zeta);
    CT_a(2,6,zeta);
    CT_a(3,7,zeta);

    CT_b(0,4,zeta);
    CT_b(1,5,zeta);
    CT_b(2,6,zeta);
    CT_b(3,7,zeta);

    //5 基2
    zeta = zetas_n761[16 + group_num * 2];
    CT_a(0,2,zeta);
    CT_a(1,3,zeta);

    CT_b(0,2,zeta);
    CT_b(1,3,zeta);

    zeta = zetas_n761[16 + group_num * 2 + 1];
    CT_a(4,6,zeta);
    CT_a(5,7,zeta);

    CT_b(4,6,zeta);
    CT_b(5,7,zeta);

    //6 基2
    zeta = zetas_n761[32 + group_num * 4];
    CT_a(0,1,zeta);
    CT_b(0,1,zeta);

    zeta = zetas_n761[32 + group_num * 4 + 1];
    CT_a(2,3,zeta);
    CT_b(2,3,zeta);

    zeta = zetas_n761[32 + group_num * 4 + 2];
    CT_a(4,5,zeta);
    CT_b(4,5,zeta);

    zeta = zetas_n761[32 + group_num * 4 + 3];
    CT_a(6,7,zeta);
    CT_b(6,7,zeta);

    ntta.coeffs[start + group_idx] = a[0];
    ntta.coeffs[start + group_idx + 24] = a[1];
    ntta.coeffs[start + group_idx + 24 * 2] = a[2];
    ntta.coeffs[start + group_idx + 24 * 3] = a[3];
    ntta.coeffs[start + group_idx + 24 * 4] = a[4];
    ntta.coeffs[start + group_idx + 24 * 5] = a[5];
    ntta.coeffs[start + group_idx + 24 * 6] = a[6];
    ntta.coeffs[start + group_idx + 24 * 7] = a[7];

    nttb.coeffs[start + group_idx] = b[0];
    nttb.coeffs[start + group_idx + 24] = b[1];
    nttb.coeffs[start + group_idx + 24 * 2] = b[2];
    nttb.coeffs[start + group_idx + 24 * 3] = b[3];
    nttb.coeffs[start + group_idx + 24 * 4] = b[4];
    nttb.coeffs[start + group_idx + 24 * 5] = b[5];
    nttb.coeffs[start + group_idx + 24 * 6] = b[6];
    nttb.coeffs[start + group_idx + 24 * 7] = b[7];

    __syncthreads();

    group_num = threadIdx.x / 3;
    group_idx = threadIdx.x % 3;
    start = group_num * 24;

    a[0] = ntta.coeffs[start + group_idx];
    a[1] = ntta.coeffs[start + group_idx + 3];
    a[2] = ntta.coeffs[start + group_idx + 3 * 2];
    a[3] = ntta.coeffs[start + group_idx + 3 * 3];
    a[4] = ntta.coeffs[start + group_idx + 3 * 4];
    a[5] = ntta.coeffs[start + group_idx + 3 * 5];
    a[6] = ntta.coeffs[start + group_idx + 3 * 6];
    a[7] = ntta.coeffs[start + group_idx + 3 * 7];

    b[0] = nttb.coeffs[start + group_idx];
    b[1] = nttb.coeffs[start + group_idx + 3];
    b[2] = nttb.coeffs[start + group_idx + 3 * 2];
    b[3] = nttb.coeffs[start + group_idx + 3 * 3];
    b[4] = nttb.coeffs[start + group_idx + 3 * 4];
    b[5] = nttb.coeffs[start + group_idx + 3 * 5];
    b[6] = nttb.coeffs[start + group_idx + 3 * 6];
    b[7] = nttb.coeffs[start + group_idx + 3 * 7];

    //7 基2
    zeta = zetas_n761[64 + group_num];
    CT_a(0,4,zeta);
    CT_a(1,5,zeta);
    CT_a(2,6,zeta);
    CT_a(3,7,zeta);

    CT_b(0,4,zeta);
    CT_b(1,5,zeta);
    CT_b(2,6,zeta);
    CT_b(3,7,zeta);

    //8 基2
    zeta = zetas_n761[128 + group_num * 2];
    CT_a(0,2,zeta);
    CT_a(1,3,zeta);

    CT_b(0,2,zeta);
    CT_b(1,3,zeta);

    zeta = zetas_n761[128 + group_num * 2 + 1];
    CT_a(4,6,zeta);
    CT_a(5,7,zeta);

    CT_b(4,6,zeta);
    CT_b(5,7,zeta);

    //9 基2
    zeta = zetas_n761[256 + group_num * 4];
    CT_a(0,1,zeta);
    CT_b(0,1,zeta);

    zeta = zetas_n761[256 + group_num * 4 + 1];
    CT_a(2,3,zeta);
    CT_b(2,3,zeta);

    zeta = zetas_n761[256 + group_num * 4 + 2];
    CT_a(4,5,zeta);
    CT_b(4,5,zeta);

    zeta = zetas_n761[256 + group_num * 4 + 3];
    CT_a(6,7,zeta);
    CT_b(6,7,zeta);

    ntta.coeffs[start + group_idx] = a[0];
    ntta.coeffs[start + group_idx + 3] = a[1];
    ntta.coeffs[start + group_idx + 3 * 2] = a[2];
    ntta.coeffs[start + group_idx + 3 * 3] = a[3];
    ntta.coeffs[start + group_idx + 3 * 4] = a[4];
    ntta.coeffs[start + group_idx + 3 * 5] = a[5];
    ntta.coeffs[start + group_idx + 3 * 6] = a[6];
    ntta.coeffs[start + group_idx + 3 * 7] = a[7];

    nttb.coeffs[start + group_idx] = b[0];
    nttb.coeffs[start + group_idx + 3] = b[1];
    nttb.coeffs[start + group_idx + 3 * 2] = b[2];
    nttb.coeffs[start + group_idx + 3 * 3] = b[3];
    nttb.coeffs[start + group_idx + 3 * 4] = b[4];
    nttb.coeffs[start + group_idx + 3 * 5] = b[5];
    nttb.coeffs[start + group_idx + 3 * 6] = b[6];
    nttb.coeffs[start + group_idx + 3 * 7] = b[7];
    
    __syncthreads();
    
    //512 个 basemul

    //0-191个
    zeta = zetas_n761[256 + threadIdx.x / 2];
    if(threadIdx.x & 1) zeta = -zeta;
    basemul_n761_cuda(ntta.coeffs + 3 * threadIdx.x, ntta.coeffs + 3 * threadIdx.x,nttb.coeffs + 3 * threadIdx.x,zeta);

    //192 - 383
    zeta = zetas_n761[256 + threadIdx.x / 2 + 96];
    if(threadIdx.x & 1) zeta = -zeta;
    basemul_n761_cuda(ntta.coeffs + 3 * threadIdx.x + 576, ntta.coeffs + 3 * threadIdx.x + 576, nttb.coeffs + 3 * threadIdx.x + 576, zeta);

    if(threadIdx.x < 128){
        zeta = zetas_n761[256 + threadIdx.x / 2 + 192];
        if(threadIdx.x & 1) zeta = -zeta;
        basemul_n761_cuda(ntta.coeffs + 3 * threadIdx.x + 1152, ntta.coeffs + 3 * threadIdx.x + 1152, nttb.coeffs + 3 * threadIdx.x + 1152, zeta);
    }

    __syncthreads();


    //inv_ntt
    group_num = threadIdx.x / 3;
    group_idx = threadIdx.x % 3;
    start = group_num * 24;

    a[0] = ntta.coeffs[start + group_idx];
    a[1] = ntta.coeffs[start + group_idx + 3];
    a[2] = ntta.coeffs[start + group_idx + 3 * 2];
    a[3] = ntta.coeffs[start + group_idx + 3 * 3];
    a[4] = ntta.coeffs[start + group_idx + 3 * 4];
    a[5] = ntta.coeffs[start + group_idx + 3 * 5];
    a[6] = ntta.coeffs[start + group_idx + 3 * 6];
    a[7] = ntta.coeffs[start + group_idx + 3 * 7];


    //1 基2 256
    zeta = zetas_inv_n761[group_num * 4];
    GS_a(0,1,zeta);

    zeta = zetas_inv_n761[group_num * 4 + 1];
    GS_a(2,3,zeta);

    zeta = zetas_inv_n761[group_num * 4 + 2];
    GS_a(4,5,zeta);

    zeta = zetas_inv_n761[group_num * 4 + 3];
    GS_a(6,7,zeta);

    //2 基2 128
    zeta = zetas_inv_n761[256 + group_num * 2];
    GS_a(0,2,zeta);
    GS_a(1,3,zeta);

    zeta = zetas_inv_n761[256 + group_num * 2 + 1];
    GS_a(4,6,zeta);
    GS_a(5,7,zeta);

    //3 基2 64
    zeta = zetas_inv_n761[384 + group_num];
    GS_a(0,4,zeta);
    GS_a(1,5,zeta);
    GS_a(2,6,zeta);
    GS_a(3,7,zeta);

    ntta.coeffs[start + group_idx] = a[0];
    ntta.coeffs[start + group_idx + 3] = a[1];
    ntta.coeffs[start + group_idx + 3 * 2] = a[2];
    ntta.coeffs[start + group_idx + 3 * 3] = a[3];
    ntta.coeffs[start + group_idx + 3 * 4] = a[4];
    ntta.coeffs[start + group_idx + 3 * 5] = a[5];
    ntta.coeffs[start + group_idx + 3 * 6] = a[6];
    ntta.coeffs[start + group_idx + 3 * 7] = a[7];

    __syncthreads();

    group_num = threadIdx.x / 24;
    group_idx = threadIdx.x % 24;
    start = group_num * 192;

    a[0] = ntta.coeffs[start + group_idx];
    a[1] = ntta.coeffs[start + group_idx + 24];
    a[2] = ntta.coeffs[start + group_idx + 24 * 2];
    a[3] = ntta.coeffs[start + group_idx + 24 * 3];
    a[4] = ntta.coeffs[start + group_idx + 24 * 4];
    a[5] = ntta.coeffs[start + group_idx + 24 * 5];
    a[6] = ntta.coeffs[start + group_idx + 24 * 6];
    a[7] = ntta.coeffs[start + group_idx + 24 * 7];

    //4 基2 32
    zeta = zetas_inv_n761[448 + group_num * 4];
    GS_a(0,1,zeta);

    zeta = zetas_inv_n761[448 + group_num * 4 + 1];
    GS_a(2,3,zeta);

    zeta = zetas_inv_n761[448 + group_num * 4 + 2];
    GS_a(4,5,zeta);

    zeta = zetas_inv_n761[448 + group_num * 4 + 3];
    GS_a(6,7,zeta);

#pragma unroll
    for(int i=0;i<8;i++){
        pseudomersenne_reduce_single_n761_cuda(a[i]);
        pseudomersenne_reduce_single_n761_cuda(b[i]);
    }
    
    //5 基2 16
    zeta = zetas_inv_n761[480 + group_num * 2];
    GS_a(0,2,zeta);
    GS_a(1,3,zeta);

    zeta = zetas_inv_n761[480 + group_num * 2 + 1];
    GS_a(4,6,zeta);
    GS_a(5,7,zeta);

    //6 基2 8
    zeta = zetas_inv_n761[496 + group_num];
    GS_a(0,4,zeta);
    GS_a(1,5,zeta);
    GS_a(2,6,zeta);
    GS_a(3,7,zeta);

    ntta.coeffs[start + group_idx] = a[0];
    ntta.coeffs[start + group_idx + 24] = a[1];
    ntta.coeffs[start + group_idx + 24 * 2] = a[2];
    ntta.coeffs[start + group_idx + 24 * 3] = a[3];
    ntta.coeffs[start + group_idx + 24 * 4] = a[4];
    ntta.coeffs[start + group_idx + 24 * 5] = a[5];
    ntta.coeffs[start + group_idx + 24 * 6] = a[6];
    ntta.coeffs[start + group_idx + 24 * 7] = a[7];

    __syncthreads();
    

    a[0] = ntta.coeffs[threadIdx.x];
    a[1] = ntta.coeffs[threadIdx.x + 192];
    a[2] = ntta.coeffs[threadIdx.x + 192 * 2];
    a[3] = ntta.coeffs[threadIdx.x + 192 * 3];
    a[4] = ntta.coeffs[threadIdx.x + 192 * 4];
    a[5] = ntta.coeffs[threadIdx.x + 192 * 5];
    a[6] = ntta.coeffs[threadIdx.x + 192 * 6];
    a[7] = ntta.coeffs[threadIdx.x + 192 * 7];

    //7 基2 4
    zeta = zetas_inv_n761[504];
    GS_a(0,1,zeta);

    zeta = zetas_inv_n761[504 + 1];
    GS_a(2,3,zeta);

    zeta = zetas_inv_n761[504 + 2];
    GS_a(4,5,zeta);

    zeta = zetas_inv_n761[504 + 3];
    GS_a(6,7,zeta);

    //8 基2 2
    zeta = zetas_inv_n761[508];
    GS_a(0,2,zeta);
    GS_a(1,3,zeta);

    zeta = zetas_inv_n761[508 + 1];
    GS_a(4,6,zeta);
    GS_a(5,7,zeta);

    //9 基2 1
    zeta = zetas_inv_n761[510];
    GS_a(0,4,zeta);
    GS_a(1,5,zeta);
    GS_a(2,6,zeta);
    GS_a(3,7,zeta);

    ntta.coeffs[threadIdx.x] = a[0];
    ntta.coeffs[threadIdx.x + 192] = a[1];
    ntta.coeffs[threadIdx.x + 192 * 2] = a[2];
    ntta.coeffs[threadIdx.x + 192 * 3] = a[3];
    ntta.coeffs[threadIdx.x + 192 * 4] = a[4];
    ntta.coeffs[threadIdx.x + 192 * 5] = a[5];
    ntta.coeffs[threadIdx.x + 192 * 6] = a[6];
    ntta.coeffs[threadIdx.x + 192 * 7] = a[7];

    __syncthreads();
    
    //1 - 192
    array_c[blockIdx.x].coeffs[threadIdx.x + 1] = fq_freeze(montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (ntta.coeffs[threadIdx.x + 1] + ntta.coeffs[threadIdx.x + 1 + FPTRU_N - 1] + ntta.coeffs[threadIdx.x + 1 + FPTRU_N])));
    
    //193 - 384
    array_c[blockIdx.x].coeffs[threadIdx.x + 193] = fq_freeze(montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (ntta.coeffs[threadIdx.x + 193] + ntta.coeffs[threadIdx.x + 193 + FPTRU_N - 1] + ntta.coeffs[threadIdx.x + 193 + FPTRU_N])));

    //385 - 576
    array_c[blockIdx.x].coeffs[threadIdx.x + 385] = fq_freeze(montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (ntta.coeffs[threadIdx.x + 385] + ntta.coeffs[threadIdx.x + 385 + FPTRU_N - 1] + ntta.coeffs[threadIdx.x + 385 + FPTRU_N])));
    
    //0,760, 577-759
    if(threadIdx.x <= 183){
        if(threadIdx.x == 183){
            array_c[blockIdx.x].coeffs[0] = fq_freeze(montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (ntta.coeffs[0] + ntta.coeffs[FPTRU_N])));
            array_c[blockIdx.x].coeffs[FPTRU_N - 1] = fq_freeze(montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (ntta.coeffs[FPTRU_N - 1] + ntta.coeffs[2 * FPTRU_N - 2])));
        }
        else{
            array_c[blockIdx.x].coeffs[threadIdx.x + 577] = fq_freeze(montgomery_reduce_n761_cuda((int64_t)FACTOR_JAOHADAH * (ntta.coeffs[threadIdx.x + 577] + ntta.coeffs[threadIdx.x + 577 + FPTRU_N - 1] + ntta.coeffs[threadIdx.x + 577 + FPTRU_N])));
        }
    }
}

__device__ void d_poly_sample_and_double(nttpoly_n761 & nttb,const uint8_t buf[FPTRU_COIN_BYTES / 2],uint16_t is_double,uint16_t is_add1){
    unsigned int i, j;
    uint32_t t, d;
    int16_t a, b;

    j = threadIdx.x % 8;

#pragma unroll
    for(i = threadIdx.x / 8; i < 72; i+=24){
        t = load32_littleendian(buf + 4 * i);
        d = t & 0x55555555;
        d += (t >> 1) & 0x55555555;

        a = (d >> (4 * j + 0)) & 0x3;
        b = (d >> (4 * j + 2)) & 0x3;
        
        a = a - b +  is_double * (a - b) + is_add1 * (!(8 * i + j));
        nttb.coeffs[8 * i + j] = a;
    }

    if(threadIdx.x < 185){//761 = 192 * 3 + 185
        if(threadIdx.x == 184){
            i = 95;
            t = buf[4 * i];
            a = 0;
            b = 0;
            //直接基于t生成b
            for (j = 0; j < 2; j++){
                a += (t >> j) & 0x1;
            }
            for (j = 2; j < 4; j++){
                b += (t >> j) & 0x1;
            }
            a = (int16_t)(a - b) + is_double * (int16_t)(a - b) + is_add1 * (!(8 * i + 0));
            nttb.coeffs[8 * i + 0] = a;
        }
        else{
            t = load32_littleendian(buf + 4 * i);
            d = t & 0x55555555;
            d += (t >> 1) & 0x55555555;

            a = (d >> (4 * j + 0)) & 0x3;
            b = (d >> (4 * j + 2)) & 0x3;
            a = (int16_t)(a - b +  is_double * (a - b) + is_add1 * (!(8 * i + j)));//int16->int32
            nttb.coeffs[8 * i + j] = a;
        }
    }
}

__global__ void poly_poly_sample_and_double_mul_761_q1(unsigned char * array_coins,uint16_t is_double,uint16_t is_add1,int interval,poly * array_c,poly * array_a){
    __shared__ nttpoly_n761 ntta;
    __shared__ nttpoly_n761 nttb;


#pragma unroll
    for(int i=0;i<8;i++){ //init is important
        ntta.coeffs[i * 192 + threadIdx.x] = 0;
        nttb.coeffs[i * 192 + threadIdx.x] = 0;
    }
    __syncthreads();
    d_poly_sample_and_double(nttb,&array_coins[blockIdx.x * interval],is_double,is_add1);
    __syncthreads();
    d_poly_mul_q1_761(ntta, nttb, array_a, array_c);


}