#include <immintrin.h>
#include "/usr/include/mkl/mkl.h"
#include <vector>
#include <cmath>
#include <inttypes.h>
#include <stdint.h>
#include <iostream>
#include <cstdint>
#include <random>
#include <numeric>
#include <limits>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
// #include <mmintrin.h>  //MMX
// #include <xmmintrin.h> //SSE(include mmintrin.h)
// #include <emmintrin.h> //SSE2(include xmmintrin.h)
// #include <pmmintrin.h> //SSE3(include emmintrin.h)
// #include <tmmintrin.h> //SSSE3(include pmmintrin.h)
// #include <smmintrin.h> //SSE4.1(include tmmintrin.h)
// #include <nmmintrin.h> //SSE4.2(include smmintrin.h)
// #include <wmmintrin.h> //AES(include nmmintrin.h)
#include <immintrin.h> //AVX(include wmmintrin.h)
#include <x86intrin.h>
#include<dlfcn.h>

using namespace std;
#define size 90000000
constexpr static uint32_t slotCnt = 1;
uint32_t slot[size];
uint32_t rr;
uint32_t hash1[size];
uint32_t hash2[size];
uint32_t alpha[size];
uint32_t beta[size];
uint32_t div_slotcnt[16] = {slotCnt, slotCnt, slotCnt, slotCnt, slotCnt, slotCnt, slotCnt, slotCnt,
                            slotCnt, slotCnt, slotCnt, slotCnt, slotCnt, slotCnt, slotCnt, slotCnt};
uint32_t result[16];
//  uint64_t r1[size];
int main()
{
    std::mt19937_64 generator(std::random_device{}());
    std::uniform_int_distribution<uint32_t> distribution(
        numeric_limits<uint32_t>::min(),
        numeric_limits<uint32_t>::max());

    for (int i = 0; i < size; i++)
    {
        hash1[i] = (uint32_t)distribution(generator);
        hash2[i] = (uint32_t)distribution(generator);
        alpha[i] = (uint32_t)distribution(generator);
        beta[i] = (uint32_t)distribution(generator);
    }
    struct timeval s2, e2;
    // __m256i slot_256;
    vector<uint32_t> v;
    v.resize(16);
    gettimeofday(&s2, NULL);
#if 0
    for(int i = 0; i < size; i++) {
         //slot[i] = (hash1[i] + alpha[i] * hash2[i] + beta[i]) % slotCnt; //36.7ms@10m keys, 317ms@90m keys
        // slot = alpha[i] * hash2[i];
        //slot = slot + hash1[i];
        //slot = slot + beta[i];
        //r1[i] = (uint64_t)slot;
        rr = hash1[i] + alpha[i] * hash2[i] + beta[i];
    }
#else
    __m512i slot_512;
    /*
    __m512i _mm512_mullo_epi32 (__m512i a, __m512i b)
    __m512i _mm512_add_epi32 (__m512i a, __m512i b)
    */
    for (int i = 0; i < size; i += 16)
    {
        uint32_t a;
        //%value = fadd u32 4, %a; //yields i32:result = 4 + %var
        slot_512 = _mm512_mullo_epi32(*((__m512i *)&alpha[i]), *((__m512i *)&hash2[i]));
        slot_512 = _mm512_add_epi64(slot_512, *((__m512i *)&hash1[i]));
        slot_512 = _mm512_add_epi64(slot_512, *((__m512i *)&beta[i]));        // 16.1ms@10m keys, 148ms@90m keys

        // slot_512 = _mm512_rem_epi32(slot_512, *((__m512i *)&div_slotcnt[0])); // SVML??

        memcpy((void *)&result[0], (void *)&slot_512, 64);

        result[0] = *((uint32_t *)&slot_512 + 0) % slotCnt;
        result[1] = *((uint32_t *)&slot_512 + 1) % slotCnt;
        result[2] = *((uint32_t *)&slot_512 + 2) % slotCnt;
        result[3] = *((uint32_t *)&slot_512 + 3) % slotCnt;
        result[4] = *((uint32_t *)&slot_512 + 4) % slotCnt;
        result[5] = *((uint32_t *)&slot_512 + 5) % slotCnt;
        result[6] = *((uint32_t *)&slot_512 + 6) % slotCnt;
        result[7] = *((uint32_t *)&slot_512 + 7) % slotCnt;
        result[8] = *((uint32_t *)&slot_512 + 8) % slotCnt;
        result[9] = *((uint32_t *)&slot_512 + 9) % slotCnt;
        result[10] = *((uint32_t *)&slot_512 + 10) % slotCnt;
        result[11] = *((uint32_t *)&slot_512 + 11) % slotCnt;
        result[12] = *((uint32_t *)&slot_512 + 12) % slotCnt;
        result[13] = *((uint32_t *)&slot_512 + 13) % slotCnt;
        result[14] = *((uint32_t *)&slot_512 + 14) % slotCnt;
        result[15] = *((uint32_t *)&slot_512 + 15) % slotCnt;
        //

        // mode
        //__m256i* a = (__m256i*)&slot_512;
        //__m256i remainder;
        // uint32_t remainder[8];
        //_mm256_udivrem_epi32(&remainder, a, *((__m256i*)&div_slotcnt[0]));

        //  uint64_t* r = (uint64_t*)&slot_256;
        /*
         for (int j = 0;j < 4;j++) {
             if (r[ j ] != r1[i + j]) {
                 cout<<j << " *****is not equal to***** "<<i+j<<endl;
                 cout<<hex<<r[j] << " != "<<r1[i + j]<<endl;
                  cout<<hex<<r[j+1] << " != "<<r1[i + j+1]<<endl;
                  cout<<hex<<r[j+2] << " != "<<r1[i + j+2]<<endl;
                 return 0;
             } else {
                 cout<<hex<<r[j]<<endl;
             }
         }
         */
    }
#endif
    gettimeofday(&e2, NULL);
    auto delta = (e2.tv_sec - s2.tv_sec) * 1000000 + (e2.tv_usec - s2.tv_usec);
    cout << "PH compute duration time us: " << delta << endl;
    cout << slot << endl;

    return 0;
}

