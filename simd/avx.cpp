#include <immintrin.h>
#include <random>
#include <sys/time.h>
#include <iostream>
#include <string.h>
#include "vcl/vectorclass.h"

using namespace std;

#define len (16 << 22)
#define soltDiv 2

uint32_t rr;
uint32_t hash1[len];
uint32_t hash2[len];
uint32_t alpha[len];
uint32_t beta1[len];

uint32_t div_slotDiv[16] = {soltDiv, soltDiv, soltDiv, soltDiv, soltDiv, soltDiv, soltDiv, soltDiv,
                            soltDiv, soltDiv, soltDiv, soltDiv, soltDiv, soltDiv, soltDiv, soltDiv};

vector<uint32_t> execp_vals;

void InitArray()
{
    std::mt19937_64 generator(2);
    std::uniform_int_distribution<uint32_t> distribution(
        numeric_limits<uint32_t>::min(),
        numeric_limits<uint32_t>::max());

    for (int i = 0; i < len; i++)
    {
        hash1[i] = (uint32_t)distribution(generator);
        hash2[i] = (uint32_t)distribution(generator);
        alpha[i] = (uint32_t)distribution(generator);
        beta1[i] = (uint32_t)distribution(generator);
    }
}

void ExpressionNormal()
{
    struct timeval s2, e2;
    gettimeofday(&s2, NULL);

    for (int i = 0; i < len; i++)
    {
        execp_vals.emplace_back((hash1[i] + alpha[i] * hash2[i] + beta1[i]) % soltDiv);
    }

    gettimeofday(&e2, NULL);

    auto delta = (e2.tv_sec - s2.tv_sec) * 1000000 + (e2.tv_usec - s2.tv_usec);
    cout << "Normal PH compute duration time us: " << delta << endl;
}

bool verify(vector<uint32_t> &avx_vals)
{
    if (execp_vals.size() != avx_vals.size())
    {
        return false;
    }
    for (int i = 0; i < execp_vals.size(); i++)
    {
        // cout << execp_vals[i] << " " << avx_vals[i] << endl;

        if (execp_vals[i] != avx_vals[i])
        {
            return false;
        }
    }
    return true;
}

void ExpressionAvx()
{
    struct timeval s2, e2;
    gettimeofday(&s2, NULL);

    __m512i slot_512;
    __m512i tmp_512;
    vector<uint32_t> avx_vals;

    uint32_t result[16];

    for (int i = 0; i < len; i += 16)
    {
        slot_512 = _mm512_mullo_epi32(*((__m512i *)&alpha[i]), *((__m512i *)&hash2[i]));
        slot_512 = _mm512_add_epi32(slot_512, *((__m512i *)&hash1[i]));
        slot_512 = _mm512_add_epi32(slot_512, *((__m512i *)&beta1[i]));

        memcpy((void *)&result[0], (void *)&slot_512, 64);

        for (auto r : result)
        {
            avx_vals.emplace_back(r % soltDiv);
        }
    }

    gettimeofday(&e2, NULL);

    auto delta = (e2.tv_sec - s2.tv_sec) * 1000000 + (e2.tv_usec - s2.tv_usec);
    cout << "AVX PH compute duration time us: " << delta << endl;

    if (verify(avx_vals))
    {
        cout << "verify successfully!!" << endl;
    }
    else
    {
        cout << "verify fail!!" << endl;
    }
}

void ExpressionVCL()
{
    struct timeval s2, e2;
    gettimeofday(&s2, NULL);

    Vec16ui slot_16ui;
    Vec16ui slot_16ui_alpha;
    Vec16ui slot_16ui_hash1;
    Vec16ui slot_16ui_hash2;
    Vec16ui slot_16ui_beta1;
    Vec16ui tmp;
    vector<uint32_t> avx_vals;
    uint32_t result[16];

    for (int i = 0; i < len; i += 16)
    {
        slot_16ui_alpha.load(&alpha[i]);
        slot_16ui_hash1.load(&hash1[i]);
        slot_16ui_hash2.load(&hash2[i]);
        slot_16ui_beta1.load(&beta1[i]);

        slot_16ui = slot_16ui_alpha * slot_16ui_hash2;
        slot_16ui += slot_16ui_hash1;
        slot_16ui += slot_16ui_beta1;

        // mod: c = a / b;  r = a - c * b;
        tmp = slot_16ui / soltDiv;
        tmp *= soltDiv;
        slot_16ui -= tmp;

        slot_16ui.store(result);

        for (auto r : result)
        {
            avx_vals.emplace_back(r);
        }
    }

    gettimeofday(&e2, NULL);

    auto delta = (e2.tv_sec - s2.tv_sec) * 1000000 + (e2.tv_usec - s2.tv_usec);
    cout << "VCL PH compute duration time us: " << delta << endl;

    if (verify(avx_vals))
    {
        cout << "verify successfully!!" << endl;
    }
    else
    {
        cout << "verify fail!!" << endl;
    }
}

void ExpressionVCL2()
{
    struct timeval s2, e2;
    gettimeofday(&s2, NULL);

    __m512i slot_512;
    Vec16ui slot_16ui;
    vector<uint32_t> avx_vals;

    uint32_t result[16];

    for (int i = 0; i < len; i += 16)
    {
        slot_512 = _mm512_mullo_epi32(*((__m512i *)&alpha[i]), *((__m512i *)&hash2[i]));
        slot_512 = _mm512_add_epi32(slot_512, *((__m512i *)&hash1[i]));
        slot_512 = _mm512_add_epi32(slot_512, *((__m512i *)&beta1[i]));

        slot_16ui.load(&slot_512);

        Vec16ui tmp = slot_16ui / soltDiv;
        tmp *= soltDiv;
        slot_16ui -= tmp;

        slot_16ui.store(result);

        for (auto r : result)
        {
            avx_vals.emplace_back(r);
        }
    }

    gettimeofday(&e2, NULL);

    auto delta = (e2.tv_sec - s2.tv_sec) * 1000000 + (e2.tv_usec - s2.tv_usec);
    cout << "VCL2 PH compute duration time us: " << delta << endl;

    if (verify(avx_vals))
    {
        cout << "verify successfully!!" << endl;
    }
    else
    {
        cout << "verify fail!!" << endl;
    }
}

int main()
{
    cout << "test len of nums :" << len << endl;
    InitArray();

    cout << "init data finish!!" << endl;

    ExpressionNormal();
    cout << endl;

    ExpressionAvx();
    cout << endl;
    ExpressionVCL();
    cout << endl;
    ExpressionVCL2();
    cout << endl;

    return 0;
}
