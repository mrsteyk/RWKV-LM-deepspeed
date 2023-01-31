#include <stdio.h>
#include <assert.h>

#include <cuda_fp16.h>

// TODO(mrsteyk): is rounding towards zero the best mode?
#define MIN_VALUE __float2half_rz(-1e38)

#define F __half

__global__ void kernel_forward(const int B, const int T, const int C,
                               const F *__restrict__ const _w, const F *__restrict__ const _u, const F *__restrict__ const _k, const F *__restrict__ const _v,
                               F *__restrict__ const _y) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    F *__restrict__ const y = _y + _offset;

    F p = __int2half_rz(0), q = __int2half_rz(0), o = MIN_VALUE;
    // p and q are running sums divided by exp(o) (to avoid overflows)
    for (int i = 0; i < T; i++) {
        const int ii = i * C;

        F no = __hmax(o, __hadd(u, k[ii]));
        F A = hexp(__hsub(o, no));
        F B = hexp(__hsub(__hadd(u, k[ii]), no));
        y[ii] = __hdiv(
            __hadd(__hmul(A, p), __hmul(B, v[ii])),
            __hfma(A, q, B)
        );

        no = __hmax(__hadd(w,  o), k[ii]);
        A = hexp(__hsub(__hadd(w, o), no));
        B = hexp(__hsub(k[ii], no));
        p = __hadd(__hmul(A, p), __hmul(B, v[ii]));
        q = __hfma(A, q, B);
        o = no;
    }
}

__global__ void kernel_backward(const int B, const int T, const int C,
                                const F *__restrict__ const _w, const F *__restrict__ const _u, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _gy,
                                F *__restrict__ const _gw, F *__restrict__ const _gu, F *__restrict__ const _gk, F *__restrict__ const _gv) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    const F *__restrict__ const gy = _gy + _offset;

    F *__restrict__ const gk = _gk + _offset;
    F *__restrict__ const gv = _gv + _offset;

    F y[Tmax], z[Tmax], zexp[Tmax];

    F gw = __int2half_rz(0), gu = __int2half_rz(0);
    F p = __int2half_rz(0), q = __int2half_rz(0);
    F dpdw = __int2half_rz(0), dqdw = __int2half_rz(0);
    F o = MIN_VALUE;
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        F no = __hmax(o, __hadd(k[ii], u));
        F A = hexp(__hsub(o, no));
        F B = hexp(__hsub(__hadd(k[ii], u), no));

        F num = __hadd(__hmul(A, p), __hmul(B, v[ii]));
        F iden = __hdiv(__int2half_rz(1), __hfma(A, q, B));

        y[i] = __hmul(num, iden);
        z[i] = iden;
        zexp[i] = __hsub(__hadd(k[ii], u), no);

        gw = __hadd(gw, __hmul(__hmul(__hmul(gy[ii], __hsub(dpdw, __hmul(dqdw, y[i]))), iden), A));
        gu = __hadd(gu, __hmul(__hmul(__hmul(gy[ii], __hsub(v[ii], y[i])), B), iden));

        no = __hmax(__hadd(w,  o), k[ii]);
        A = hexp(__hsub(__hadd(w, o), no));
        B = hexp(__hsub(k[ii], no));
        dpdw = __hmul(A, __hadd(p, dpdw));
        dqdw = __hmul(A, __hadd(q, dqdw));
        p = __hadd(__hmul(A, p), __hmul(B, v[ii]));
        q = __hfma(A, q, B);
        o = no;
    }

    F gp = __int2half_rz(0), gq = __int2half_rz(0);
    o = MIN_VALUE;
    for (int i = T - 1; i >= 0; i--) {
        const int ii = i * C;
        F A = __hmul(__hmul(gy[ii], z[i]), hexp(zexp[i]));
        F B = hexp(__hadd(k[ii], o));
        gk[ii] = __hfma(A, __hsub(v[ii], y[i]),
            __hmul(B, __hfma(gp, v[ii], gq))
        ); // A * (v[ii] - y[i]) + B * (gp * v[ii] + gq);
        gv[ii] = __hfma(B, gp, A); // A + B * gp;

        F no = __hmax(__hadd(w, o), __hsub(__hsub(zexp[i], k[ii]), u));
        A = hexp(__hsub(__hadd(w, o), no));
        B = __hmul(__hmul(gy[ii], z[i]), hexp(__hsub(__hsub(__hsub(zexp[i], k[ii]), u), no)));
        gp = __hfma(A, gp, B);
        gq = __hsub(__hmul(A, gq), __hmul(B, y[i]));
        o = no;
    }

    // Multiply by w because the w -> -exp(w) preprocessing is halfway in the backwards pass, even though it's not in the forward pass
    const int _offsetBC = _b * C + _c;
    // _gw[_offsetBC] += gw * _w[_c];
    // _gu[_offsetBC] += gu;
    _gw[_offsetBC] = __hfma(gw, _w[_c], _gw[_offsetBC]);
    _gu[_offsetBC] = __hadd(_gu[_offsetBC], gu);
}

void cuda_forward_half(int B, int T, int C, __half *w, __half *u, __half *k, __half *v, __half *y) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y);
}

void cuda_backward_half(int B, int T, int C, __half *w, __half *u, __half *k, __half *v, __half *gy, __half *gw, __half *gu, __half *gk, __half *gv) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, gy, gw, gu, gk, gv);
}
