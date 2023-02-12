#include <stdio.h>
#include <assert.h>

#define MIN_VALUE (-1e38)

// https://discord.com/channels/992359628979568762/992363236370436136/1055843600484810802
// user snowflake - 468093332535640064

// GPUs don't have a calling convention, every single function call is specific to that exact function, so specific argument ordering isn't required

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C,
                               const F *__restrict__ const _w, const F *__restrict__ const _u, const F *__restrict__ const _k, const F *__restrict__ const _v,
                               F *__restrict__ const _y,
                               const F *__restrict__ const last_state, F *__restrict__ const new_state) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    const int state_offset = (_b * C + _c)*3;

    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    F *__restrict__ const y = _y + _offset;

    F p = 0, q = 0, o = MIN_VALUE;
    if(last_state != NULL) {
        p = last_state[state_offset+0];
        q = last_state[state_offset+1];
        o = last_state[state_offset+2];
    }

    // p and q are running sums divided by exp(o) (to avoid overflows)
    for (int i = 0; i < T; i++) {
        const int ii = i * C;

        F no = max(o, u + k[ii]);
        F A = exp(o - no);
        F B = exp(u + k[ii] - no);
        y[ii] = (A * p + B * v[ii]) / (A * q + B);

        no = max(w + o, k[ii]);
        A = exp(w + o - no);
        B = exp(k[ii] - no);
        p = A * p + B * v[ii];
        q = A * q + B;
        o = no;
    }

    if (new_state != NULL) {
        new_state[state_offset+0] = p;
        new_state[state_offset+1] = q;
        new_state[state_offset+2] = o;
    }
}

template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C,
                                const F *__restrict__ const _w, const F *__restrict__ const _u, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _gy,
                                F *__restrict__ const _gw, F *__restrict__ const _gu, F *__restrict__ const _gk, F *__restrict__ const _gv,
                                const F *__restrict__ const last_state, const F *__restrict__ const gnew_state, F *__restrict__ const glast_state) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;

    const int state_offset  = (_b * C + _c)*3;

    F u = _u[_c];
    F w = _w[_c];
    const F *__restrict__ const k = _k + _offset;
    const F *__restrict__ const v = _v + _offset;
    const F *__restrict__ const gy = _gy + _offset;

    F *__restrict__ const gk = _gk + _offset;
    F *__restrict__ const gv = _gv + _offset;

    F y[Tmax], z[Tmax], zexp[Tmax];

    F gw = 0, gu = 0;
    F p = 0, q = 0;
    F dpdw = 0, dqdw = 0;
    F o = MIN_VALUE;

    if (last_state != NULL) {
        p = last_state[state_offset+0];
        q = last_state[state_offset+1];
        o = last_state[state_offset+2];
    }

    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        F no = max(o, k[ii] + u);
        F A = exp(o - no);
        F B = exp(k[ii] + u - no);

        F num = A * p + B * v[ii];
        F iden = 1 / (A * q + B);

        y[i] = num * iden;
        z[i] = iden;
        zexp[i] = k[ii] + u - no;

        gw += gy[ii] * (dpdw - dqdw * y[i]) * iden * A;
        gu += gy[ii] * (v[ii] - y[i]) * B * iden;

        no = max(w + o, k[ii]);
        A = exp(w + o - no);
        B = exp(k[ii] - no);
        dpdw = A * (p + dpdw);
        dqdw = A * (q + dqdw);
        p = A * p + B * v[ii];
        q = A * q + B;
        o = no;
    }

    F gp = 0, gq = 0;
    //o = MIN_VALUE;

    if (gnew_state != NULL) {
        gp = gnew_state[state_offset+0];
        gq = gnew_state[state_offset+1];
        F go = gnew_state[state_offset+2];
        if (gp == 0 && gq == 0) go = MIN_VALUE;
        gw += (gp * dpdw + gq * dqdw) * exp(o+go);
        o = go;
    } else {
        o = MIN_VALUE;
    }

    for (int i = T - 1; i >= 0; i--) {
        const int ii = i * C;
        F A = gy[ii] * z[i] * exp(zexp[i]);
        F B = exp(k[ii] + o);
        gk[ii] = A * (v[ii] - y[i]) + B * (gp * v[ii] + gq);
        gv[ii] = A + B * gp;

        F no = max(w + o, zexp[i] - k[ii] - u);
        A = exp(w + o - no);
        B = gy[ii] * z[i] * exp(zexp[i] - k[ii] - u - no);
        gp = A * gp + B;
        gq = A * gq - B * y[i];
        o = no;
    }

    // glast_state[2] is not the gradient w.r.t of last_state[2]
    // o (index 2) in last_state is just an exponent for p and q
    // so there are really only 2 elements to differentiate on
    // Similary go (glast_state index 2) is just an exponent for gp and gq
    if (glast_state != NULL) {
        glast_state[state_offset+0] = gp;
        glast_state[state_offset+1] = gq;
        glast_state[state_offset+2] = o;
    }

    // Multiply by w because the w -> -exp(w) preprocessing is halfway in the backwards pass, even though it's not in the forward pass
    const int _offsetBC = _b * C + _c;
    _gw[_offsetBC] += gw * _w[_c];
    _gu[_offsetBC] += gu;
}

void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y, float* last_state, float* new_state) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y, last_state, new_state);
}

void cuda_backward(int B, int T, int C, float *w, float *u, float *k, float *v, float *gy, float *gw, float *gu, float *gk, float *gv, float *last_state, float *gnew_state, float *glast_state) {
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, gy, gw, gu, gk, gv, last_state, gnew_state, glast_state);
}
