#include <torch/extension.h>
#include <cuda_fp16.h>

void cuda_forward_half(int B, int T, int C, __half *w, __half *u, __half *k, __half *v, __half *y);
void cuda_backward_half(int B, int T, int C, __half *w, __half *u, __half *k, __half *v, __half *gy, __half *gw, __half *gu, __half *gk, __half *gv);

void forward(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
    cuda_forward_half(B, T, C, (__half*)w.data_ptr<c10::Half>(), (__half*)u.data_ptr<c10::Half>(), (__half*)k.data_ptr<c10::Half>(), (__half*)v.data_ptr<c10::Half>(), (__half*)y.data_ptr<c10::Half>());
}
void backward(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {
    cuda_backward_half(B, T, C, (__half*)w.data_ptr<c10::Half>(), (__half*)u.data_ptr<c10::Half>(), (__half*)k.data_ptr<c10::Half>(), (__half*)v.data_ptr<c10::Half>(), (__half*)gy.data_ptr<c10::Half>(), (__half*)gw.data_ptr<c10::Half>(), (__half*)gu.data_ptr<c10::Half>(), (__half*)gk.data_ptr<c10::Half>(), (__half*)gv.data_ptr<c10::Half>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "wkv fp16 forward");
    m.def("backward", &backward, "wkv fp16 backward");
}

TORCH_LIBRARY(wkv, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}
