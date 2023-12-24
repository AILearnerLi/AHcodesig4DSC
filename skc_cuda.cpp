#include <torch/extension.h>
#include <vector>

// CHECK_CUDA������������Ƿ���gpu�ϣ� CHECK_CONTIGUOUS������������Ƿ��������д洢�� CHECK_INPUT����ͬʱ���������������
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> skc_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor output,
    int stride
);

std::vector<torch::Tensor> skc_cuda_backward(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor d_input,
    torch::Tensor d_weights,
    int stride
);

std::vector<torch::Tensor> skc_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor output,
    int stride
) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(output);
  return skc_cuda_forward(input, weights, output, stride);
}

std::vector<torch::Tensor> skc_backward(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor d_input,
    torch::Tensor d_weights,
    int stride) {

  CHECK_INPUT(d_output);
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(d_input);
  CHECK_INPUT(d_weights);

  return skc_cuda_backward(d_output, input, weights, d_input, d_weights, stride);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &skc_forward, "RPW forward (CUDA)");
  m.def("backward", &skc_backward, "RPW backward (CUDA)");
}
