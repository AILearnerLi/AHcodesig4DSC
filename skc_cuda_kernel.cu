#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <vector>


#define input_centric_backward      // input-centric backward computing.
#define enforce_atomic              // for input-centric backward correctness guarantee

template <typename scalar_t>
__global__ void skc_cuda_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights, 
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> new_tensor,
    int batch_size,
    int input_channel,
    int input_height,
    int input_width,
    int output_channel,
    int input_unit_dim,
    int stride
);

template <typename scalar_t>
__global__ void skc_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_weights,
    int batch_size,
    int input_channel,
    int height,
    int width,
    int output_channel,
    int input_unit_dim,
    int stride
);
////////////////////////////////////////////
// foward pass
////////////////////////////////////////////
std::vector<torch::Tensor> skc_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor new_tensor,
    int stirde
) {

    // input: batch_size * input_channel * input_width * input_height.
    const int batch_size = input.size(0);
    const int input_channel = input.size(1);
    const int input_height = input.size(2);
    const int input_width = input.size(3);

    // output: batch_size * opt_channel * input_width * input_height.
    const int output_channel = weights.size(0);

    // weight: output_channel * input_units_dim * 1.
    const int input_unit_dim = weights.size(1); 

    // new tensor for output.
    const int threads = 1024;   //һ��block��1024���߳�
    const int blocks = (batch_size * output_channel * input_width * input_height + threads - 1) / threads;   //����һ��grid��Ҫ���ٸ�block,������ȡ��
	
	/*AT_DISPATCH_FLOATING_TYPES����꣬ʵ���˶�̬�ַ����ƣ�dynamic dispatch����������������ʱ����������������ֵ���ͣ�ȥ����֮ǰCUDA kernelģ�麯����Ҫʵ����Ϊ���ֺ�����
	�������Ҫ�����������ͣ������������ַ�����Ϣ��һ��lambda��������������������src.scalar_type()��ȡ��lambda�������Ǻ������CUDA kernel�������֡�
	�������ڴ�ATen Tensor�л�ȡĳһ��������ָ����õ���<<< >>>��һд������kernel��*/
    AT_DISPATCH_FLOATING_TYPES(input.type(), "skc_forward_cuda", ([&] {
                                skc_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                                    input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                    weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    new_tensor.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                    batch_size,
                                    input_channel,
                                    input_height,
                                    input_width,
                                    output_channel,
                                    input_unit_dim,
                                    stirde
                                );
                            }));
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    return {new_tensor};
}
//The argument torch::RestrictPtrTraits indicates that the __restrict__ keyword must be used.
template <typename scalar_t>
__global__ void skc_cuda_forward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights, 
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> new_tensor,
    int batch_size,
    int input_channel,
    int input_height,
    int input_width,
    int output_channel,
    int input_unit_dim,
    int stride
) {
  const int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int g_dim = batch_size * output_channel * input_width * input_height;
  const int item_size_dim = output_channel * input_height * input_width;
  const int feature_map_dim = input_height * input_width;

  const int item_idx = g_idx / item_size_dim;
  const int item_channel_idx =  (g_idx - item_idx * item_size_dim) / feature_map_dim;
  const int item_feat_y_idx = (g_idx - item_idx * item_size_dim - item_channel_idx * feature_map_dim) / input_width;
  const int item_feat_x_idx = (g_idx - item_idx * item_size_dim - item_channel_idx * feature_map_dim) % input_width;
  const int b = item_idx;
  const int c = item_channel_idx;
  const int y = item_feat_y_idx;
  const int x = item_feat_x_idx;
  
  const int input_c_start = __float2int_rd(item_channel_idx * stride) % input_channel;
  const int input_c_end = (input_c_start + input_unit_dim) % input_channel;
  const int input_x = x;
  const int input_y = y;

//   new_tensor[b][c][y][x] = 5;
//   printf("gid: %d, total thread: %d\n", g_idx, g_dim);
  if (g_idx < g_dim) {
        float tmp = 0;
        // printf("input_c_start, %d, input_c_end, %d\n", input_c_start, input_c_end);
        if (input_c_start < input_c_end)
            for(int c_input_d = input_c_start; c_input_d < input_c_end; c_input_d++){
                tmp += input[b][c_input_d][input_y][input_x] * weights[c][c_input_d - input_c_start];
            }
        else
        {
            for(int c_input_d = input_c_start; c_input_d < input_channel; c_input_d++){
                tmp += input[b][c_input_d][input_y][input_x] * weights[c][c_input_d - input_c_start];
            }
            for(int c_input_d = 0; c_input_d < input_c_end; c_input_d++){
                tmp += input[b][c_input_d][input_y][input_x] * weights[c][c_input_d + input_channel - input_c_start];
            } 
        }
        new_tensor[b][c][y][x] = tmp;
        // printf("gid: %d, new tensor (%d, %d, %d, %d) --- %f\n", g_idx, b, c, y, x, new_tensor[0][0][0][0]);
  }
}

#ifdef input_centric_backward
std::vector<torch::Tensor> skc_cuda_backward(
        torch::Tensor d_output,
        torch::Tensor input,
        torch::Tensor weights,
        torch::Tensor d_input,
        torch::Tensor d_weights,
        int stride
    ) {
    
        // input: batch_size * input_channel * input_width * input_height.
        const int batch_size = d_output.size(0);
        const int output_channel = d_output.size(1);
        const int height = d_output.size(2);
        const int width = d_output.size(3);
    
        // output: batch_size * opt_channel * input_width * input_height.
        const int input_channel = d_input.size(1);
    
        // weight: output_channel * input_units_dim * 1.
        const int input_unit_dim = weights.size(1); 
    
        const int threads = 1024;
        const int blocks = (batch_size * input_channel * width * height + threads - 1) / threads; 
    
        AT_DISPATCH_FLOATING_TYPES(d_output.type(), "skc_backward_cuda", ([&] {
                                skc_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                                        d_output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                        input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                        weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                        d_input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                                        d_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                        batch_size,
                                        input_channel,
                                        height,
                                        width,
                                        output_channel,
                                        input_unit_dim,
                                        stride
                                    );
                                }));
        // check for error
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
        return {d_input, d_weights};
    }
    
template <typename scalar_t>
__global__ void skc_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_weights,
    int batch_size,
    int input_channel,
    int height,
    int width,
    int output_channel,
    int input_unit_dim,
    int stride
) {
    const int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int g_dim = batch_size * input_channel * width * height;

//   printf("gid: %d, total thread: %d\n", g_idx, g_dim);
    if (g_idx < g_dim) {
        const int item_size_dim = input_channel * height * width;
        const int feature_map_dim = height * width;
        const int item_idx = g_idx / item_size_dim;
        const int item_channel_idx =  (g_idx - item_idx * item_size_dim) / feature_map_dim;
        const int b = item_idx;
        const int y = (g_idx - item_idx * item_size_dim - item_channel_idx * feature_map_dim) / width;
        const int x = (g_idx - item_idx * item_size_dim - item_channel_idx * feature_map_dim) % width;
        int cid;
        int const_term = stride;

        for (int v_cid = item_channel_idx; true; v_cid += input_channel){
            
            #ifdef debug
            if (item_idx == 0 && x == 0 && y == x && item_channel_idx == 3)
            {
                printf("v_cid: %d\n", v_cid);
            }
            #endif

            int output_start_idx = v_cid / const_term;
            int output_start_offset = v_cid % const_term;

            int output_end_idx = 0;
            int output_end_offset = 0;
            
            if (v_cid < input_unit_dim){
                output_end_idx = 0;
                output_end_offset = v_cid;
            } 
            else{
                output_end_idx = __float2int_rd((v_cid - input_unit_dim) * 1.0f / const_term) + 1; //���ƾ�������((n-f)/s+1),��ʾ�������ͨ������ڼ���������
                output_end_offset = v_cid - const_term * output_end_idx; 
            }  //��������˳��ϲ���s����ʾ�����˵ĵ�һ��ͨ����λ�ã� v_cid��ʾĿǰ����ͨ��λ�ã� v_cid-�þ����˵ĵ�һ��ͨ����λ�õ���v_cid�ڸþ����˵�һ��ͨ������Ծ���

            #ifdef debug
            if (item_idx == 0 && x == 0 && y == x && item_channel_idx == 3)
            printf("opt_start_idx: %d\nopt_start_offset: %d\nopt_end_idx: %d\nopt_end_offset: %d\n\n", output_start_idx, output_start_offset, output_end_idx, output_end_offset);
            #endif

            if (output_start_idx >= output_channel && output_end_idx >= output_channel) break;
						
            cid = v_cid % input_channel;
            if (output_start_idx == output_end_idx){
                d_input[b][cid][y][x] += weights[output_start_idx][output_start_offset] * d_output[b][output_start_idx][y][x];

                #ifdef enforce_atomic
                atomicAdd((float*)&d_weights[output_start_idx][output_start_offset], input[b][cid][y][x] * d_output[b][output_start_idx][y][x]);
                #else
                d_weights[output_start_idx][output_start_offset] += input[b][cid][y][x] * d_output[b][output_start_idx][y][x];
                #endif
            }
            else{
				for(int chout = output_end_idx; chout <= output_start_idx; chout++)
				{	
					if(chout < output_channel)
					{
						int chout_offset = (chout - output_end_idx) * const_term;
						d_input[b][cid][y][x] += weights[chout][output_end_offset - chout_offset] * d_output[b][chout][y][x];
						
						#ifdef enforce_atomic
						atomicAdd((float*)&d_weights[chout][output_end_offset - chout_offset], input[b][cid][y][x] * d_output[b][chout][y][x]);
						#else
						d_weights[chout][output_end_offset - chout_offset] += input[b][cid][y][x] * d_output[b][chout][y][x];
						#endif						
					}
				}
            }

        }
    }
}
#endif


