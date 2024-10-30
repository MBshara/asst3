#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 1024


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}


__global__ void
d_upSweep(int N, int two_d, int* output){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int two_dplus1 = 2*two_d;
    int real_index = index*two_dplus1;
    if(index<N){
        output[real_index+two_dplus1-1] += output[real_index+two_d - 1];
    }
}

__global__ void 
d_downSweep(int N, int two_d, int* output){
    int index =  blockIdx.x * blockDim.x + threadIdx.x;
    int two_dplus1 = 2*two_d;
    int real_index = index*two_dplus1;
    if(index<N){
        int t = output[real_index+two_d -1];
        output[real_index+two_d - 1] = output[real_index+two_dplus1 -1];
        output[real_index+two_dplus1 - 1] +=t;
    }
}

// Doesn't work because does not sure ordering of warps match with data ordering.
__global__ void
ex_scan(int N, int* input, int*output)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int two_d = 1; two_d <= N/2; two_d*=2){
        int two_dplus1 = 2*two_d;
        int real_index = index*two_dplus1;
        if (real_index < N){
            output[real_index+two_dplus1-1] += output[real_index+two_d - 1];
        }
    }

    output[N-1] = 0;

    for (int two_d = N/2; two_d >= 1;  two_d/=2){
        int two_dplus1 = 2*two_d;
        int real_index = index*two_dplus1;
        if(real_index < N){
            int t = output[real_index+two_d -1];
            output[real_index+two_d - 1] = output[real_index+two_dplus1 -1];
            output[real_index+two_dplus1 - 1] +=t;
        }
    }
}
// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{
    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.
    if(N==1){
        d_downSweep<<<1,1>>>(1,1,result);
    }
    else{
        int N_next = nextPow2(N); // For the 1 case
        for (int two_d = 1; two_d < N_next/2; two_d*=2){
            int two_dplus1 = 2*two_d;
            int num_of_threads = N_next/two_dplus1;
            int num_of_blocks = (num_of_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            d_upSweep<<<num_of_blocks,std::min(num_of_threads,THREADS_PER_BLOCK)>>>(num_of_threads,two_d,result);
            cudaDeviceSynchronize();
        }
        cudaMemset(&result[N_next - 1], 0, sizeof(int));
        cudaDeviceSynchronize();

        for (int two_d = N_next/2; two_d >= 1;  two_d/=2){
            int two_dplus1 = 2*two_d;
            int num_of_threads = N_next/two_dplus1;
            int num_of_blocks = (num_of_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            d_downSweep<<<num_of_blocks,std::min(num_of_threads,THREADS_PER_BLOCK)>>>(num_of_threads,two_d,result);
            cudaDeviceSynchronize();
        }
    }
    // printf("%d",N_next);
    // ex_scan<<<num_of_blocks, threads_per_block>>>(N2,input,result);
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// gather operation, kind off.
__global__ void
d_GatherSameVals(int* input, int length, int* output){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index<length-1){
        output[index] = (input[index]==input[index+1]) ? 1 : 0;
    }
}

// 
__global__ void
d_GatherIndices(int* input, int length, int* output){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index<length-1){
        if((input[index]!=input[index+1])){
            output[input[index]] = index;
        }
    }
}

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // Check if the next value is the same as current
    // If so make it 0.
    // If not make it 1.
    int N2 = nextPow2(length);

    int *temp;
    cudaMalloc(&temp, N2 * sizeof(int));
    int num_of_blocks = (THREADS_PER_BLOCK + length -1) / THREADS_PER_BLOCK;

    d_GatherSameVals<<<num_of_blocks,std::min(length,THREADS_PER_BLOCK)>>>(device_input,length, temp);
    cudaDeviceSynchronize();

    exclusive_scan(temp, N2, temp);
    
    d_GatherIndices<<<num_of_blocks,std::min(length,THREADS_PER_BLOCK)>>>(temp,length, device_output);
    cudaDeviceSynchronize();

    int result;
    cudaMemcpy(&result, &temp[length-1], sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(temp);

    return result; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
