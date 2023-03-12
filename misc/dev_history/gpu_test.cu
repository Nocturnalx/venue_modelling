#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define audio_leng 10000000
#define rooms 3
#define MAX_ERR 1e-6

using namespace std;

//~~~~PROGRAM~~~~
__global__ void vector_add(short int *d_combined, short int *audio, float abs, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handling arbitrary vector size
    if (tid < n){
        d_combined[tid] += (audio[tid] * abs) / 3;
    }
}

int main(){
    short int *audio, *combined;
    short int *d_audio, *d_combined;
    // *d_abs;

    float abs[3] = {0.4,0.7,0.2};

    //allocate host memory
    audio = (short int*)malloc(sizeof(short int) * audio_leng);
    combined = (short int*)malloc(sizeof(short int) * audio_leng);

    // Allocate device memory
    cudaMalloc((void**)&d_audio, sizeof(short int) * audio_leng);
    cudaMalloc((void**)&d_combined, sizeof(short int) * audio_leng);
    // cudaMalloc((void**)&d_abs, sizeof(float) * rooms);

    // Initialize array
    for(int i = 0; i < audio_leng; i++){
        audio[i] = 1000;
        combined[i] = 0;
    }

    // Transfer data from host to device memory
    cudaMemcpy(d_audio, audio, sizeof(short int) * audio_leng, cudaMemcpyHostToDevice);
    cudaMemcpy(d_combined, combined, sizeof(short int) * audio_leng, cudaMemcpyHostToDevice);

    // Executing kernel 
    int block_size = 256;
    int grid_size = ((audio_leng + block_size) / block_size); //add extra 256 to N so that when dividing it will round down to > required threads
    for (int r = 0; r < rooms; r++){
        vector_add<<<grid_size,block_size>>>(d_combined, d_audio, abs[r], audio_leng);
    }

    //transfer out array from device to host
    cudaMemcpy(combined, d_combined, sizeof(short int) * audio_leng, cudaMemcpyDeviceToHost);

    cout << "combined[7]: " << combined[7] << endl;

    // Cleanup after kernel execution
    // Deallocate device memory
    cudaFree(d_audio);
    cudaFree(d_combined);

    // Deallocate host memory
    free(audio); 
    free(combined);

    return 0;
}