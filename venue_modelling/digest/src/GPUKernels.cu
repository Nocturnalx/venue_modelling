#include "GPUKernels.h"

__global__ void combine(short int *d_monoBuff, short int *d_left_in, short int *d_right_in, float abs, int delay_l, int delay_r, int n){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //thread id must be less than array length
    if (tid < n){
        short int sample_l = 0;
        //if larger than delay start reading the audio
        if (tid >= delay_l){
            sample_l = d_left_in[tid - delay_l];
            sample_l *= abs; // absorption loss
            sample_l = sample_l / 2; //average of both audio streams for combining to mono
        }
        d_monoBuff[tid] += sample_l; 

        short int sample_r = 0;
        //if larger than delay start reading the audio
        if (tid >= delay_r){
            sample_r = d_right_in[tid - delay_r];
            sample_r = sample_r * abs; // absorption loss
            sample_r = sample_r / 2; //average of both audio streams for combining to mono
        }
        d_monoBuff[tid] += sample_r; 
    }
}