#pragma once

//keep these in here as compiler can get confused and not know where to look for these if using g++ for header file
// #include <cuda.h>
// #include <cuda_runtime.h>

// calculates all sample values for an individual ray hitting the reciever
__global__ void combine(short int *d_monoBuff, short int *d_left_in, short int *d_right_in, float abs, int delay_l, int delay_r, int n);