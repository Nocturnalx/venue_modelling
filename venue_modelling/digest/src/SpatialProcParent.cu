#include "SpatialProcParent.h"

#include <cuda.h>
#include <cuda_runtime.h>

SpatialProcParent::SpatialProcParent(/* args */)
{
}

SpatialProcParent::~SpatialProcParent()
{
}

//gets and sets
int SpatialProcParent::getAudioLeng(){

    return m_audioLeng;
}

void SpatialProcParent::setAudioLeng(int val){

    m_audioLeng = val;
}

int SpatialProcParent::getSampleRate(){
    return m_sampleRate;
}

void SpatialProcParent::setSampleRate(int val){
    m_sampleRate = val;
}

int SpatialProcParent::getFrameSize(){
    return m_frameSize;
}

void SpatialProcParent::setFrameSize(int val){
    m_frameSize = val;
}

point SpatialProcParent::getSpeakerPosition(char side){
    if (side == 0){
        return m_speaker_L;
    } else {
        return m_speaker_R;
    }
}

dimensions SpatialProcParent::getDimensions(){

    dimensions dims;

    dims.xLength = m_xLength;
    dims.yLength = m_yLength;
    dims.zLength = m_zLength;

    return dims;
}


//host and device memory untilities

void SpatialProcParent::copyStereoToDevice(){

    cudaMemcpy(d_left_in, left_in, sizeof(short int) * m_audioLeng, cudaMemcpyHostToDevice);
    cudaMemcpy(d_right_in, right_in, sizeof(short int) * m_audioLeng, cudaMemcpyHostToDevice);
}

//stereo buffers dev and host
void SpatialProcParent::deviceMallocStereo(){

    //need checks audioleng is set

    cudaMalloc((void**)&d_left_in, sizeof(short int) * m_audioLeng);
    cudaMalloc((void**)&d_right_in, sizeof(short int) * m_audioLeng);
}

void SpatialProcParent::deviceFreeStereo(){
    cudaFree(d_left_in);
    cudaFree(d_right_in);
}

void SpatialProcParent::hostMallocStereo(){

    //need checks audioleng is set

    left_in = new short int [m_audioLeng];
    right_in = new short int [m_audioLeng];
}

void SpatialProcParent::hostFreeStereo(){

    delete [] left_in;
    delete [] right_in;
}

//mono buffer dev and host
void SpatialProcParent::deviceMallocMono(){

    //need checks audioleng is set

    cudaMalloc((void**)&d_monoBuff, sizeof(short int) * m_audioLeng);
}

void SpatialProcParent::deviceFreeMono(){
    cudaFree(d_monoBuff);
}

void SpatialProcParent::hostMallocMono(){

    //need checks audioleng is set
    
    monoBuff = new short int [m_audioLeng];
}

void SpatialProcParent::hostFreeMono(){
    delete [] monoBuff;
}