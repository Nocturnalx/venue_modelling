#include "XCorrProcParent.h"

XCorrProcParent::XCorrProcParent(/* args */)
{
}

XCorrProcParent::~XCorrProcParent()
{
}

void XCorrProcParent::deviceMallocSrc(int audioLeng){

    cudaMalloc((void**)&m_d_srcMonoBuff, sizeof(short int) * audioLeng);
}

void XCorrProcParent::deviceFreeSrc(){
    cudaFree(m_d_srcMonoBuff);
}

void XCorrProcParent::hostMallocSrc(int audioLeng){

    m_srcMonoBuff = new short int [audioLeng];
}

void XCorrProcParent::hostFreeSrc(){
    delete [] m_srcMonoBuff;
}
