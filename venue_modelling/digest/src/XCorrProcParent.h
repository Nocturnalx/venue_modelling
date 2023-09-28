#pragma once

class XCorrProcParent
{
    protected:
        /* data */
        short int frameNo = 0;
        float xcor = 0; //xcorr value
        float numerator = 0;
        float divisor = 0;
        float src_sum = 0;
        float res_sum = 0;

    public:

        short int * m_srcMonoBuff; //audio buffer for only the first ray that hits the reciever
        short int * m_d_srcMonoBuff;

        XCorrProcParent(/* args */);
        ~XCorrProcParent();

        //memory management methods
        void deviceMallocSrc(int audioLeng);
        void deviceFreeSrc();
        void hostMallocSrc(int audioLeng);
        void hostFreeSrc();
};

