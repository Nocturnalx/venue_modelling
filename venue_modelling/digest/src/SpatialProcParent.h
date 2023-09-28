#pragma once

#include "AudioProcStructs.h"

class SpatialProcParent {

    protected:

        //gathered from sql user table
        int m_resolution; //if resolution was 100 then it would be per cm
        int m_xLength;
        int m_yLength;
        int m_zLength;
        int m_points;
        int m_gain;
        int m_order = 4; //default 4? why? 
        coefs m_coefs_x;
        coefs m_coefs_y;
        coefs m_coefs_z;

        //set in bake()
        point m_speaker_L;
        point m_speaker_R;
        int m_rooms;
        room * m_roomArr;

        //got from file read
        int m_audioLeng;
        int m_sampleRate;
        int m_frameSize;

        //13230000 is 5 mins @ 44,100 so hard cap on song length at 5 mins
        int LENG_MAX = 13230000; //max length of song in samples

    public:

        bool fileRead = 0;

        short int * left_in;
        short int * right_in;
        short int * d_left_in;
        short int * d_right_in;
        short int * monoBuff;
        short int * d_monoBuff;

        SpatialProcParent();
        ~SpatialProcParent();

        int getAudioLeng();
        void setAudioLeng(int val);
        int getSampleRate();
        void setSampleRate(int val);
        int getFrameSize();
        void setFrameSize(int val);
        //0 = left, 1 = right
        point getSpeakerPosition(char side);

        dimensions getDimensions();


        void copyMonoFromDevice();
        void copyStereoToDevice();


        void deviceMallocStereo();
        void deviceFreeStereo();
        void hostMallocStereo();
        void hostFreeStereo();

        void deviceMallocMono();
        void deviceFreeMono();
        void hostMallocMono();
        void hostFreeMono();
};

