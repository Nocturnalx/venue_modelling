#pragma once

#include <string.h>
#include <iostream>

#include "SQLUtils.h"

struct point {
    int x;
    int y;
    int z;
};

struct dimensions {
    int xLength;
    int yLength;
    int zLength;
};

struct room{
    int pos_x;
    int pos_y;
    int pos_z;

    bool mirrored_x = 0;
    bool mirrored_y = 0;
    bool mirrored_z = 0;

    float totalAbs = 1;
};

struct coefs {
    float neg = 1;
    float pos = 1;
};

class AudioProc {
    private:
        /* data */
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

        int m_audioLeng;
        int m_sampleRate;
        int m_frameSize;

        //used for file read not this 
        // int nb;	// variable storing number of bytes returned
        // int * data_size;

        //13230000 is 5 mins @ 44,100 so hard cap on song length at 5 mins
        int LENG_MAX = 13230000; //max length of song in samples

        void bake();

    public:

        bool fileRead = 0;

        short int * left_in;
        short int * right_in;
        short int * d_left_in;
        short int * d_right_in;
        short int * monoBuff;
        short int * d_monoBuff;

        AudioProc(/* args */);
        ~AudioProc();

        void init(std::string username);

        float get_dist(int x, int y, int z, point speaker);

        void process(int x, int y, int z);

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


