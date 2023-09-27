#pragma once

#include <fstream>
#include "AudioProc.h"

// WAVE PCM soundfile format
typedef struct header_file
{
    char chunk_id[4];
    int chunk_size;
    char format[4]; //16 bytes

    char subchunk1_id[4];
    int subchunk1_size;
    short int audio_format;
    short int num_channels;
    int sample_rate;			// sample_rate denotes the sampling rate.
    int byte_rate;
    short int block_align;
    short int bits_per_sample; //36 bytes
} header;

typedef struct header_file* header_p;

// struct procData{
//     int sampleRate;
//     int frameSize_out;
//     int audio_leng;

//     int err_code;
// };

enum read_err_codes{
    readSuccess,
    wrongFormat, //is wav but not 16bit/stereo/pcm
    badFile, //not wav or corrupted in some way
    miscErr
};

class FileHandler {

    private:
        /* data */
        FILE * m_infile; // Open wave file in read mode
        FILE * m_outfile; // Create output ( wave format) file in write mode

        std::string m_inPath;
        std::string m_tempPath;
        std::string m_outPath;

        std::string m_username;

        int m_data_size;

        header_p m_meta;

        void writeVISUHeaders(std::unique_ptr<AudioProc> & AProc);

    public:
        FileHandler(std::string username);
        ~FileHandler();

        read_err_codes readFile(std::unique_ptr<AudioProc> & AProc);

        void writeWav(std::unique_ptr<AudioProc> & AProc);

        void writeVISU(std::unique_ptr<AudioProc> & AProc);

        void finalise();

        void writeErr(read_err_codes err_code);
};
