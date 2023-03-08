// This program is tested on linux machine with g++ compiler.

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <bitset>

using namespace std;

char readingData = 0;

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

int main()
{
	FILE * infile = fopen("testfile.wav","rb");		    // Open wave file in read mode
	FILE * outfile = fopen("Output.wav","wb");		    // Create output ( wave format) file in write mode

    int BUFSIZE = 524287;					            // BUFSIZE can be changed according to the frame size required (eg:512)
	int count = 0;						                // For counting number of frames in wave file.
	short int buff[BUFSIZE];				            // short int - 2 bytes - 16 bit
	header_p meta = (header_p)malloc(sizeof(header));	// header_p points to a header struct that contains the wave file metadata fields
	int nb;							                    // variable storing number of bytes returned

    char check_buff[1];
    char data_buff[4];
    int data_size[1];

    short int leftBuff[BUFSIZE/2];
    short int rightBuff[BUFSIZE/2];

	if (infile)
	{
		fread(meta, 1, sizeof(header), infile);
		fwrite(meta,1, sizeof(*meta), outfile);
		cout << " Size of Header file is "<<sizeof(*meta)<<" bytes" << endl;
		cout << " Sampling rate of the input wave file is "<< meta->sample_rate <<" Hz" << endl;

        //testing
        for (int i = 0; i < 4; i++){
            cout << meta->chunk_id[i] << endl;
        }

        int num = meta->subchunk1_size;
        cout << num <<endl;

        //checking for data tag
        while (!readingData){
            fread(check_buff, 1, sizeof(char), infile);
            fwrite(check_buff, 1, sizeof(char), outfile);

            for (int i = 0; i < 3; i++){
                data_buff[i] = data_buff[i + 1];
            }

            data_buff[3] = check_buff[0];

            //there must be a better way to do this?????
            if (data_buff[3] == 'a' && data_buff[2] == 't' && data_buff[1] == 'a' && data_buff[0] == 'd'){
                readingData = !readingData;
            }
        }

        //read next 4 bytes which is data size
        fread(data_size, 1, sizeof(int), infile);
        fwrite(data_size, 1, sizeof(int), outfile);

        cout << "data size: " << data_size[0] << endl;

        //read file untill end
		while (!feof(infile))
		{
            // Reading data in chunks of BUFSIZE
			nb = fread(buff,1,BUFSIZE,infile);

            //proc
            int n = 0;
            for(int i = 0; i < BUFSIZE; i+=2){
                leftBuff[n] = buff[i];
                rightBuff[n] = buff[i + 1];

                n++;
            }

            // Writing read data into output file
			fwrite(buff,1,nb,outfile);

            // Increment Number of frames
			count++;
		}

	    cout << " Number of frames in the input wave file are " << count << endl;
	}

    return 0;
}