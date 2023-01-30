// This program is tested on linux machine with g++ compiler.

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <bitset>
#include <cmath>

using namespace std;

const float pi = 3.141592;

int BUFSIZE;
int sampleRate = 41000; //could be changed but for now sample rate stays hard coded for 41000

//for file read
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

//for room simulation
int resolution = 100; //if resolution was 100 then it would be per cm
int xLength = 2; //get x,y or z from the actual 3D model then times by resolution (so "10" should be model.getlength() or whatever)
int yLength = 2;
int zLength = 4;
int points;

float* pointArray;

struct point{
    int x;
    int y;
    int z;
};

point speaker_L;
point speaker_R;

int reflections_x = 1;
int reflections_y = 1;
int reflections_z = 1;
int rooms;

struct room{
    int xOffset;
    int yOffset;
    int zOffset;
    //room struct can also contain total absorption coeficient
};

room* roomArr;


//funcs

void init(){
    //init dimensions and point num
    xLength = xLength * resolution;
    yLength = yLength * resolution;
    zLength = zLength * resolution;
    points = xLength * yLength * zLength;
    pointArray = new float[points]; //array that will contain the index score for each point

    //init rooms
    //reflections is extra on the border, this adds both sides of border plus middle to get total dimension length
    int rooms_x = (reflections_x * 2) + 1;
    int rooms_y = (reflections_y * 2) + 1;
    int rooms_z = (reflections_z * 2) + 1;

    //total rooms is x*y*z
    rooms = rooms_x * rooms_y * rooms_z;

    roomArr = new room[rooms];

    //total min offset is the length of room * num of reflections in a specific direction
    int minOffset_x = 0 - (xLength * reflections_x);
    int minOffset_y = 0 - (yLength * reflections_y);
    int minOffset_z = 0 - (zLength * reflections_z);

    //applying x,y,z offsets for each room
    int i = 0;
    for(int z = 0; z < rooms_z; z++){
        for(int y = 0; y < rooms_y; y++){
            for(int x = 0; x < rooms_x; x++){
                roomArr[i].xOffset = minOffset_x + (xLength * x);
                roomArr[i].yOffset = minOffset_y + (yLength * y);
                roomArr[i].zOffset = minOffset_z + (zLength * z);

                // cout << "x: " << roomArr[i].xOffset;
                // cout << " y: " << roomArr[i].yOffset;
                // cout << " z: " << roomArr[i].zOffset << endl;

                i++;
            }
        }
    }

    //speaker positioning
    speaker_L.x = xLength/4;
    speaker_L.y = yLength/2;
    speaker_L.z = 1;

    speaker_R.x = (xLength/4) * 3;
    speaker_R.y = yLength/2;
    speaker_R.z = 1;
}

//returns distance between point and speaker in full meters
float get_dist(int x, int y, int z, point speaker){

    float dist = sqrt(pow(x-speaker.x, 2)+pow(y-speaker.y, 2)+pow(z-speaker.z, 2)) / resolution;

    return dist;
}


void process(short int * leftBuff_p, short int * rightBuff_p){

    //short int monoBuff_src[BUFSIZE/2];
    //for BUFFSIZE/2 combine left and right to mono without delay
    
    int p = 0;

    for (int z = 1; z <= zLength; z++){
        for (int y = 1; y <= yLength; y++){
            for (int x = 1; x <= xLength; x++){
                
                short int leftBuffArr[rooms][BUFSIZE/2];
                short int rightBuffArr[rooms][BUFSIZE/2];

                short int comb_leftBuff[BUFSIZE];
                short int comb_rightBuff[BUFSIZE];

                short int monoBuff[BUFSIZE/2];

                //populate room buffer arrays
                for (int r = 0; r < rooms; r++){
                    
                    int room_x = x + roomArr[r].xOffset;
                    int room_y = y + roomArr[r].yOffset;
                    int room_z = z + roomArr[r].zOffset;

                    //dist to left
                    float dist_L = get_dist(room_x, room_y, room_z, speaker_L);
                    //dist to right
                    float dist_R = get_dist(room_x, room_y, room_z, speaker_R);

                    //time delays in sec
                    int delay_L = (dist_L/343);
                    int delay_R = (dist_R/343);

                    //time delays in samples
                    int delay_L_samp = delay_L * sampleRate;
                    int delay_R_samp = delay_R * sampleRate;

                    //for each smaple in each buff
                    for (int i = BUFSIZE/2; i >= 0; i++){
                        //apply delays L&R, if index - delay is negative then that index should be 0
                        if (i >= delay_L_samp){
                            leftBuff_p[i] = leftBuff_p[i - delay_L_samp];
                        } else {
                            leftBuff_p[i] = 0;
                        }

                        if (i >= delay_L_samp){
                            rightBuff_p[i] = rightBuff_p[i - delay_R_samp];
                        } else {
                            rightBuff_p[i] = 0;
                        }

                        //times by invers square for distance loss 
                        leftBuff_p[i] = leftBuff_p[i] * (4*pi*pow(dist_L, 2));
                        rightBuff_p[i] = rightBuff_p[i] * (4*pi*pow(dist_R, 2));

                        //times by abs
                        //roomArr[r].totalAbs

                        //add delayed buffs to buff array - divide by rooms to get avverage when summing
                        leftBuffArr[r][i] = leftBuff_p[i] / rooms;
                        rightBuffArr[r][i] = rightBuff_p[i] / rooms;
                    }


                    //below section could be combined to reduce mem usage?

                    //combine each room audio samples for both channels
                    for (int r = 0; r < rooms; r++){
                        for (int i = 0; i < BUFSIZE/2; i++){
                            comb_leftBuff[i] += leftBuffArr[r][i];
                            comb_rightBuff[i] += rightBuffArr[r][i];
                        }
                    }

                    //here wav file could be reconstructed

                    //combine channels to mono
                    for (int i = 0; i < BUFSIZE/2; i++){
                        monoBuff[i] = (comb_leftBuff[i] / 2) + (comb_rightBuff[i] / 2);
                    }

                    //split monobuff into 3 sec chunks and do cross corr on it vs combined source 
                }

                p++;
            }
        }
    }
}

int main()
{
    init();

	FILE * infile = fopen("testfile.wav","rb");		    // Open wave file in read mode
	FILE * outfile = fopen("Output.wav","wb");		    // Create output ( wave format) file in write mode

    //BUFSIZE = 524288;					                // 524288 is ~ 3 seconds at 41,000
	int count = 0;						                // For counting number of frames in wave file.
	short int buff[BUFSIZE];				            // short int - 2 bytes - 16 bit
	header_p meta = (header_p)malloc(sizeof(header));	// header_p points to a header struct that contains the wave file metadata fields
	int nb;							                    // variable storing number of bytes returned

    char check_buff[1];
    char data_buff[4];
    int data_size[1];

    short int leftBuff[BUFSIZE/2];
    short int * leftBUff_p = leftBuff;
    short int rightBuff[BUFSIZE/2];
    short int * rightBuff_p = rightBuff;

	if (infile)
	{
		fread(meta, 1, sizeof(header), infile);
		fwrite(meta,1, sizeof(*meta), outfile); // - TO BE DELETED
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
            fwrite(check_buff, 1, sizeof(char), outfile); // - TO BE DELETED

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
        BUFSIZE = data_size[0]; //amount read per pass is total length of data

        fwrite(data_size, 1, sizeof(int), outfile); //  - TO BE DELETED

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

            // Writing read data into output file - TO BE DELETED
			fwrite(buff,1,nb,outfile);

            //get delay times, delay samples by delay times, combine samples, do cross corr
            process(leftBUff_p, rightBuff_p);

            // Increment Number of frames
			count++;
		}

	    cout << " Number of frames in the input wave file are " << count << endl;
	}

    delete [] pointArray;
    delete [] roomArr;
    return 0;
}