// This program is tested on linux machine with g++ compiler.

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <bitset>
#include <cmath>

using namespace std;

const float pi = 3.141592;

FILE * infile = fopen("testfile.wav","rb");		    // Open wave file in read mode
FILE * outfile = fopen("Output.wav","wb");		    // Create output ( wave format) file in write mode

int BUFFSIZE;
int sampleRate = 41000; //could be changed but for now sample rate stays hard coded for 41000
int nb;	// variable storing number of bytes returned

short int * buff;
short int * leftBuff;
short int * rightBuff;

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

    //short int monoBuff_src[BUFFSIZE/2];
    //for BUFFSIZE/2 combine left and right to mono without delay

    cout << "processing\n";

    short int comb_leftBuff[BUFFSIZE];
    short int comb_rightBuff[BUFFSIZE];

    short int monoBuff[BUFFSIZE/2];

    cout << "allocated proc buffs\n";

    //for testing, to reverse use nested x,y,z loops for co-ord vals
    int x = xLength/2;
    int y = yLength/2;
    int z = (zLength/4) * 3;

    //populate room buffer arrays
    for (int r = 0; r < rooms; r++){
        
        cout << "room: " << r << endl;

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
        for (int i = BUFFSIZE/2; i >= 0; i++){
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

            //add delayed buffs to buff array - divide by rooms to get average when summing
            comb_leftBuff[i] += leftBuff_p[i] / rooms;
            comb_rightBuff[i] += rightBuff_p[i] / rooms;
        }

        cout << "gathered stereo buffs\n";

        //here wav file could be reconstructed
        short int out[BUFFSIZE];
        for (int i = 0; i < BUFFSIZE/2; i++){
                out[i*2] = comb_leftBuff[i];
                out[(i*2) + 1] = comb_rightBuff[i];
        }
        fwrite(out,1,BUFFSIZE,outfile);

        //combine channels to mono
        for (int i = 0; i < BUFFSIZE/2; i++){
            monoBuff[i] = (comb_leftBuff[i] / 2) + (comb_rightBuff[i] / 2);
        }

        cout << "combined stereo buffs to mono\n";

        //split monobuff into 3 sec chunks and do cross corr on it vs combined source 
    }
}

int main()
{
    init();

    int frameBuffSize = 4096;					                // 524288 is ~ 3 seconds at 41,000
	int count = 0;						                // For counting number of frames in wave file.
	header_p meta = (header_p)malloc(sizeof(header));	// header_p points to a header struct that contains the wave file metadata fields
    int nb_tot = 0; //total bytes read

    char check_buff[1];
    char data_buff[4];
    int data_size[1];

	if (infile)
	{
		fread(meta, 1, sizeof(header), infile);
		fwrite(meta,1, sizeof(*meta), outfile); // - TO BE DELETED
		cout << " Size of Header file is "<<sizeof(*meta)<<" bytes" << endl;
		cout << " Sampling rate of the input wave file is "<< meta->sample_rate <<" Hz" << endl;

        //testing - prints WAVE
        for (int i = 0; i < 4; i++){
            cout << meta->format[i];
        }
        cout << endl;

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
        cout << "data found!\n";

        //read next 4 bytes which is data size
        fread(data_size, 1, sizeof(int), infile);
        BUFFSIZE = data_size[0]/2; //amount read per pass is total length of data, data size is in bytes so use /2 to get short int length
        cout << "data size: " << BUFFSIZE << endl;

        buff = new short int[frameBuffSize];
        leftBuff = new short int[BUFFSIZE/2];
        rightBuff = new short int[BUFFSIZE/2];

        cout << "allocated left and right buffs\n";

        fwrite(data_size, 1, sizeof(int), outfile); //  - TO BE DELETED

        //read file untill end
		while (!feof(infile))
		{
            // Reading data in chunks of BUFFSIZE
			nb = fread(buff,1,frameBuffSize,infile);
            
            cout << "splitting\n";
            //send left and right channels to respective buffers
            int n = 0;
            for(int i = 0; i < nb/2; i+=2){
                leftBuff[(nb_tot/2) + n] = buff[i];
                rightBuff[(nb_tot/2) + n] = buff[i + 1];

                n++;
            }
            cout << bitset<sizeof(leftBuff[0])*8>(leftBuff[nb_tot/2]) << endl;
            nb_tot += nb;

            cout << "split buff into stereo\n";

            // Writing read data into output file - TO BE DELETED
			fwrite(buff,1,nb,outfile);

            // Increment Number of frames
            cout << "frame: " << count << endl;
			count++;
		}

        //get delay times, delay samples by delay times, combine samples, do cross corr
        //process(leftBuff, rightBuff);

	    cout << " Number of frames in the input wave file are " << count << endl;
	}

    delete [] pointArray;
    delete [] roomArr;
    return 0;
}