// This program is tested on linux machine with g++ compiler.

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <bitset>
#include <cmath>

using namespace std;

FILE * infile = fopen("slaves.wav","rb");		    // Open wave file in read mode
FILE * outfile = fopen("Output.vis.wav","wb");		    // Create output ( wave format) file in write mode

int audio_leng;
int sampleRate = 44100; //could be changed but for now sample rate stays hard coded for 41000
int nb;	// variable storing number of bytes returned
int data_size[1];

//13230000 is 5 mins @ 44,100 so hard cap on song length at 5 mins
int LENG_MAX = 13230000; //max length of song in samples
short int left_in[13230000];
short int right_in[13230000];

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

int frameSize_out = 88200;

//for room simulation
int resolution = 1; //if resolution was 100 then it would be per cm
int xLength = 6; //get x,y or z from the actual 3D model then times by resolution (so "10" should be model.getlength() or whatever)
int yLength = 20;
int zLength = 20;
int points;

float* pointArray;

struct point{
    int x;
    int y;
    int z;
};

point speaker_L;
point speaker_R;

int order = 4;
int rooms;

struct room{
    int pos_x;
    int pos_y;
    int pos_z;

    bool mirrored_x = 0;
    bool mirrored_y = 0;
    bool mirrored_z = 0;

    float totalAbs = 1;
};

room* roomArr;

struct coefs {
    float neg = 1;
    float pos = 1;
};

coefs coefs_x;
coefs coefs_y;
coefs coefs_z;

//funcs
void init(){
    //init dimensions and point num
    xLength = xLength * resolution;
    yLength = yLength * resolution;
    zLength = zLength * resolution;
    points = xLength * yLength * zLength;

    //init rooms
    //reflections is extra on the border, this adds both sides of border plus middle to get total dimension length
    int rooms_x = (order * 2) + 1;
    int rooms_y = (order * 2) + 1;
    int rooms_z = (order * 2) + 1;

    //total rooms is x*y*z
    rooms = rooms_x * rooms_y * rooms_z;

    roomArr = new room[rooms]; 

    //total min offset is the length of room * num of reflections in a specific direction
    int minOffset_x = 0 - (xLength * order);
    int minOffset_y = 0 - (yLength * order);
    int minOffset_z = 0 - (zLength * order);

    //wall absorption coeffs
    coefs_x.neg = 0.9;
    coefs_x.pos = 0.9;
    coefs_y.neg = 0.7;
    coefs_y.pos = 0.9;
    coefs_z.neg = 0.9;
    coefs_z.pos = 0.9;

    //calculate total abs coeffs and check if room is mirrored
    int r = 0;
    for(int y = 0; y < rooms_y; y++){
        for(int z = 0; z < rooms_z; z++){
            for(int x = 0; x < rooms_x; x++){

                roomArr[r].pos_x = x - order;
                roomArr[r].pos_y = y - order;
                roomArr[r].pos_z = z - order;

                int absolute_x = abs(roomArr[r].pos_x);
                int absolute_y = abs(roomArr[r].pos_y);
                int absolute_z = abs(roomArr[r].pos_z);

                if (roomArr[r].pos_x % 2 != 0){
                    //is mirrored in this axis
                    roomArr[r].mirrored_x = 1;

                    float extra_coef = 1;
                    //extra coef on the end changes whether going in positive or negative direction
                    if (roomArr[r].pos_x < 0){
                        extra_coef *= coefs_x.pos;
                    } else {
                        extra_coef *= coefs_x.neg;
                    }

                    roomArr[r].totalAbs *= pow(coefs_x.neg, (absolute_x - 1) / 2) * pow(coefs_x.pos, (absolute_x - 1) / 2) * extra_coef;
                } else {
                    roomArr[r].totalAbs *= pow(coefs_x.neg, absolute_x / 2) * pow(coefs_x.pos, absolute_x / 2);
                }

                if (roomArr[r].pos_y % 2 != 0){
                    roomArr[r].mirrored_y = 1;

                    float extra_coef = 1;
                    //extra coef on the end changes whether going in positive or negative direction
                    if (roomArr[r].pos_y < 0){
                        extra_coef *= coefs_y.pos;
                    } else {
                        extra_coef *= coefs_y.neg;
                    }

                    roomArr[r].totalAbs *= pow(coefs_y.neg, (absolute_y - 1) / 2) * pow(coefs_y.pos, (absolute_y - 1) / 2) * extra_coef;
                } else {
                    roomArr[r].totalAbs *= pow(coefs_y.neg, absolute_y / 2) * pow(coefs_y.pos, absolute_y / 2);
                }

                if (roomArr[r].pos_z % 2 != 0){
                    roomArr[r].mirrored_z = 1;

                    float extra_coef = 1;
                    //extra coef on the end changes whether going in positive or negative direction
                    if (roomArr[r].pos_z < 0){
                        extra_coef *= coefs_z.pos;
                    } else {
                        extra_coef *= coefs_z.neg;
                    }

                    roomArr[r].totalAbs *= pow(coefs_z.neg, (absolute_z - 1) / 2) * pow(coefs_z.pos, (absolute_z - 1) / 2) * extra_coef;
                } else {
                    roomArr[r].totalAbs *= pow(coefs_z.neg, absolute_z / 2) * pow(coefs_z.pos, absolute_z / 2);
                }

                r++;
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

void process(){

    cout << "processing\n";

    short int * comb_leftBuff;
    short int * comb_rightBuff;

    int * monoBuff;
    int * src_monoBuff;

    //for testing, to reverse use nested x,y,z loops for co-ord vals
    int x = xLength/2;
    int y = yLength/2;
    int z = (zLength/4) * 3;

    //for (x,y,z)...
    //one of these for each point
    comb_leftBuff = new short int[audio_leng];
    comb_rightBuff = new short int[audio_leng];
    monoBuff = new int[audio_leng];
    src_monoBuff = new int[audio_leng];

    //populate room buffer arrays
    for (int r = 0; r < rooms; r++){

        int room_x = x + (roomArr[r].pos_x * xLength);
        int room_y = y + (roomArr[r].pos_y * yLength);
        int room_z = z + (roomArr[r].pos_z * zLength);

        if (roomArr[r].mirrored_x){
            room_x = room_x + (2 * ((xLength/2) - x));
        }

        if (roomArr[r].mirrored_y){
            room_y = room_y + (2 * ((yLength/2) - y));
        }

        if (roomArr[r].mirrored_z){
            room_z = room_z + (2 * ((zLength/2) - z));
        }

        //dist to left
        float dist_L = get_dist(room_x, room_y, room_z, speaker_L);
        //dist to right
        float dist_R = get_dist(room_x, room_y, room_z, speaker_R);

        //time delays in sec
        float delay_L = (dist_L/343);
        float delay_R = (dist_R/343);

        //time delays in samples
        int delay_L_samp = delay_L * sampleRate;
        int delay_R_samp = delay_R * sampleRate;

        //for each smaple in each buff
        for (int i = audio_leng; i >= 0; i--){

            short int sample;

            //apply delays L&R, if index-delay is negative then that index should be 0
            if (i >= delay_L_samp){
                sample = left_in[i - delay_L_samp];
            } else {
                sample = 0;
            }

            if (i >= delay_R_samp){
                sample = right_in[i - delay_R_samp];
            } else {
                sample = 0;
            }

            //frequency dependant air absorption, could be done later

            //times by abs
            sample *= roomArr[r].totalAbs;
            sample *= roomArr[r].totalAbs;

            //add delayed buffs to buff array - divide by rooms to get average when summing
            comb_leftBuff[i] += sample / 6;
            comb_rightBuff[i] += sample / 6;
        }
    }

    //proccesing done

    //write 'DATA' identifier to wav file
    char dataTag[4] = {'d', 'a', 't', 'a'};
    fwrite(dataTag, 1, 4, outfile);

    //write dataSize to file
    fwrite(data_size, 1, sizeof(int), outfile);

    //here wav file could be reconstructed (i dont know why its *4 think it should be *2 but 4 works so ???????)
    short int * out;
    out = new short int[audio_leng*4];
    for (int i = 0; i < audio_leng; i++){
            out[i*2] = comb_leftBuff[i];
            out[(i*2) + 1] = comb_rightBuff[i];
    }

    fwrite(out,1,audio_leng*4,outfile);


    //write new 'visu' segment in ouput file
    char visuTag[4] = {'v', 'i', 's', 'u'};
    fwrite(visuTag, 1, 4, outfile);
    
    short int frameNo = 0;
    short int * frameNo_p = &frameNo;
    int val = 0; //xcorr value
    int * val_p = &val;

    int n = 0; //point in xcorr frame

    //combine channels to mono
    //split monobuff into 3 sec chunks and do cross corr on it vs combined source
    for (int i = 0; i < audio_leng; i++){

        src_monoBuff[i] = (left_in[i]/2) + (right_in[i]/2);
        monoBuff[i] = (comb_leftBuff[i] / 2) + (comb_rightBuff[i] / 2);

        //xcorr happens here
        val += src_monoBuff[i] * monoBuff[i];

        if (n == frameSize_out){
            cout << "val: " << val << endl << endl;

            //frameNo, value
            fwrite(frameNo_p, 1, 2, outfile);
            fwrite(val_p, 1, 4, outfile);

            val = 0;
            n = 0;
            frameNo++;
        }

        n++;
    }

    delete [] comb_leftBuff;
    delete [] comb_rightBuff;
    delete [] monoBuff;
    delete [] src_monoBuff;
    delete [] out;
}

void readFile(){
    int frameSize_in = 16384;
    short int buff[16384];
	int count = 0;						                // For counting number of frames in wave file.
	header_p meta = (header_p)malloc(sizeof(header));	// header_p points to a header struct that contains the wave file metadata fields
    int nb_tot = 0; //total bytes read

    char check_buff[1];
    char data_buff[4];

	if (infile)
	{
		fread(meta, 1, sizeof(header), infile);
        // meta->num_channels = 1; //change for mono
		fwrite(meta,1, sizeof(*meta), outfile);
		cout << "Size of Header file is "<<sizeof(*meta)<<" bytes" << endl;
		cout << "Sampling rate of the input wave file is "<< meta->sample_rate <<" Hz" << endl;

        //testing - prints WAVE
        for (int i = 0; i < 4; i++){
            cout << meta->format[i];
        }
        cout << endl;

        //checking for data tag
        while (!readingData){
            fread(check_buff, 1, sizeof(char), infile);
            // fwrite(check_buff, 1, sizeof(char), outfile);

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
        audio_leng = data_size[0]/4; //leng in samples: /2 to get short int then /2 to split stereo channels
        // data_size[0] = data_size[0]/2; //half data size if writing mono

        cout << "data size: " << audio_leng << endl;
        // fwrite(data_size, 1, sizeof(int), outfile); 

        int sampsWritten = 0; //no samples (short int) written to left and right buffs

        //read file untill end
		while (!feof(infile))
		{
            // Increment Number of frames
			count++;

            // Reading data in chunks of BUFFSIZE
			nb = fread(buff,1,frameSize_in,infile);
            
            //send left and right channels to respective buffers, nb is bytes so /2 to get short ints
            if (sampsWritten < LENG_MAX){
                for(int i = 0; i < nb/2; i+=2){
                    left_in[sampsWritten] = buff[i];
                    right_in[sampsWritten] = buff[i + 1];

                    sampsWritten++;
                }
            }
		}

        //get delay times, delay samples by delay times, combine samples, do cross corr
        process();

	    cout << "Number of frames in the input wave file are " << count << endl;
	}

    delete [] pointArray;
    delete [] roomArr;
}

int main()
{
    init();

    readFile();
    
    return 0;
}