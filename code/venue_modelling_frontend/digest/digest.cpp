#include <iostream>
#include <string.h>
#include <fstream>
#include <chrono>
#include <thread>
#include <stdlib.h>
#include <bitset>
#include <cmath>

#include "mysql_connection.h"

#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>

using namespace std;
using namespace std::this_thread; // sleep_for, sleep_until
using namespace std::chrono; // nanoseconds, system_clock, seconds

// ##file reading vars##
FILE * infile; // Open wave file in read mode
FILE * outfile; // Create output ( wave format) file in write mode

int audio_leng;
int sampleRate = 44100; //could be changed but for now sample rate stays hard coded for 41000
int nb;	// variable storing number of bytes returned
int * data_size;

//13230000 is 5 mins @ 44,100 so hard cap on song length at 5 mins
int LENG_MAX = 13230000; //max length of song in samples
short int * left_in;
short int * right_in;

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

int frameSize_out = 22050;

// ##room simulation vars##
int resolution; //if resolution was 100 then it would be per cm
int xLength;
int yLength;
int zLength;
int points;

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

room * roomArr;

struct coefs {
    float neg = 1;
    float pos = 1;
};

coefs coefs_x;
coefs coefs_y;
coefs coefs_z;

//#### sql functions ####
int selectInt(string sql, string valueName){

    int returnVal;

    try {
        sql::Driver *driver;
        sql::Connection *con;
        sql::Statement *stmt;
        sql::ResultSet *res;

        /* Create a connection */
        driver = get_driver_instance();
        con = driver->connect("tcp://127.0.0.1:3306", "webuser", "webuser");
        /* Connect to the MySQL test database */
        con->setSchema("venmodDB");

        stmt = con->createStatement();
        res = stmt->executeQuery(sql);

        while (res -> next()) {
            /* Access column data by alias or column name */
            returnVal = res -> getInt(valueName);
        }

        delete res;
        delete stmt;
        delete con;

    } catch (sql::SQLException &e) {
        cout << "# ERR: SQLException in " << __FILE__;
        cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << endl;
        cout << "# ERR: " << e.what();
        cout << " (MySQL error code: " << e.getErrorCode();
        cout << ", SQLState: " << e.getSQLState() << " )" << endl;
    }
    
    return returnVal;
}

float selectFloat(string sql, string valueName){

    float returnVal;

    try {
        sql::Driver *driver;
        sql::Connection *con;
        sql::Statement *stmt;
        sql::ResultSet *res;

        /* Create a connection */
        driver = get_driver_instance();
        con = driver->connect("tcp://127.0.0.1:3306", "webuser", "webuser");
        /* Connect to the MySQL test database */
        con->setSchema("venmodDB");

        stmt = con->createStatement();
        res = stmt->executeQuery(sql);

        while (res -> next()) {
            /* Access column data by alias or column name */
            returnVal = (float)(res -> getDouble(valueName)); //cast double result to float as cppconn does not have a getFloat()
        }

        delete res;
        delete stmt;
        delete con;

    } catch (sql::SQLException &e) {
        cout << "# ERR: SQLException in " << __FILE__;
        cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << endl;
        cout << "# ERR: " << e.what();
        cout << " (MySQL error code: " << e.getErrorCode();
        cout << ", SQLState: " << e.getSQLState() << " )" << endl;
    }
    
    return returnVal;
}

string selectString(string sql, string valueName){

    string returnVal;

    try {
        sql::Driver *driver;
        sql::Connection *con;
        sql::Statement *stmt;
        sql::ResultSet *res;

        /* Create a connection */
        driver = get_driver_instance();
        con = driver->connect("tcp://127.0.0.1:3306", "webuser", "webuser");
        /* Connect to the MySQL test database */
        con->setSchema("venmodDB");

        stmt = con->createStatement();
        res = stmt->executeQuery(sql);

        while (res -> next()) {
            /* Access column data by alias or column name */
            returnVal = res -> getString(valueName);
        }

        delete res;
        delete stmt;
        delete con;

    } catch (sql::SQLException &e) {
        cout << "# ERR: SQLException in " << __FILE__;
        cout << "(" << __FUNCTION__ << ") on line " << __LINE__ << endl;
        cout << "# ERR: " << e.what();
        cout << " (MySQL error code: " << e.getErrorCode();
        cout << ", SQLState: " << e.getSQLState() << " )" << endl;
    }
    
    return returnVal;
}

void ticketReady(string username){
    //sql to set ready on ticketTable to 1 for username
    try {
        sql::Driver *driver;
        sql::Connection *con;
        sql::Statement *stmt;
        sql::ResultSet *res;

        /* Create a connection */
        driver = get_driver_instance();
        con = driver->connect("tcp://127.0.0.1:3306", "webuser", "webuser");
        /* Connect to the MySQL test database */
        con->setSchema("venmodDB");

        stmt = con->createStatement();
        res = stmt->executeQuery("UPDATE ticketTable SET ready = 1 WHERE username = '" + username + "'");

        delete res;
        delete stmt;
        delete con;

    } catch (sql::SQLException &e) {
        cout << "# ERR: SQLException in " << __FILE__;
        cout << " (" << __FUNCTION__ << ") on line " << __LINE__ << endl;
        cout << "# ERR: " << e.what();
        cout << " (MySQL error code: " << e.getErrorCode();
        cout << ", SQLState: " << e.getSQLState() << " )" << endl;
    }
}


// #### simulation functions ####
void init(string username){

    //dimensions
    xLength = selectInt("SELECT xLength FROM userTable WHERE username = '" + username + "'", "xLength");
    yLength = selectInt("SELECT yLength FROM userTable WHERE username = '" + username + "'", "yLength");
    zLength = selectInt("SELECT zLength FROM userTable WHERE username = '" + username + "'", "zLength");

    //resolution and order 
    resolution = selectInt("SELECT resolution FROM userTable WHERE username = '" + username + "'", "resolution");
    order = selectInt("SELECT reflections FROM userTable WHERE username = '" + username + "'", "reflections");

    //absorption coeficients
    coefs_x.neg = selectFloat("SELECT xNeg FROM userTable WHERE username = '" + username + "'", "xNeg");
    coefs_x.pos = selectFloat("SELECT xPos FROM userTable WHERE username = '" + username + "'", "xPos");
    coefs_y.neg = selectFloat("SELECT yNeg FROM userTable WHERE username = '" + username + "'", "yNeg");
    coefs_y.pos = selectFloat("SELECT yPos FROM userTable WHERE username = '" + username + "'", "yPos");
    coefs_z.neg = selectFloat("SELECT zNeg FROM userTable WHERE username = '" + username + "'", "zNeg");
    coefs_z.pos = selectFloat("SELECT zPos FROM userTable WHERE username = '" + username + "'", "zPos");

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

    cout << "rooms: " << rooms << endl;

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

                // if (roomArr[r].totalAbs != 0){
                //     cout << "room: " << r << " = " << roomArr[r].totalAbs << endl;
                //     cout << " x: " << roomArr[r].pos_x << " y: " << roomArr[r].pos_y << " z: " << roomArr[r].pos_z << endl;
                // }

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

void process(int x, int y, int z, int * monoBuff){ 
    short int * comb_leftBuff;
    short int * comb_rightBuff;

    //one of these for each point
    comb_leftBuff = new short int[audio_leng];
    comb_rightBuff = new short int[audio_leng];

    for (int i = 0; i < audio_leng; i++){
        comb_leftBuff[i] = 0;
        comb_rightBuff[i] = 0;
        monoBuff[i] = 0;
    }

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
            comb_leftBuff[i] += sample / 4;
            comb_rightBuff[i] += sample / 4;
        }
    }

    for (int i = 0; i < audio_leng; i++){
        monoBuff[i] = (comb_leftBuff[i]/2) + (comb_rightBuff[i]/2);
    }

    delete [] comb_leftBuff;
    delete [] comb_rightBuff;

    comb_leftBuff = NULL;
    comb_rightBuff = NULL;
}

//for each point, delay by amount and multiply by wall coefs, do xcor and write vals to file
void writeWav(){

    cout << "writing wav\n";

    //write 'data' identifier to wav file
    char dataTag[4] = {'d', 'a', 't', 'a'};
    fwrite(dataTag, 1, 4, outfile);

    int * monoBuff;
    monoBuff = new int[audio_leng];

    //for testing, to reverse use nested x,y,z loops for co-ord vals
    int x = xLength/2;
    int y = yLength/2;
    int z = (zLength/4) * 3;

    process(x, y, z, monoBuff);

    //begin writing output data and do xcor

    //write dataSize to file
    fwrite(data_size, 1, sizeof(int), outfile);

    //here wav file could be reconstructed (i dont know why its *4 think it should be *2 but 4 works so ???????)
    short int * out;
    out = new short int[audio_leng*4];
    for (int i = 0; i < audio_leng; i++){
        out[i*2] = monoBuff[i];
        out[(i*2) + 1] = monoBuff[i];
    }

    fwrite(out,1,audio_leng*4,outfile);
    delete [] out;
    delete [] monoBuff;
}

void writeVisuHeaders(){

    //write new 'visu' segment in ouput file
    char visuTag[4] = {'v', 'i', 's', 'u'};
    fwrite(visuTag, 1, 4, outfile);

    fwrite(&xLength, 1, 4, outfile);
    fwrite(&yLength, 1, 4, outfile);
    fwrite(&zLength, 1, 4, outfile);
    fwrite(&audio_leng, 1, 4, outfile);
    fwrite(&frameSize_out, 1, 4, outfile);

    char indexTag[4] = {'i', 'd', 'e', 'x'};
    fwrite(indexTag, 1, 4, outfile);
}

void writeVisu(){
    cout << "writing visualiser data\n";

    writeVisuHeaders();

    int * src_monoBuff;
    src_monoBuff = new int[audio_leng];

    int * monoBuff;
    monoBuff = new int[audio_leng];

    int pointNo = 0;

    for (int z = 0; z < zLength; z++){
        for (int y = 0; y < yLength; y++){
            for (int x = 0; x < xLength; x++){

                short int frameNo = 0;
                float xcor = 0; //xcorr value
                float numerator = 0;
                float divisor = 0;
                float src_sum = 0;
                float res_sum = 0;

                // cout << "pointNo: " << pointNo << endl;

                //l&r buffs get filled with processed values
                process(x, y, z, monoBuff);

                //dist to left
                float dist_L = get_dist(x, y, z, speaker_L);
                //dist to right
                float dist_R = get_dist(x, y, z, speaker_R);

                //time delays in samples
                int delay_L_samp = (dist_L/343) * sampleRate;
                int delay_R_samp = (dist_R/343) * sampleRate;

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

                    //add delayed buffs to array
                    src_monoBuff[i] = (sample/2) + (sample/2);
                }

                int n = 0; //point in xcorr frame
                //combine channels to mono
                //split monobuff into 3 sec chunks and do cross corr on it vs combined source
                for (int i = 0; i < audio_leng; i++){

                    //xcorr happens here
                    numerator += src_monoBuff[i] * monoBuff[i];

                    src_sum += pow(src_monoBuff[i], 2);
                    res_sum += pow(monoBuff[i], 2);

                    if (n == frameSize_out){
                        divisor = sqrt(src_sum * res_sum);

                        xcor = numerator / divisor;

                        short int out = xcor * 32767;

                        // cout << "xcor: " << out << endl;

                        //this will be add value to point array when doing gpu accel
                        fwrite(&out, 1, 2, outfile);

                        numerator = 0;
                        divisor = 0;
                        xcor = 0;
                        src_sum = 0;
                        res_sum = 0;
                        n = 0;
                        frameNo++;
                    }

                    n++;
                }

                pointNo++;
                cout << "Point No; " << pointNo << " of " << points << endl;
            }
        }
    }

    delete [] monoBuff;
    delete [] src_monoBuff;
}

// read header and data from input file
void readFile(){
    int frameSize_in = 16384;
    short int buff[16384];
	int count = 0;						                // For counting number of frames in wave file.
	header_p meta = (header_p)malloc(sizeof(header));	// header_p points to a header struct that contains the wave file metadata fields

    char check_buff[1];
    char data_buff[4];

	if (infile)
	{
		fread(meta, 1, sizeof(header), infile);
        // meta->num_channels = 1; //change for mono
		fwrite(meta,1, sizeof(*meta), outfile);

        //testing - prints WAVE
        for (int i = 0; i < 4; i++){
            cout << meta->format[i];
        }
        cout << endl;

		cout << "Size of Header file is "<<sizeof(*meta)<<" bytes" << endl;
		cout << "Sampling rate of the input wave file is "<< meta->sample_rate <<" Hz" << endl;

        //for file read
        char readingData = 0;

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

        data_size = new int[1];
        //read next 4 bytes which is data size
        fread(data_size, 1, sizeof(int), infile);
        audio_leng = data_size[0]/4; //leng in samples: /2 to get short int then /2 to split stereo channels
        // data_size[0] = data_size[0]/2; //half data size if writing mono

        cout << "data size: " << audio_leng << endl;

        left_in = new short int [audio_leng];
        right_in = new short int [audio_leng];

        for (int i = 0; i < audio_leng; i++){
            left_in[i] = 0;
            right_in[i] = 0;
        }

        int sampsWritten = 0; //no samples (short int) written to left and right buffs

        //read file untill end
		while (!feof(infile))
		{
            // Increment Number of frames
			count++;

            // Reading data in chunks of BUFFSIZE
			nb = fread(buff,1,frameSize_in,infile);
            
            //send left and right channels to respective buffers, nb is bytes so /2 to get short ints
            // if (sampsWritten < LENG_MAX)...
            for(int i = 0; i < nb/2; i+=2){
                left_in[sampsWritten] = buff[i];
                right_in[sampsWritten] = buff[i + 1];

                sampsWritten++;
            }
		}

	    cout << "Number of frames in the input wave file are " << count << endl;
	}
}


int main(void){
    cout << "runing digest alg. waiting for ticket...\n";

    while(true){
        int cnt = selectInt("SELECT COUNT(username) AS cnt FROM ticketTable WHERE ready = 0", "cnt");

        if (cnt > 0){
            string username;
            username = selectString("SELECT username FROM ticketTable WHERE ready = 0 LIMIT 1", "username");
            
            cout << "converting file for user: " << username << endl << endl;

            string inPath = "in/" + username ;
            string tempPath = "temp/" + username + ".vis.wav";
            string outPath = "out/" + username + ".vis.wav";

            infile = fopen(inPath.data(),"rb"); // Open wave file in read mode
            outfile = fopen(tempPath.data(),"wb"); // Create output file in write mode

            //initialise params rooms co-ords and speaker position 
            init(username);
            //read file contents and add to input buffers
            readFile();
            //process and write ouput of central point to wav
            writeWav();
            //process, do visu calculations and write them to file
            writeVisu();
            
            delete [] roomArr;
            roomArr = NULL;
            delete [] data_size;
            delete [] left_in;
            left_in = NULL;
            delete [] right_in;
            right_in = NULL;

            fclose(infile);
            fclose(outfile);

            //send ready file to output folder
            rename(tempPath.data(), outPath.data());

            //delete input file
            string path = "in/" + username;
            unlink(path.c_str());

            //set ticket to ready
            ticketReady(username); 

            cout << "completed processing for " + username + "\n";
        }

        sleep_for(seconds(5));
    }

    return EXIT_SUCCESS;
}