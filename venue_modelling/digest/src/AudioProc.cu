#include "AudioProc.h"

//keep these in here as compiler can get confused and not know where to look for these if using g++ for header file
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void combine(short int *d_monoBuff, short int *d_left_in, short int *d_right_in, float abs, int delay_l, int delay_r, int n){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    //thread id must be less than array length
    if (tid < n){
        short int sample_l = 0;
        //if larger than delay start reading the audio
        if (tid >= delay_l){
            sample_l = d_left_in[tid - delay_l];
            sample_l *= abs; // absorption loss
            sample_l = sample_l / 2; //average of both audio streams for combining to mono
        }
        d_monoBuff[tid] += sample_l; 

        short int sample_r = 0;
        //if larger than delay start reading the audio
        if (tid >= delay_r){
            sample_r = d_right_in[tid - delay_r];
            sample_r = sample_r * abs; // absorption loss
            sample_r = sample_r / 2; //average of both audio streams for combining to mono
        }
        d_monoBuff[tid] += sample_r; 
    }
}

//spatial processor
SpatialProc::SpatialProc(/* args */){

}

SpatialProc::~SpatialProc(){
    
    //will need to use the destructors for audio pointers
    //or use shared smart pointers for the audio
    //wait destruct in main loop right?
    //or it will be a double free

    //also make room arr smart pointer

    deviceFreeMono();
    deviceFreeStereo();
    hostFreeMono();
    hostFreeStereo();
}


void SpatialProc::process(int x, int y, int z){

    //means that everything is set (audioleng, smaple rate, buffs) ie fileread check bool
    if (!fileRead){

        std::cout << "process was called before a file was read in!";
        return;
    }

    //reset monobuff then copy over to device
    for (int i = 0; i < m_audioLeng; i++){
        monoBuff[i] = 0;
    }
    cudaMemcpy(d_monoBuff, monoBuff, sizeof(short int) * m_audioLeng, cudaMemcpyHostToDevice); //copies 

    //populate room buffer arrays
    for (int r = 0; r < m_rooms; r++){

        //get virtual coords of point
        int room_x = x + (m_roomArr[r].pos_x * m_xLength);
        int room_y = y + (m_roomArr[r].pos_y * m_yLength);
        int room_z = z + (m_roomArr[r].pos_z * m_zLength);
        
        //if room mirrored in dimension, add extra difference to get to the actual coord
        if (m_roomArr[r].mirrored_x){
            room_x = room_x + (2 * ((m_xLength/2) - x));
        }

        if (m_roomArr[r].mirrored_y){
            room_y = room_y + (2 * ((m_yLength/2) - y));
        }

        if (m_roomArr[r].mirrored_z){
            room_z = room_z + (2 * ((m_zLength/2) - z));
        }

        //dist to left
        float dist_L = get_dist(room_x, room_y, room_z, m_speaker_L);
        //dist to right
        float dist_R = get_dist(room_x, room_y, room_z, m_speaker_R);

        //time delays in samples
        int delay_L_samp = (dist_L/343) * m_sampleRate;
        int delay_R_samp = (dist_R/343) * m_sampleRate;

        // Executing kernel 
        int block_size = 256;
        int grid_size = ((m_audioLeng + block_size) / block_size); //add extra 256 to N so that when dividing it will round down to > required threads
        combine<<<grid_size,block_size>>>(d_monoBuff, d_left_in, d_right_in, m_roomArr[r].totalAbs / m_gain, delay_L_samp, delay_R_samp, m_audioLeng);
    }

    //no need to syncronise as memcpy has blocking built in

    cudaMemcpy(monoBuff, d_monoBuff, sizeof(short int) * m_audioLeng, cudaMemcpyDeviceToHost);

}


void SpatialProc::init(std::string username){

    //move all this somewhere else and just init all this in the constructor then call bake

    //dimensions
    m_xLength = selectInt("SELECT xLength FROM userTable WHERE username = '" + username + "'", "xLength");
    m_yLength = selectInt("SELECT yLength FROM userTable WHERE username = '" + username + "'", "yLength");
    m_zLength = selectInt("SELECT zLength FROM userTable WHERE username = '" + username + "'", "zLength");

    //resolution and order 
    m_resolution = selectInt("SELECT resolution FROM userTable WHERE username = '" + username + "'", "resolution");
    m_order = selectInt("SELECT reflections FROM userTable WHERE username = '" + username + "'", "reflections");

    //absorption coeficients
    m_coefs_x.neg = selectFloat("SELECT xNeg FROM userTable WHERE username = '" + username + "'", "xNeg");
    m_coefs_x.pos = selectFloat("SELECT xPos FROM userTable WHERE username = '" + username + "'", "xPos");
    m_coefs_y.neg = selectFloat("SELECT yNeg FROM userTable WHERE username = '" + username + "'", "yNeg");
    m_coefs_y.pos = selectFloat("SELECT yPos FROM userTable WHERE username = '" + username + "'", "yPos");
    m_coefs_z.neg = selectFloat("SELECT zNeg FROM userTable WHERE username = '" + username + "'", "zNeg");
    m_coefs_z.pos = selectFloat("SELECT zPos FROM userTable WHERE username = '" + username + "'", "zPos");

    bake();
}

//calculate distance between two points
float SpatialProc::get_dist(int x, int y, int z, point speaker){

    float dist = sqrt(pow(x-speaker.x, 2)+pow(y-speaker.y, 2)+pow(z-speaker.z, 2)) / m_resolution;

    return dist;
}

void SpatialProc::bake(){

    //init dimensions and point num
    m_xLength = m_xLength * m_resolution;
    m_yLength = m_yLength * m_resolution;
    m_zLength = m_zLength * m_resolution;
    m_points = m_xLength * m_yLength * m_zLength;

    //init rooms
    //reflections is extra on the border, this adds both sides of border plus middle to get total dimension length
    int rooms_x = (m_order * 2) + 1;
    int rooms_y = (m_order * 2) + 1;
    int rooms_z = (m_order * 2) + 1;

    //total rooms is x*y*z
    m_rooms = rooms_x * rooms_y * rooms_z;

    m_roomArr = new room[m_rooms]; 

    std::cout << "rooms: " << m_rooms << std::endl;

    m_gain = ceil(((rooms_x*m_coefs_x.neg*m_coefs_x.pos)+(rooms_y*m_coefs_y.neg*m_coefs_y.pos)+(rooms_y*m_coefs_y.neg*m_coefs_y.pos))/3);

    //calculate total abs coeffs and check if room is mirrored
    int r = 0;
    for(int y = 0; y < rooms_y; y++){
        for(int z = 0; z < rooms_z; z++){
            for(int x = 0; x < rooms_x; x++){

                m_roomArr[r].pos_x = x - m_order;
                m_roomArr[r].pos_y = y - m_order;
                m_roomArr[r].pos_z = z - m_order;

                int absolute_x = abs(m_roomArr[r].pos_x);
                int absolute_y = abs(m_roomArr[r].pos_y);
                int absolute_z = abs(m_roomArr[r].pos_z);

                if (m_roomArr[r].pos_x % 2 != 0){
                    //is mirrored in this axis
                    m_roomArr[r].mirrored_x = 1;

                    float extra_coef = 1;
                    //extra coef on the end changes whether going in positive or negative direction
                    if (m_roomArr[r].pos_x < 0){
                        extra_coef *= m_coefs_x.pos;
                    } else {
                        extra_coef *= m_coefs_x.neg;
                    }

                    m_roomArr[r].totalAbs *= pow(m_coefs_x.neg, (absolute_x - 1) / 2) * pow(m_coefs_x.pos, (absolute_x - 1) / 2) * extra_coef;
                } else {
                    m_roomArr[r].totalAbs *= pow(m_coefs_x.neg, absolute_x / 2) * pow(m_coefs_x.pos, absolute_x / 2);
                }

                if (m_roomArr[r].pos_y % 2 != 0){
                    m_roomArr[r].mirrored_y = 1;

                    float extra_coef = 1;
                    //extra coef on the end changes whether going in positive or negative direction
                    if (m_roomArr[r].pos_y < 0){
                        extra_coef *= m_coefs_y.pos;
                    } else {
                        extra_coef *= m_coefs_y.neg;
                    }

                    m_roomArr[r].totalAbs *= pow(m_coefs_y.neg, (absolute_y - 1) / 2) * pow(m_coefs_y.pos, (absolute_y - 1) / 2) * extra_coef;
                } else {
                    m_roomArr[r].totalAbs *= pow(m_coefs_y.neg, absolute_y / 2) * pow(m_coefs_y.pos, absolute_y / 2);
                }

                if (m_roomArr[r].pos_z % 2 != 0){
                    m_roomArr[r].mirrored_z = 1;

                    float extra_coef = 1;
                    //extra coef on the end changes whether going in positive or negative direction
                    if (m_roomArr[r].pos_z < 0){
                        extra_coef *= m_coefs_z.pos;
                    } else {
                        extra_coef *= m_coefs_z.neg;
                    }

                    m_roomArr[r].totalAbs *= pow(m_coefs_z.neg, (absolute_z - 1) / 2) * pow(m_coefs_z.pos, (absolute_z - 1) / 2) * extra_coef;
                } else {
                    m_roomArr[r].totalAbs *= pow(m_coefs_z.neg, absolute_z / 2) * pow(m_coefs_z.pos, absolute_z / 2);
                }

                r++;
            }
        }
    }

    //speaker positioning
    m_speaker_L.x = m_xLength/4;
    m_speaker_L.y = m_yLength/2;
    m_speaker_L.z = 1;

    m_speaker_R.x = (m_xLength/4) * 3;
    m_speaker_R.y = m_yLength/2;
    m_speaker_R.z = 1;
}


//XCorr processor
XCorrProc::XCorrProc(){

}

XCorrProc::~XCorrProc(){

    hostFreeSrc();
    deviceFreeSrc();
}

void XCorrProc::fillSourceBuffer(int x, int y, int z, std::unique_ptr<SpatialProc> & SProc){

    //delay audio for first sonic impact

    //dist to left
    float dist_L = SProc->get_dist(x, y, z, SProc->getSpeakerPosition(0));
    //dist to right
    float dist_R = SProc->get_dist(x, y, z, SProc->getSpeakerPosition(1));

    //time delays in samples
    int delay_L_samp = (dist_L/343) * SProc->getSampleRate();
    int delay_R_samp = (dist_R/343) * SProc->getSampleRate();

    // Executing kernel 
    int block_size = 256;
    int grid_size = ((SProc->getAudioLeng() + block_size) / block_size); //add extra 256 to N so that when dividing it will round down to > required threads
    combine<<<grid_size,block_size>>>(m_d_srcMonoBuff, SProc->d_left_in, SProc->d_right_in, 1, delay_L_samp, delay_R_samp, SProc->getAudioLeng()); //reusing combine func for central virt room with abs=1

    cudaMemcpy(m_srcMonoBuff, m_d_srcMonoBuff, sizeof(short int) * SProc->getAudioLeng(), cudaMemcpyDeviceToHost);
}

void XCorrProc::processXcorr(std::unique_ptr<SpatialProc> & SProc, FILE * outfile){

    int n = 0; //point in xcorr frame
    //split monobuff into 3 sec chunks and do cross corr on it vs combined source
    for (int i = 0; i < SProc->getAudioLeng(); i++){

        //xcorr happens here
        numerator += m_srcMonoBuff[i] * SProc->monoBuff[i];

        src_sum += pow(m_srcMonoBuff[i], 2);
        res_sum += pow(SProc->monoBuff[i], 2);

        if (n == SProc->getFrameSize()){
            divisor = sqrt(src_sum * res_sum);

            xcor = numerator / divisor;

            short int out = xcor * 32767;

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
}