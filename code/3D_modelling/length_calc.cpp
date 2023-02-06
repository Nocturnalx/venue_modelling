#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>

using namespace std;

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

int order = 2;
int rooms;

struct room{
    int xOffset;
    int yOffset;
    int zOffset;

    float totalAbsorption;
};

room* roomArr;

void init(){
    xLength = xLength * resolution;
    yLength = yLength * resolution;
    zLength = zLength * resolution;
    points = xLength * yLength * zLength;
    pointArray = new float[points]; //array that will contain the index score for each point

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

    //applying x,y,z offsets for each room
    int i = 0;
    for(int y = 0; y < rooms_y; y++){
        for(int z = 0; z < rooms_z; z++){
            for(int x = 0; x < rooms_x; x++){

                int pos_y = y - order;
                int pos_z = z - order;
                int pos_z = z - order;

                if (x % 2 == 0){
                    
                }

                if (y % 2 == 0){

                }

                if (z % 2 == 0){

                }

                roomArr[i].xOffset = minOffset_x + (xLength * x);
                roomArr[i].yOffset = minOffset_y + (yLength * y);
                roomArr[i].zOffset = minOffset_z + (zLength * z);

                

                i++;
            }
        }
    }
}

float get_delay(int x, int y, int z, point speaker){

    float dist = sqrt(pow(x-speaker.x, 2)+pow(y-speaker.y, 2)+pow(z-speaker.z, 2)) / resolution;

    return dist;
}

int main(){

    init();

    //speaker positioning
    speaker_L.x = xLength/4;
    speaker_L.y = yLength/2;
    speaker_L.z = 1;

    speaker_R.x = (xLength/4) * 3;
    speaker_R.y = yLength/2;
    speaker_R.z = 1;

    int x = xLength/2;
    int y = yLength/2;
    int z = (zLength/4) * 3;

    for (int r = 0; r < rooms; r++){
        
        int room_x = x + roomArr[r].xOffset;
        int room_y = y + roomArr[r].yOffset;
        int room_z = z + roomArr[r].zOffset;

        //dist to left
        float dist_L = get_delay(room_x, room_y, room_z, speaker_L);
        //dist to right
        float dist_R = get_delay(room_x, room_y, room_z, speaker_R);

        //apply delay to buff, times all by abs, combine with previous buff
        
        cout << "room: " << r + 1 << endl;
        cout << "left: " << dist_L << " right: " << dist_R << endl;
    }

    delete [] pointArray;
    delete [] roomArr;
    return 0;
}