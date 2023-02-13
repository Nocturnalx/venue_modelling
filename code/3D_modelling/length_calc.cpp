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

int order = 1;
int rooms;

struct room{
    int pos_x;
    int pos_y;
    int pos_z;

    bool mirrored_x = 0;
    bool mirrored_y = 0;
    bool mirrored_z = 0;

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

                roomArr[i].pos_y = y - order;
                roomArr[i].pos_z = z - order;
                roomArr[i].pos_x = x - order;

                if (roomArr[i].pos_x % 2 != 0){
                    roomArr[i].mirrored_x = 1;
                }

                if (roomArr[i].pos_y % 2 != 0){
                    roomArr[i].mirrored_y = 1;
                }

                if (roomArr[i].pos_z % 2 != 0){
                    roomArr[i].mirrored_z = 1;
                }

                // roomArr[i].xOffset = minOffset_x + (xLength * x);
                // roomArr[i].yOffset = minOffset_y + (yLength * y);
                // roomArr[i].zOffset = minOffset_z + (zLength * z);

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

    int x = 80;
    int y = 80;
    int z = 320;

    for (int r = 0; r < rooms; r++){

        cout << "room: " << r + 1 << endl;
        
        int room_x = x + (roomArr[r].pos_x * xLength);
        int room_y = y + (roomArr[r].pos_y * yLength);
        int room_z = z + (roomArr[r].pos_z * zLength);

        if (roomArr[r].mirrored_x){
            room_x = room_x + (2 * ((xLength/2) - x));
            cout << "x mirrored\n";
        }

        if (roomArr[r].mirrored_y){
            room_y = room_y + (2 * ((yLength/2) - y));
            cout << "y mirrored\n";
        }

        if (roomArr[r].mirrored_z){
            room_z = room_z + (2 * ((zLength/2) - z));
            cout << "z mirrored\n";
        }

        cout << "final room_x: " << room_x;
        cout << " final room_y: " << room_y;
        cout << " final room_z: " << room_z << endl;

        //dist to left
        float dist_L = get_delay(room_x, room_y, room_z, speaker_L);
        //dist to right
        float dist_R = get_delay(room_x, room_y, room_z, speaker_R);

        //apply delay to buff, times all by abs, combine with previous buff
        
        cout << "left: " << dist_L << " right: " << dist_R << endl << endl;
    }

    delete [] pointArray;
    delete [] roomArr;
    return 0;
}