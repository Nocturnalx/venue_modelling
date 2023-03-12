#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>

using namespace std;

const float pi = 3.141592;

int resolution; //if resolution was 100 then it would be per cm
int xLength; //get x,y or z from the actual 3D model then times by resolution (so "10" should be model.getlength() or whatever)
int yLength;
int zLength;
int points;
int reflections;
int waves;

//float* freqArray;
float freqArray[7] = {440.0, 493.88, 554.37, 587.33, 659.25, 739.99, 830.61};
int nfreqs = sizeof(freqArray) / 4; // float is 4 bytes long and sizeof is total bytes in memory so length is size/4

float* pointArray;

void initArr(){
    cout << "xLength: ";
    cin >> xLength;
    cout << "yLength: ";
    cin >> yLength;
    cout << "zLength: ";
    cin >> zLength;

    cout << "resolution: ";
    cin >> resolution;
    cout << "reflections: ";
    cin >> reflections;
    cout << "\n";

    //this isnt right but i need to find the right formula
    waves = reflections;

    xLength = xLength * resolution;
    yLength = yLength * resolution;
    zLength = zLength * resolution;
    points = xLength * yLength * zLength;
    pointArray = new float[points];

    //freqArray = whatever;
}

//useless - for testing - outputs random numbers
float* getTimes(int x, int y, int z){
    float* timeArray;
    timeArray = new float[waves];

    //for testing
    float t = 0;

    //do some trig to find distances and therefore timings
    //this gives time for 1m, 2m, 3m etc
    for(int i = 0; i < waves; i++){
        t = i + 1;
        t = t / 343;
        timeArray[i] = t;

        cout << t << "\n";
    }

    return timeArray;
}

void getAmp(int pointIndex, float* timeArray){
    float index = 0;

    float ampArray[nfreqs];

    //get val for each frequency and add to ampArray
    for(int i = 0; i < nfreqs; i++){
        float val = 0;
        float f = freqArray[i];

        //loss = 1/(4*pi*(r^2)), r = v*d, v = 343m/s, 343*343 = 117,549
        const float invSq = 4*pi*117549;
        //this is outside for loop due to lack of absorbtion coeficients
        val = sin(2*pi*f*timeArray[0]) / (invSq * timeArray[0] * timeArray[0]);

        //add sin waves for each wave that hits point
        for(int n = 1; n < waves; n++){
            float D = timeArray[0] - timeArray[n];

            val = val + (sin(2*pi*f*D) / (invSq * timeArray[n] * timeArray[n]));

            //absorption coeficient of the walls hit
            // cArray = {c1, c2, c3};
            // val = val * c1 * c2 * c3
        }

        ampArray[i] = val;
    }

    //set first value as min and max then compare root(square) to get abs magnitude
    float max = sqrt(ampArray[0]*ampArray[0]);
    float min = max;

    for(int i = 1; i < nfreqs; i++){
        float current = sqrt(ampArray[i] * ampArray[i]);

        if (current > max){max = current;}
        if (current < min){min = current;}
    }

    index = max - min; //not final index but this is difference between loudest and quietest freq you hear at a point

    pointArray[pointIndex] = index;
    cout << "\npoint: " << pointIndex << " index: " << index << "\n";
    cout << "min: " << min << " max: " << max << "\n";
}

int main(){
    //float array[points];
    initArr();

    int i = 0;
 
    for (int x = 0; x < xLength; x++){
        for (int y = 0; y < yLength; y++){
            for (int z = 0; z < zLength; z++){
                //in new thread
                // thread thread(getVal, i, x, y, z);
                // thread.join();

                getAmp(i, getTimes(x,y,z));

                i++;
            }
        }
    }

    cout << "\npoints: " << points << "\n";
    cout << "x: " << xLength << "\n";
    cout << "y: " << yLength << "\n";
    cout << "z: " << zLength << "\n";

    delete [] pointArray;
    return 0;
}