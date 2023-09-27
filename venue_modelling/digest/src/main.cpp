#include "SQLUtils.h"
#include "FileHandler.h"
#include "AudioProc.h"

#include <iostream>
#include <string.h>
#include <chrono>
#include <thread>

#include <memory.h>

using namespace std::this_thread;
using namespace std::chrono;

int main(int argc, char** argv){

    int running = 1;
    
    std::cout << "running digest alg. waiting for ticket...\n";

    while (running){

       //look for new ticket
        int cnt = selectInt("SELECT COUNT(username) AS cnt FROM ticketTable WHERE ready = 0", "cnt"); 

        if (cnt > 0){

            std::string username;
            username = selectString("SELECT username FROM ticketTable WHERE ready = 0 LIMIT 1", "username"); //get top ticket from pile

            std::cout << "converting file for user: " << username << std::endl << std::endl;

            std::unique_ptr<AudioProc> AProc = std::make_unique<AudioProc>();
            std::unique_ptr<FileHandler> fileHandler = std::make_unique<FileHandler>(username);

            AProc->init(username);

            read_err_codes err_code = fileHandler->readFile(AProc);

            if (err_code == readSuccess){
                fileHandler->writeWav(AProc);
                // fileHandler->writeVISU(AProc);

                std::cout << "completed processing for " + username + "\n";
                
            } else {

                fileHandler->writeErr(err_code);
            }

            //move file to output
            fileHandler->finalise();

            //set ticket to ready
            // ticketReady(username);

            //audio proc will destroy in/out memory on destructing
        }
        
        running--; //testing

        sleep_for(seconds(5));
    }

    return EXIT_SUCCESS;
}