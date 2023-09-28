#include "FileHandler.h"

FileHandler::FileHandler(std::string username){

    m_username = username;
    
    // m_inPath = "/var/venue_modelling/digest/in/" + username ;
    // m_tempPath = "/var/venue_modelling/digest/temp/" + username + ".vwav";
    // m_outPath = "/var/venue_modelling/digest/out/" + username + ".vwav";

    std::string homeDir = getenv("HOME");

    m_inPath = homeDir + "/.venue_modelling/digest/in/" + username ;
    m_tempPath = homeDir + "/.venue_modelling/digest/temp/" + username + ".vwav";
    m_outPath = homeDir + "/.venue_modelling/digest/out/" + username + ".vwav";

    m_infile = fopen(m_inPath.data(),"r"); // Open wave file in read mode
}


FileHandler::~FileHandler(){


}

read_err_codes FileHandler::readFile(std::unique_ptr<SpatialProc> & SProc){

    int frameSize_in = 16384; // no of bytes read in at a time from file
    short int buff[16384];
	int count = 0;						        // For counting number of frames in wave file.
	m_meta = (header_p)malloc(sizeof(header));	// header_p points to a header struct that contains the wave file metadata fields

    char check_buff[1];
    char data_buff[4];

    read_err_codes err_code;
    err_code = miscErr; // 0 is good read, 1 is incorrect format (16 bit, stereo, PCM), and 2 is bad file (no WAVE or data chunk), 3 is defualt so undefined err

	if (m_infile){
        fread(m_meta, 1, sizeof(header), m_infile);

        char WAVE[4] = {'W','A','V','E'};

        //testing - prints WAVE if its a wav file else returns bad read
        for (int i = 0; i < 4; i++){
            char ch = m_meta->format[i];
            if (ch == WAVE[i]){
                std::cout << ch;
            } else {
                std::cout << std::endl;

                err_code = badFile;
                return badFile;
            }
        }

        std::cout << std::endl;

        //check bit size
        if (m_meta->bits_per_sample != 16 || m_meta->num_channels != 2 || m_meta->audio_format != 1){
            err_code = wrongFormat;
            return err_code;
        }

		std::cout << "Sampling rate of the input wave file is "<< m_meta->sample_rate <<" Hz" << std::endl;

        int sampleRate = m_meta->sample_rate;
        int frameSize = sampleRate/20;
        SProc->setSampleRate(sampleRate);
        SProc->setFrameSize(frameSize);

        //for file read
        bool dataTagFound = false;

        //checking for data tag
        for (int i = 0; i < 1500; i++){
            fread(check_buff, 1, sizeof(char), m_infile);
            // fwrite(check_buff, 1, sizeof(char), outfile);

            for (int i = 0; i < 3; i++){
                data_buff[i] = data_buff[i + 1];
            }

            data_buff[3] = check_buff[0];

            //there must be a better way to do this?????
            if (data_buff[3] == 'a' && data_buff[2] == 't' && data_buff[1] == 'a' && data_buff[0] == 'd'){
                dataTagFound = true;
                break;
            }
        }

        //if data tag is found begin read
        if (dataTagFound){
            std::cout << "data found!\n";

            //read next 4 bytes which is data size
            fread(&m_dataSize, 1, sizeof(int), m_infile);
            int audioLeng = m_dataSize/4;
            SProc->setAudioLeng(audioLeng); //leng in samples: /2 to get short int then /2 to split stereo channels
            
            std::cout << "audio leng: " << audioLeng << std::endl;

            //now audioleng is set we can allocate memory
            SProc->hostMallocMono();    //allocate output buff on device and host
            SProc->deviceMallocMono();

            SProc->hostMallocStereo();  //allocate buffers to read file into
            SProc->deviceMallocStereo();    //allocate device buffers to copy into

            for (int i = 0; i < audioLeng; i++){
                SProc->left_in[i] = 0;
                SProc->right_in[i] = 0;
            }

            int sampsWritten = 0; //no samples (short int) written to left and right buffs

            //read file untill end
            while (!feof(m_infile))
            {
                // Increment Number of frames
                count++;

                int numBytes;

                // Reading data in chunks of BUFFSIZE
                numBytes = fread(buff,1,frameSize_in,m_infile);

                //send left and right channels to respective buffers, numBytes is bytes so /2 to get short ints
                // if (sampsWritten < LENG_MAX)...
                for(int i = 0; i < numBytes/2; i+=2){
                    SProc->left_in[sampsWritten] = buff[i];
                    SProc->right_in[sampsWritten] = buff[i + 1];

                    sampsWritten++;
                }
            }
            
            err_code = readSuccess;

            SProc->copyStereoToDevice(); //copies l+r in arrays to device arrays
        }
	}

    if (err_code == readSuccess){
        SProc->fileRead = true;
    }

    //close and delete input file
    fclose(m_infile);

    //delete input file
    // std::string path = "~/.venue_modelling/digest/in/" + m_username;
    // unlink(path.c_str());

    //open output file
    m_outfile = fopen(m_tempPath.data(),"w"); // Create output file in write mode

    return err_code;
}


void FileHandler::writeWav(std::unique_ptr<SpatialProc> & SProc){

    int audio_leng = SProc->getAudioLeng();

    dimensions dims = SProc->getDimensions();

    int xLength = dims.xLength;
    int yLength = dims.yLength;
    int zLength = dims.zLength;

    std::cout << "writing wav\n";

    m_meta->num_channels = 1; //change for mono
    fwrite(m_meta, 1, sizeof(*m_meta), m_outfile); 

    //write 'data' identifier to wav file
    char dataTag[4] = {'d', 'a', 't', 'a'};
    fwrite(dataTag, 1, 4, m_outfile);

    //write dataSize to file
    int out_dataSize = m_dataSize/2; //half data size if writing mono
    fwrite(&out_dataSize, 1, sizeof(int), m_outfile);

    //coords of a typical central audience listening location
    int x = xLength/2;
    int y = yLength/2;
    int z = (zLength/4) * 3;

    //fills combined mono buff with calcualted audio values
    SProc->process(x, y, z);

    //write audio, doing mono means output audio is just monobuff
    fwrite(SProc->monoBuff, 1, audio_leng * 2, m_outfile);
}

void FileHandler::writeVISUHeaders(std::unique_ptr<SpatialProc> & SProc){
    
    std::cout << "writing visu headers\n";

    //write new 'visu' segment in ouput file
    char visuTag[4] = {'v', 'i', 's', 'u'};
    fwrite(visuTag, 1, 4, m_outfile);

    dimensions dims = SProc->getDimensions();
    int audioLeng = SProc->getAudioLeng();
    int frameSize = SProc->getFrameSize();

    fwrite(&dims.xLength, 1, 4, m_outfile);
    fwrite(&dims.yLength, 1, 4, m_outfile);
    fwrite(&dims.zLength, 1, 4, m_outfile);
    fwrite(&audioLeng, 1, 4, m_outfile);
    fwrite(&frameSize, 1, 4, m_outfile);

    char indexTag[4] = {'i', 'd', 'e', 'x'};
    fwrite(indexTag, 1, 4, m_outfile);
}

void FileHandler::writeVISU(std::unique_ptr<SpatialProc> & SProc){
    std::cout << "writing visualiser data\n";

    writeVISUHeaders(SProc);

    std::unique_ptr<XCorrProc> XCProc = std::make_unique<XCorrProc>();

    //init mem buffers
    XCProc->hostMallocSrc(SProc->getAudioLeng());
    XCProc->deviceMallocSrc(SProc->getAudioLeng());

    dimensions dims = SProc->getDimensions();

    int points = dims.xLength * dims.yLength * dims.zLength;
    int pointNo = 0;

    for (int z = 0; z < dims.zLength; z++){
        for (int y = 0; y < dims.yLength; y++){
            for (int x = 0; x < dims.xLength; x++){

                //fills combined mono buff with calcualted audio values
                SProc->process(x, y, z);

                //calculate audio for first ray to use as reference for xcorr
                XCProc->fillSourceBuffer(x, y, z, SProc);

                //do xcorr
                XCProc->processXcorr(SProc, m_outfile);

                pointNo++;
                std::cout << "Point No; " << pointNo << " of " << points << std::endl;
            }
        }
    }
}

//move file from temp folder to output file
void FileHandler::finalise(){

    //close outfile
    fclose(m_outfile);

    //send ready file to output folder
    rename(m_tempPath.data(), m_outPath.data());
}

void FileHandler::writeErr(read_err_codes err_code){

    std::cout << "error: " << (short int)err_code << std::endl; // cast because char wasnt showing

    if (err_code == wrongFormat){
        const char *str = "Incorrect format this program requires 16 bit, stereo, PCM, wav.";
        fwrite(str,1, sizeof(char) * 64, m_outfile);
    } else if (err_code == badFile){
        const char *str = "There is a problem with the file, it is either not a wav file or it has been corrupted.";
        fwrite(str,1, sizeof(char) * 87, m_outfile);
    } else {
        const char *str = "Undefined error, please double check the file you are uploading.";
        fwrite(str,1, sizeof(char) * 64, m_outfile);
    }
}