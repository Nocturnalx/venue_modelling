#pragma once

#include <fstream>
#include "AudioProc.h"
#include "FileHandlerStructs.h"

class FileHandler {

    private:
        /* data */
        FILE * m_infile; // Open wave file in read mode
        FILE * m_outfile; // Create output ( wave format) file in write mode

        std::string m_inPath;
        std::string m_tempPath;
        std::string m_outPath;

        std::string m_username;

        int m_dataSize; //wav specific data size - size in bytes

        header_p m_meta;

        void writeVISUHeaders(std::unique_ptr<SpatialProc> & AProc);

    public:
        FileHandler(std::string username);
        ~FileHandler();

        read_err_codes readFile(std::unique_ptr<SpatialProc> & AProc);

        void writeWav(std::unique_ptr<SpatialProc> & AProc);

        void writeVISU(std::unique_ptr<SpatialProc> & AProc);

        void finalise();

        void writeErr(read_err_codes err_code);
};
