#pragma once

#include <string.h>
#include <iostream>

#include "AudioProcStructs.h"

#include "SQLUtils.h"

#include "SpatialProcParent.h"
#include "XCorrProcParent.h"

class SpatialProc : public SpatialProcParent {
    private:
        /* data */

        void bake();

    public:

        SpatialProc(/* args */);
        ~SpatialProc();

        void init(std::string username);

        float get_dist(int x, int y, int z, point speaker);

        void process(int x, int y, int z);
};

class XCorrProc : public XCorrProcParent {
    
    private:


    public:

        XCorrProc();
        ~XCorrProc();

        void fillSourceBuffer(int x, int y, int z, std::unique_ptr<SpatialProc> & SProc);

        void processXcorr(std::unique_ptr<SpatialProc> & SProc, FILE * outfile);
};