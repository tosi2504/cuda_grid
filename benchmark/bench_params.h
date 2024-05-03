#pragma once

#include <unistd.h>
#include <getopt.h>
#include <string>
#include <stdexcept>


#ifndef BENCH_PARAM_T
	#pragma message "WARNING: BENCH_PARAM_T not defined!"
    #define BENCH_PARAM_T realF
#endif

#ifndef BENCH_PARAM_N
	#pragma message "WARNING: BENCH_PARAM_N not defined!"
    #define BENCH_PARAM_N 64
#endif

#ifndef BENCH_PARAM_numRHS
	#pragma message "WARNING: BENCH_PARAM_numRHS not defined!"
    #define BENCH_PARAM_numRHS 12
#endif

#ifndef BENCH_PARAM_blkSize
    #pragma message "WARNING: BENCH_PARAM_blkSize not defined!"
    #define BENCH_PARAM_blkSize 128
#endif


void parseArgs(int argc, char * argv[], unsigned * Lx, unsigned * Ly, unsigned * Lz, unsigned * Lt, unsigned * mu, bool * isForward) {
    if (argc != 7) {
        throw std::invalid_argument("Wrong number of arguments given to binary");
    }
    std::string sLx(argv[1]);
    std::string sLy(argv[2]);
    std::string sLz(argv[3]);
    std::string sLt(argv[4]);

    std::string smu(argv[5]);
    std::string sisForward(argv[6]);

    // what now? -> get the unsigned vals
    *Lx = std::stoul(sLx);
    *Ly = std::stoul(sLy);
    *Lz = std::stoul(sLz);
    *Lt = std::stoul(sLt);

    *mu = std::stoul(smu);
    *isForward = (!strcmp(argv[6],"true")) ? true : false;
}
