#include <string>
#include <stdint.h>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>  // Required for stringstreams
#include <ctime>
#include <cmath>
#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;
template <class NT>
class Dist 
{ 
public: 
	typedef SpDCCols < int, NT > DCCols;
	typedef SpParMat < int, NT, DCCols > MPI_DCCols;
	typedef SpParMat1D < int, NT, DCCols > MPI_DCCols1D;
};
int main(int argc, char ** argv){
    int nprocs, myrank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    string graphtype = string(argv[1]);
    unsigned scale = (unsigned) atoi(argv[2]);
    unsigned EDGEFACTOR = (unsigned) atoi(argv[3]);
    int permute = atoi(argv[4]);
    double initiator[4];
    string permstring = "RandPerm";
    if (permute == 0) permstring = "NoPerm";
    string filename = graphtype + "_scale_" + to_string(scale) + "_EdgeFactor_" + to_string(EDGEFACTOR) + 
    "_" + permstring + ".mtx";
    if(graphtype == string("ER"))
    {
        initiator[0] = .25;
        initiator[1] = .25;
        initiator[2] = .25;
        initiator[3] = .25;
    }
    else if(graphtype == string("G500"))
    {
        initiator[0] = .57;
        initiator[1] = .19;
        initiator[2] = .19;
        initiator[3] = .05;
        EDGEFACTOR  = 16;
    }
    else if(graphtype == string("SSCA"))
    {
        initiator[0] = .6;
        initiator[1] = .4/3;
        initiator[2] = .4/3;
        initiator[3] = .4/3;
        EDGEFACTOR  = 8;
    }
    else {
        if(myrank == 0) printf("The initiator parameter - %s - is not recognized.\n", argv[5]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    {
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
        MPI_Comm comm = MPI_COMM_WORLD;
        DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>(comm);
        ostringstream minfo;
        int nprocs = DEL->commGrid->GetSize();
        minfo << "Started Generation of scale "<< scale << endl;
        minfo << "Using " << nprocs << " MPI processes" << endl;
        SpParHelper::Print(minfo.str());
        DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, permute, false );
        // don't generate packed edges, that function uses MPI_COMM_WORLD which can not be used in a single layer!
        SpParHelper::Print("Generated renamed edge lists\n");
        ostringstream tinfo;
        SpParHelper::Print(tinfo.str());
        Dist<int>::MPI_DCCols *Aread = new Dist<int>::MPI_DCCols(*DEL, false);
        delete DEL;
        Aread->ParallelWriteMM(filename, true);
        delete Aread;
    }
    MPI_Finalize();
    return 0;
}