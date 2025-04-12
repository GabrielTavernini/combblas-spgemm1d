#ifndef _COMM_GRID_1D_H_
#define _COMM_GRID_1D_H_

#include <cstdint>
#include <iostream>
#include <cmath>
#include <cassert>
#include <mpi.h>
#include <sstream>
#include <string>
#include <fstream>
#include <stdint.h>
#include "MPIType.h"

namespace combblas {

class CommGrid1D
{
public:
    CommGrid1D(MPI_Comm world)
    {
        MPI_Comm_dup(world, &commWorld);
        MPI_Comm_rank(commWorld, &myrank);
        MPI_Comm_size(commWorld,&worldsize);
    }

    ~CommGrid1D(){MPI_Comm_free(&commWorld);}
    CommGrid1D (const CommGrid1D & rhs)
    :worldsize(rhs.worldsize),myrank(rhs.myrank) // copy constructor
    {MPI_Comm_dup(rhs.commWorld, &commWorld);}

    CommGrid1D & operator=(const CommGrid1D & rhs)	// assignment operator
    {
        if(this != &rhs)		
        {
            MPI_Comm_free(&commWorld);
            myrank = rhs.myrank;
            worldsize = rhs.worldsize;
            MPI_Comm_dup(rhs.commWorld, &commWorld);
        }
        return *this;
    }
    bool operator== (const CommGrid1D & rhs) const {
        int result;
        MPI_Comm_compare(commWorld, rhs.commWorld, &result);
        if ((result != MPI_IDENT) && (result != MPI_CONGRUENT))
        {
            // A call to MPI::Comm::Compare after MPI::Comm::Dup returns MPI_CONGRUENT
            // MPI::CONGRUENT means the communicators have the same group members, in the same order
            return false;
        }
        return true;
    }
    bool operator!= (const CommGrid1D & rhs) const {return (! (*this == rhs));}
    int GetRank(){return myrank;}
    int GetSize(){return worldsize;}
    MPI_Comm GetWorld() const { return commWorld; }
private:
    MPI_Comm commWorld;
    int myrank;
    int worldsize;


    template <class IT, class NT, class DER>
    friend class SpParMat;
    template <class IT, class NT, class DER>
    friend class SpParMat1D;

    template <class IT, class NT>
    friend class FullyDistSpVec;
};

}

#endif
