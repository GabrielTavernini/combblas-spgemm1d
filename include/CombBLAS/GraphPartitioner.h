#ifndef GRAPH_PARTITIONER_H
#define GRAPH_PARTITIONER_H

#include <string>
#include <vector>
#include "SpParMat.h"
#include "SpParMat1D.h"
#include <parmetis.h>


namespace combblas {

enum GraphPartitioner{
    METIS_KWAY,                //   METIS    V3 PartKway
    METIS_GEOMKWAY,            //   METIS    V3 PartGeomKway
    PARMETIS_KWAY,             //   ParMETIS V3 PartKway
    PARMETIS_GEOMKWAY,         //   ParMETIS V3 PartGeomKway
};

class GraphPartition{

// partition a SpParMat1D matrix into nparts using ParMETIS API METIS ParMETIS_V3_PartKway API
// 1. Input A must be symmetric! For the performance reason, it won't check this feature.
static void ParmetisGraphPartition(SpParMat1D<int64_t, double, SpDCCols<int64_t, double>> & A, int nparts);

};

inline void GraphPartition::ParmetisGraphPartition(SpParMat1D<int64_t, double, SpDCCols<int64_t, double>> & A, int nparts){
    int64_t Anrows = A.getnrow();
    int64_t Ancols = A.getncol();
    
}


// int64_t *vtxdist, int64_t *xadj, int64_t* adjncy, int64_t *vwgt, int64_t *adjwgt, int64_t *wgtflag,
//     int64_t *numflag, int64_t *ncon, int64_t *nparts, double *tpwgts, double *ubvec, int64_t *options,
//     int64_t *edgecut, int64_t *part, MPI_Comm *comm)
// // we compile the index type to be 64 and value type to be 64.
// // so perhaps no need for template.
// // we want to get permute array from this class!
// // input matrix should be symmetric!
// class GraphPartitioner{
//     typedef int64_t IT;
//     typedef double  NT;
// protected: // data member
//     std::vector<IT> vtxdist;
//     std::vector<IT> xadj; 
//     std::vector<IT> adjncy;
//     std::vector<IT> vwgt;
//     std::vector<IT> adjwgt;
//     std::vector<NT> tpwgts;
//     std::vector<NT> ubvec;
//     std::vector<IT> options;
//     std::vector<IT> edgecut;
//     std::vector<IT> parts;
// public:
// // default constructor
// GraphPartitioner() = default;
// // read from a mtx file.
// GraphPartitioner(Csc<IT, NT> * csc){
    
// }

// void Partition(int nparts);



}

#endif

