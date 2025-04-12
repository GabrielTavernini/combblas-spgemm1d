#ifndef SPA_H_
#define SPA_H_

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <algorithm>
#include <tuple>
#include <vector>
#include <parallel/algorithm>
#include "CombBLAS/SpDCCols.h"
#include "csc.h"

namespace combblas{

// SPA that is thread-safe but not multithreaded
// SPA is used in numeric phase
template<bool sortOutput, class IT, class NT>
class SPA {
public:
    SPA(IT range) {
    values.resize(range);
    flags.resize(range, 0);
    }

    template<typename AddOperation>
    void Insert(IT tkey, NT tval, AddOperation addop) {
    if (flags[tkey] == 0) {
        values[tkey] = tval;
        flags[tkey] = 1;
        ids.push_back(tkey);
    } else {    // previously set
        values[tkey] = addop(tval, values[tkey]);
    }
    }

    size_t Size() {
    return ids.size();
    }

    template<typename KeyIterator, typename ValueIterator>
    void OutputReset(KeyIterator firstkey, ValueIterator firstvalue) {
    if(sortOutput) __gnu_parallel::sort(ids.begin(), ids.end());
    // the range starting from first is large enough to hold all output elements
    for (auto index: ids) {
        (*firstkey) = index;
        (*firstvalue) = values[index];
        ++firstkey;
        ++firstvalue;
        flags[index] = 0;
    }
    ids.clear();
    }
    template<typename TupleIterator>
    void OutputReset(TupleIterator iter, int colid) {
    if(sortOutput) __gnu_parallel::sort(ids.begin(), ids.end());
    // the range starting from first is large enough to hold all output elements
    for (auto index: ids) {
        *iter = make_tuple(index, colid, values[index]);
        iter++;
        flags[index] = 0;
    }
    ids.clear();
    }

    std::vector<bool> flags;     // 0: not-set, 1: set
    std::vector <IT> ids;         // initially empty
    std::vector <NT> values;      // O(n), do not need to initialize
};

// SPAStructure that is thread-safe but not multithreaded
// SPAStructure is used in symbolic phase
template<class IT>
class SPAStructure {
public:
    SPAStructure(IT range) {
    flags.resize(range, 0);
    }

    void Insert(IT tkey) {
    if (flags[tkey] == 0) {
        flags[tkey] = 1;
        ids.push_back(tkey);
    }
    }

    size_t Size() {
    return ids.size();
    }

    void Reset() {
    for (auto index: ids) {
        flags[index] = 0;
    }
    ids.clear();
    }

    std::vector<bool> flags;  // 0: not-set, 1: set
    std::vector <IT> ids;     // initially empty
};


}
#endif