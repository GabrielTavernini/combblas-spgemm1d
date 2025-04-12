/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */


#ifndef _PROMOTE_H_
#define _PROMOTE_H_
#include <cstdint>
#include "myenableif.h"

namespace combblas {

template <class T1, class T2, class Enable = void>
struct promote_trait  { };

// typename disable_if< is_boolean<NT>::value, NT >::type won't work, 
// because then it will send Enable=NT which is different from the default template parameter
template <class NT> struct promote_trait< NT , bool, typename combblas::disable_if< combblas::is_boolean<NT>::value >::type >      
{                                           
	typedef NT T_promote;                    
};
template <class NT> struct promote_trait< bool , NT, typename combblas::disable_if< combblas::is_boolean<NT>::value >::type >      
{                                           
	typedef NT T_promote;                    
};

template<class NT> struct promote_trait<NT,NT>	// always allow self promotion
{
	typedef NT T_promote;
};

#define DECLARE_PROMOTE(A,B,C)                  \
    template <> struct promote_trait<A,B>       \
    {                                           \
        typedef C T_promote;                    \
    };


DECLARE_PROMOTE(bool, bool, bool);
DECLARE_PROMOTE(float, float, float);
DECLARE_PROMOTE(double, double, double);
DECLARE_PROMOTE(int, int, int);
DECLARE_PROMOTE(unsigned, unsigned, unsigned);
DECLARE_PROMOTE(unsigned long long, unsigned long long, unsigned long long);


DECLARE_PROMOTE(double, int, double);
DECLARE_PROMOTE(int, double, double);

DECLARE_PROMOTE(int, float, float);
DECLARE_PROMOTE(float, int, float);

DECLARE_PROMOTE(double, int64_t, double);
DECLARE_PROMOTE(int64_t, double, double);

DECLARE_PROMOTE(double, uint64_t, double);
DECLARE_PROMOTE(uint64_t, double, double);


DECLARE_PROMOTE(int64_t, bool, int64_t);
DECLARE_PROMOTE(int64_t, int, int64_t);

DECLARE_PROMOTE(bool, int64_t, int64_t);
DECLARE_PROMOTE(int, int64_t, int64_t);
DECLARE_PROMOTE(int64_t, int64_t, int64_t);
DECLARE_PROMOTE(int, bool, int);
DECLARE_PROMOTE(short, bool,short);
DECLARE_PROMOTE(unsigned, bool, unsigned);
DECLARE_PROMOTE(float, bool, float);
DECLARE_PROMOTE(double, bool, double);
DECLARE_PROMOTE(unsigned long long, bool, unsigned long long);
DECLARE_PROMOTE(bool, int, int);
DECLARE_PROMOTE(bool, short, short);
DECLARE_PROMOTE(bool, unsigned, unsigned);
DECLARE_PROMOTE(bool, float, float);
DECLARE_PROMOTE(bool, double, double);
DECLARE_PROMOTE(bool, unsigned long long, unsigned long long);

template <class IT, class NT>
class SpDCCols;

template <class IT, class NT>
class SpCCols;

template <class IT, class NT>
class SpTuples;



// Below are necessary constructs to be able to define a SpMat<NT,IT> where
// all we know is DER (say SpDCCols<int, double>) and NT,IT
// in other words, we infer the templated SpDCCols<> type
// This is not a type conversion from an existing object, 
// but a type inference for the newly created object
// NIT: New IT, NNT: New NT
template <class DER, class NIT, class NNT>
struct create_trait
{
	// none
};

// Capture everything of the form SpDCCols<OIT, ONT>
// it may come as a surprise that the partial specializations can 
// involve more template parameters than the primary template
template <class NIT, class NNT, class OIT, class ONT>
struct create_trait< SpDCCols<OIT, ONT> , NIT, NNT >
{
     typedef SpDCCols<NIT,NNT> T_inferred;
};

// Capture everything of the form SpCCols<OIT, ONT>
// it may come as a surprise that the partial specializations can
// involve more template parameters than the primary template
template <class NIT, class NNT, class OIT, class ONT>
struct create_trait< SpCCols<OIT, ONT> , NIT, NNT >
{
    typedef SpCCols<NIT,NNT> T_inferred;
};


// At this point, complete type of of SpDCCols is known, safe to declare these specialization (but macros won't work as they are preprocessed)
// General case #1: When both NT is the same
template <class IT, class NT> struct promote_trait< SpDCCols<IT,NT> , SpDCCols<IT,NT> >          
	{                                           
        typedef SpDCCols<IT,NT> T_promote;                    
    };
// General case #2: First is boolean the second is anything except boolean (to prevent ambiguity) 
template <class IT, class NT> struct promote_trait< SpDCCols<IT,bool> , SpDCCols<IT,NT>, typename combblas::disable_if< combblas::is_boolean<NT>::value >::type >      
    {                                           
        typedef SpDCCols<IT,NT> T_promote;                    
    };
// General case #3: Second is boolean the first is anything except boolean (to prevent ambiguity) 
template <class IT, class NT> struct promote_trait< SpDCCols<IT,NT> , SpDCCols<IT,bool>, typename combblas::disable_if< combblas::is_boolean<NT>::value >::type >      
	{                                           
		typedef SpDCCols<IT,NT> T_promote;                    
	};
template <class IT> struct promote_trait< SpDCCols<IT,int> , SpDCCols<IT,float> >       
    {                                           
        typedef SpDCCols<IT,float> T_promote;                    
    };

template <class IT> struct promote_trait< SpDCCols<IT,float> , SpDCCols<IT,int> >       
    {                                           
        typedef SpDCCols<IT,float> T_promote;                    
    };
template <class IT> struct promote_trait< SpDCCols<IT,int> , SpDCCols<IT,double> >       
    {                                           
        typedef SpDCCols<IT,double> T_promote;                    
    };
template <class IT> struct promote_trait< SpDCCols<IT,double> , SpDCCols<IT,int> >       
    {                                           
        typedef SpDCCols<IT,double> T_promote;                    
    };



// At this point, complete type of of SpCCols is known, safe to declare these specialization (but macros won't work as they are preprocessed)
// General case #1: When both NT is the same
template <class IT, class NT> struct promote_trait< SpCCols<IT,NT> , SpCCols<IT,NT> >
{
    typedef SpCCols<IT,NT> T_promote;
};
// General case #2: First is boolean the second is anything except boolean (to prevent ambiguity)
template <class IT, class NT> struct promote_trait< SpCCols<IT,bool> , SpCCols<IT,NT>, typename combblas::disable_if< combblas::is_boolean<NT>::value >::type >
{
    typedef SpCCols<IT,NT> T_promote;
};
// General case #3: Second is boolean the first is anything except boolean (to prevent ambiguity)
template <class IT, class NT> struct promote_trait< SpCCols<IT,NT> , SpCCols<IT,bool>, typename combblas::disable_if< combblas::is_boolean<NT>::value >::type >
{
    typedef SpCCols<IT,NT> T_promote;
};
template <class IT> struct promote_trait< SpCCols<IT,int> , SpCCols<IT,float> >
{
    typedef SpCCols<IT,float> T_promote;
};

template <class IT> struct promote_trait< SpCCols<IT,float> , SpCCols<IT,int> >
{
    typedef SpCCols<IT,float> T_promote;
};
template <class IT> struct promote_trait< SpCCols<IT,int> , SpCCols<IT,double> >
{
    typedef SpCCols<IT,double> T_promote;
};
template <class IT> struct promote_trait< SpCCols<IT,double> , SpCCols<IT,int> >
{
    typedef SpCCols<IT,double> T_promote;
};




// At this point, complete type of of SpTuples is known, safe to declare these specialization (but macros won't work as they are preprocessed)
template <> struct promote_trait< SpTuples<int,int> , SpTuples<int,int> >       
    {                                           
        typedef SpTuples<int,int> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,float> , SpTuples<int,float> >       
    {                                           
        typedef SpTuples<int,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,double> , SpTuples<int,double> >       
    {                                           
        typedef SpTuples<int,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,bool> , SpTuples<int,int> >       
    {                                           
        typedef SpTuples<int,int> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,int> , SpTuples<int,bool> >       
    {                                           
        typedef SpTuples<int,int> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,int> , SpTuples<int,float> >       
    {                                           
        typedef SpTuples<int,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,float> , SpTuples<int,int> >       
    {                                           
        typedef SpTuples<int,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,int> , SpTuples<int,double> >       
    {                                           
        typedef SpTuples<int,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,double> , SpTuples<int,int> >       
    {                                           
        typedef SpTuples<int,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,unsigned> , SpTuples<int,bool> >       
    {                                           
        typedef SpTuples<int,unsigned> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,bool> , SpTuples<int,unsigned> >       
    {                                           
        typedef SpTuples<int,unsigned> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,bool> , SpTuples<int,double> >       
    {                                           
        typedef SpTuples<int,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,bool> , SpTuples<int,float> >       
    {                                           
        typedef SpTuples<int,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,double> , SpTuples<int,bool> >       
    {                                           
        typedef SpTuples<int,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,float> , SpTuples<int,bool> >       
    {                                           
        typedef SpTuples<int,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,int> , SpTuples<int64_t,int> >       
    {                                           
        typedef SpTuples<int64_t,int> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,float> , SpTuples<int64_t,float> >       
    {                                           
        typedef SpTuples<int64_t,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,double> , SpTuples<int64_t,double> >       
    {                                           
        typedef SpTuples<int64_t,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,int64_t> , SpTuples<int64_t,int64_t> >       
    {                                           
        typedef SpTuples<int64_t,int64_t> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,bool> , SpTuples<int64_t,int> >       
    {                                           
        typedef SpTuples<int64_t,int> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,int> , SpTuples<int64_t,bool> >       
    {                                           
        typedef SpTuples<int64_t,int> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,int> , SpTuples<int64_t,float> >       
    {                                           
        typedef SpTuples<int64_t,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,float> , SpTuples<int64_t,int> >       
    {                                           
        typedef SpTuples<int64_t,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,int> , SpTuples<int64_t,double> >       
    {                                           
        typedef SpTuples<int64_t,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,double> , SpTuples<int64_t,int> >       
    {                                           
        typedef SpTuples<int64_t,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,unsigned> , SpTuples<int64_t,bool> >       
    {                                           
        typedef SpTuples<int64_t,unsigned> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,bool> , SpTuples<int64_t,unsigned> >       
    {                                           
        typedef SpTuples<int64_t,unsigned> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,bool> , SpTuples<int64_t,double> >       
    {                                           
        typedef SpTuples<int64_t,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,bool> , SpTuples<int64_t,float> >       
    {                                           
        typedef SpTuples<int64_t,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,double> , SpTuples<int64_t,bool> >       
    {                                           
        typedef SpTuples<int64_t,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,float> , SpTuples<int64_t,bool> >       
    {                                           
        typedef SpTuples<int64_t,float> T_promote;                    
    };




}

#endif
