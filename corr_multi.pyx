# File: corr_multi.pyx


cdef extern from "math.h":
    double sqrt(double m)
    double pow(double m)

from numpy cimport ndarray, float32_t
cimport numpy
cimport cython
import bottleneck as bn

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(True)

#def corr_multi_cy( ndarray[float32_t, ndim=1]  arr, ndarray[float32_t, ndim=2]  mat):
def corr_multi_cy(arr,mat):

    cdef int numcompare
    cdef int numel
   
    numcompare = len(mat.T)
    numel = int(len(arr))

    coll=ndarray(numcompare)

    arrdemean=arr-bn.nanmean(arr)
    arrss=sqrt(bn.ss(arrdemean))


    for n in range(0,numcompare):
        bdemean=mat[:,n]-bn.nanmean(mat[:,n])
        bss=sqrt(bn.ss(bdemean))

        r=bn.nansum(arrdemean*bdemean)/(arrss*bss)
 
        coll[n]=(r)

    return coll
