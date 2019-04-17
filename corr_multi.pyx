# File: corr_multi.pyx


cdef extern from "math.h":
    double sqrt(double m)
    
cdef int numel
cdef int numel_m1
cdef int numcompare
cdef int n

from numpy cimport ndarray
cimport numpy as np
cimport cython
import bottleneck as bn


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(True)

#def corr_multi_cy( ndarray[float32_t, ndim=1]  arr, ndarray[float32_t, ndim=2]  mat):
def corr_multi_cy(arr,mat):

    numcompare = len(mat.T)
    numel = int(len(arr))
    numel_m1 = int(numel-1)

    coll=ndarray(numcompare)

    arrdemean=arr-bn.nanmean(arr)
    arrss=sqrt(bn.ss(arrdemean))
    arrstd=bn.nanstd(arr)


    for n in range(0,numcompare):
        submat=mat[:,n]
        bdemean=submat-bn.nanmean(submat)
        bss=sqrt(bn.ss(bdemean))

        cross_mul=bn.nansum(arrdemean*bdemean)
        r=cross_mul/(arrss*bss)
        #p=cross_mul/(numel_m1*arrstd*bn.nanstd(submat))
        


        coll[n]=(r)

    return coll
