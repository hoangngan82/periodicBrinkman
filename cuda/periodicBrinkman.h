/*
 * This file is the Ewald summation method for image system of a regularized
 * Stokeslet.
 */
#ifndef PERIODIC_BRINKMAN_H 
#define PERIODIC_BRINKMAN_H 
#include "../../include/matrix.h"
#include <cuda.h>
#include <math_functions.h>
//#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

// Store 3x3 matrix in col-major to be compatible with lapack
#define IDX2D(row, col) (((col)*3) + row)

__device__ __host__ double 
experf (double x, double z, const double &c = 2) {/*{{{*/
  double a = 0;
  if ((x*z < 0) || (x*x + z*z < 700))
    a = exp(c*x*z) * erfc(z + x);
  return a;
}/*}}}*/

// === singH: CUDA DEVICE AND HOST FUNCTION ======================{{{
//         Name:  singH
//  Description: Compute the singular version of H1 and H2
// =============================================================================
__device__ __host__ void
singH ( /* argument list: {{{*/
    const double& r2, const double& a
    , double& H1, double& H2 
    ) /* ------------- end of argument list -------------------------------}}}*/ 
{ /* singH implementation: {{{*/
  double expx2 = exp(-r2);
  double r     = sqrt(r2);

  if (r > eps) {
    H1 = -1/(4*PI*a*a*r*r2)    + exp(-a*r)/(4*PI*r)*(1 + 1/(a*r) + 1/(a*a*r2));
    H2 =  3/(4*PI*a*a*r*r2*r2) - exp(-a*r)/(4*PI*r2*r)*(1 + 3/(a*r) + 3/(a*a*r2));
  } else {
    H1 = H2 = 1.79e+308;
  }
} /*}}}*/
/* ---------------  end of DEVICE function singH  -------------- }}}*/

// === regH: CUDA DEVICE AND HOST FUNCTION ======================{{{
//         Name:  regH
//  Description:  Compute the regularized H1 and H2
// =============================================================================
__device__ __host__ void
regH ( /* argument list: {{{*/
    const double& r2, const double& a, const double& d
   , double& H1, double& H2 
    ) /* ------------- end of argument list -------------------------------}}}*/ 
{ /* regH implementation: {{{*/
  double expx2 = exp(-r2/d/d);
  double r     = sqrt(r2);
  double Erf   = erf(r/d);
  
  if (r > eps) {
    H1 = -1/(4*PI*a*a*r2*r)*Erf;
    H1+= 1/(2*a*a*d*d*d*PI*SPI*r2)*expx2*(d*d + 2*r2)*(1 - exp(-a*a*d*d/4));
    H1+= 1/(8*a*a*PI*r2*r)*(1 + a*r + a*a*r2)*experf(a*d/2,-r/d);
    H1-= 1/(8*a*a*PI*r2*r)*(1 - a*r + a*a*r2)*experf(a*d/2, r/d);
    
    H2 = 3/(4*PI*a*a*r2*r2*r)*Erf;
    H2-= (1 - exp(-a*a*d*d/4))/(2*a*a*d*d*d*PI*SPI*r2*r2)*expx2*(3*d*d + 2*r2);
    H2-= 1/(8*a*a*PI*r2*r2*r)*(3 + 3*a*r + a*a*r2)*experf(a*d/2, -r/d);
    H2+= 1/(8*a*a*PI*r2*r2*r)*(3 - 3*a*r + a*a*r2)*experf(a*d/2,  r/d);
  } else {
    H1 = -a*erfc(a*d/2)/(6*PI) + (2 + exp(-a*a*d*d/4)*(a*a*d*d - 2))/(3*a*a*d*d*d*PI*SPI);
    H2 = (-24 + 2*a*a*d*d*(2 - a*a*d*d))*exp(-a*a*d*d/4) + 24;
    H2 = H2/(60*a*a*d*d*d*d*d*PI*SPI) + a*a*a/(60*PI)*erfc(a*d/2);
  }
} /*}}}*/
/* ---------------  end of DEVICE function regH  -------------- }}}*/

// === diffH: CUDA DEVICE AND HOST FUNCTION ======================{{{
//         Name:  diffH
//  Description:  Difference between singular and regularized versioni
// =============================================================================
__device__ __host__ void
diffH ( /* argument list: {{{*/
    const double& r2, const double& a, const double& d
   , double& dH1, double& dH2 
    ) /* ------------- end of argument list -------------------------------}}}*/ 
{ /* diffH implementation: {{{*/
  double expx2 = exp(-r2/d/d)*(1 - exp(-a*a*d*d/4));
  double r     = sqrt(r2);
  double Erfc   = erfc(r/d);

  if (r > eps) {
    dH1 = -1/(4*PI*a*a)/(r2*r)*Erfc;
    dH1-= (d*d + 2*r2)/(2*a*a*d*d*d*PI*SPI*r2)*expx2;
    dH1+= 1/(8*PI*a*a*r2*r)*(1 + a*r + a*a*r2)*experf(-a*d/2,r/d);
    dH1+= 1/(8*PI*a*a*r2*r)*(1 - a*r + a*a*r2)*experf( a*d/2,r/d);
    
    dH2 = 3/(4*PI*a*a)/(r2*r2*r)*Erfc;
    dH2+= (3*d*d + 2*r2)/(2*a*a*d*d*d*PI*SPI*r2*r2)*expx2;
    dH2-= 1/(8*PI*a*a*r2*r2*r)*(3 + 3*a*r + a*a*r2)*experf(-a*d/2,r/d);
    dH2-= 1/(8*PI*a*a*r2*r2*r)*(3 - 3*a*r + a*a*r2)*experf( a*d/2,r/d);
  } else {
    dH1 = -( 4 + 2*( -2 + a*a*d*d )*exp( -a*a*d*d/4 ) ) / ( a*a*d*d*d*6*PI*SPI ) 
      + a*erfc( a*d/2 ) / (6*PI);
    dH2 = -( 2 - a*a*d*d )*exp( -a*a*d*d/4 ) / ( 30*d*d*d*PI*SPI ) 
      - 2*( 1 - exp( -a*a*d*d/4 ) ) / ( 5*a*a*d*d*d*d*d*PI*SPI ) 
      - a*a*a*erfc( a*d/2 ) / ( 60*PI );
  }
} /*}}}*/
/* ---------------  end of DEVICE function diffH  -------------- }}}*/

// === Brinkmanlet: CUDA DEVICE AND HOST FUNCTION ======================{{{
//         Name:  Brinkmanlet
//  Description:  Given H1 and H2, compute the velocity field.
// =============================================================================
__device__ __host__ void
Brinkmanlet ( /* argument list: {{{*/
    double* A, 
    const double* xh, const double& a, const double& d = 0
    ) /* ------------- end of argument list -------------------------------}}}*/ 
{ /* Brinkmanlet implementation: {{{*/
  assert(d >= 0);
  double H1, H2;
  double r2 = dot(xh, xh);

  if (d > 0) {
    regH (r2, a, d, H1, H2);
  } else {
    singH(r2, a, H1, H2);
  }
  // Compute Sij = dij * H1 + xixj * H2;
  for ( int i = 0; i < 3; i++ ) { /* loop by row {{{*/
    for ( int j = 0; j < 3; j++ ) { /* loop by columns {{{*/
      A[IDX2D(i, j)] = xh[i]*xh[j]*H2;
    }                      /*---------- end of for loop ----------------}}}*/
    A[IDX2D(i, i)] += H1;
  }                        /*---------- end of for loop ----------------}}}*/
} /*}}}*/
/* ---------------  end of DEVICE function Brinkmanlet  -------------- }}}*/

// === vR: CUDA DEVICE AND HOST FUNCTION ======================{{{
//         Name:  vR
//  Description:  Compute the real space term which is the difference between
//  the (regularized) Brinkmanlet with other chosen Brinkmanlet.
//  Input      :  
//  d is the splitting parameter
//  e is the blob parameter (e = 0 meaning singular version)
// =============================================================================
__device__ __host__ void
vR ( /* argument list: {{{*/
    double* A, const double* const xh, const double& a
    , const double& d, const double& e 
   ) /* ------------- end of argument list -------------------------------}}}*/ 
{ /* vR implementation: {{{*/
  assert(d > 0);
  assert(e >= 0);
  double H1, H2, tH1, tH2;
  double r2 = dot(xh, xh);
  diffH (r2, a, d, H1, H2);
  tH1 = tH2 = 0;
  if (e > 0) {
    diffH (r2, a, e, tH1, tH2);
  } 
  H1 -= tH1;
  H2 -= tH2;
  // Compute Sij = dij * H1 + xixj * H2;
  for ( int i = 0; i < 3; i++ ) { /* loop by row {{{*/
    for ( int j = 0; j < 3; j++ ) { /* loop by columns {{{*/
      A[IDX2D(i, j)] = xh[i]*xh[j]*H2;
    }                      /*---------- end of for loop ----------------}}}*/
    A[IDX2D(i, i)] += H1;
  }                        /*---------- end of for loop ----------------}}}*/
} /*}}}*/
/* ---------------  end of DEVICE function vR  -------------- }}}*/

// === vF: CUDA DEVICE AND HOST FUNCTION ======================{{{
//         Name:  vF
//  Description:  Compute the Fourier space term.
// =============================================================================
__device__ __host__ void
vF ( /* argument list: {{{*/
    double* A, const double* const xh, const double& a
   , const double& d, const double* const k 
   ) /* ------------- end of argument list -------------------------------}}}*/ 
{ /* vF implementation: {{{*/
  double kx = dot (xh, k);
  double kn = dot (k, k);
  double x  = a*d*a*d;
  double w  = d*d*kn;
  double a1 = 1 - exp(-x/4);
  double a0 = x + 6*a1;
  double bhat = 1/x*(a1*(w-6) + a0)*exp(-w/4)*cos(kx);
  bhat *= 1/(kn*(kn + a*a));

  // Compute Sij = dij * H1 + xixj * H2;
  for ( int i = 0; i < 3; i++ ) { /* loop by row {{{*/
    A[IDX2D(i, i)] = 0;
    for ( int j = 0; j < 3; j++ ) { /* loop by columns {{{*/
      if (i != j) {
        A[IDX2D(i, j)]  = -k[i]*k[j]*bhat;
        A[IDX2D(i, i)] += k[j]*k[j]*bhat;
      }
    }                      /*---------- end of for loop ----------------}}}*/
  }                        /*---------- end of for loop ----------------}}}*/
} /*}}}*/
/* ---------------  end of DEVICE function vF  -------------- }}}*/

// === kernelReal: CUDA KERNEL ========================================{{{
//         Name:  kernelReal
//  Description:  Compute the action of one (1) point force on one (1) field
//  point.
//  Each block will compute the matrix corresponding to one (1) field point and
//  one (1) point force.
//  BlockSize must be a power of 2.
// =============================================================================
__global__ void
kernelReal ( /* argument list: {{{*/
    MatrixOnDevice dA
    , MatrixOnDevice dx
    , MatrixOnDevice dx0 
    , double a
    , double d, double e 
    , int Mx // number of real shells in x direction
    , int My // number of real shells in y direction
    , int Mz // number of real shells in z direction
    , double Lx, double Ly, double Lz
    , uint bitSize
    ) /* ------------- end of argument list ------------------------------}}}*/ 
{ /* kernelReal implementation: {{{*/
  uint blockSize = 1 << bitSize;
  assert ( blockSize == blockDim.x * blockDim.y );
  extern __shared__ double U[];
  __shared__ double xh0[3];
  double A[9];

  uint xid   =  blockIdx.x;
  uint yid   =  blockIdx.y;
  uint xSize =  gridDim.x;
  uint ySize =  gridDim.y;
  uint tid, xyid;
  int l1, l2, l3;

  double xh[3];
  while (xid < dx.columns()) { 
    yid = blockIdx.y;
    while (yid < dx0.columns()) {
      tid = threadIdx.x + threadIdx.y * blockDim.x;

      for ( int i = 0; i < 9; i++ ) { /* initialize data */
        U[tid + i*blockSize] = 0;
        A[i] = 0;
      }         /*---------- end of for loop ----------------*/
      
      if ( tid == 0 ) { /* compute a common xh for all threads in a block */
        xh0[0]  = dx(0, xid) - dx0(0, yid);
        xh0[1]  = dx(1, xid) - dx0(1, yid);
        xh0[2]  = dx(2, xid) - dx0(2, yid);
      }         /*---------- end of if ----------------------*/

      __syncthreads();

      while ( tid < (2*Mx + 1) * (2*My + 1) * (2*Mz + 1) ) {
        xyid = tid % ((2*Mx + 1) * (2*My + 1));
        l1 = (xyid % (2*Mx + 1)) - Mx;
        l2 = (xyid / (2*Mx + 1)) - My;
        l3 = (tid / ((2*Mx + 1) * (2*My + 1))) - Mz;

        xh[0] = xh0[0] - l1 * Lx;
        xh[1] = xh0[1] - l2 * Ly;
        xh[2] = xh0[2] - l3 * Lz;
  
        vR ( A, xh, a, d, e ); 
  
        for ( int i = 0; i < 9; i++ ) { /*  */
          U[(tid & (blockSize - 1)) + (i << bitSize)] += A[i];
          A[i] = 0;
        }         /*---------- end of for loop ----------------*/
        tid += blockSize;
      }

      __syncthreads();
      
      tid =  threadIdx.x + threadIdx.y * blockDim.x;
      uint index = 0;
    
      for ( int i = 0; i < 9; i++ ) { /* running sum reduction on data */
        index = (tid & (blockSize - 1)) + (i << bitSize);
        // Now do the partial reduction
        if ( blockSize > 512 ) { if ( tid < 512 ) { U[index] += U[index + 512]; } __syncthreads (); } 
        if ( blockSize > 256 ) { if ( tid < 256 ) { U[index] += U[index + 256]; } __syncthreads (); } 
        if ( blockSize > 128 ) { if ( tid < 128 ) { U[index] += U[index + 128]; } __syncthreads (); }
        if ( blockSize >  64 ) { if ( tid <  64 ) { U[index] += U[index +  64]; } __syncthreads (); }
        if ( tid < 32 ) {
          warpReduce ( &U[blockSize*i], tid, blockSize );
        }
      }         /*---------- end of for loop ----------------*/

      __syncthreads();

      while (tid < 9) {
        dA ((tid % 3) + xid * 3, (tid / 3) + yid * 3) += U[tid << bitSize];
        tid += blockSize;
      }
    
      yid += ySize;
      __syncthreads();
    }
    xid += xSize;
    __syncthreads();
  }
} /*}}}*/
/* ----------------  end of CUDA kernel kernelReal  ----------------- }}}*/

// === kernelFourier: CUDA KERNEL ========================================{{{
//         Name:  kernelFourier
//  Description:  Compute the action of one (1) point force on one (1) field
//  point.
//  Each block will compute the matrix corresponding to one (1) field point and
//  one (1) point force.
//  BlockSize must be a power of 2.
// =============================================================================
__global__ void
kernelFourier ( /* argument list: {{{*/
    MatrixOnDevice dA
    , MatrixOnDevice dx
    , MatrixOnDevice dx0 
    , double a
    , double d, double e 
    , int Mx // number of Fourier shells in l1 direction
    , int My // number of Fourier shells in l2 direction
    , int Mz // number of Fourier shells in l3 direction
    , double Lx, double Ly, double Lz
    , int bitSize
    ) /* ------------- end of argument list ------------------------------}}}*/ 
{ /* kernelFourier implementation: {{{*/
  uint blockSize = 1 << bitSize;
  assert ( blockSize == blockDim.x * blockDim.y );
  extern __shared__ double U[];
  double dl[3];
  __shared__ double xh[3];
  double A[9];

  uint xid   =  blockIdx.x;
  uint yid   =  blockIdx.y;
  uint xSize =  gridDim.x;
  uint ySize =  gridDim.y;
  uint tid, xyid;
  int l1, l2, l3;

  while (xid < dx.columns()) { 
    yid = blockIdx.y;
    while (yid < dx0.columns()) {
      tid =  threadIdx.x + threadIdx.y * blockDim.x;

      for ( int i = 0; i < 9; i++ ) { /* initialize data */
        U[tid + (i << bitSize)] = 0;
        A[i] = 0;
      }         /*---------- end of for loop ----------------*/

      if ( tid == 0 ) { /* compute xh common to all threads in a block */
        xh[0]   = dx(0, xid) - dx0(0, yid);
        xh[1]   = dx(1, xid) - dx0(1, yid);
        xh[2]   = dx(2, xid) - dx0(2, yid);
      }         /*---------- end of if ----------------------*/

      __syncthreads();

      while ( tid < (2*Mx + 1)*(2*My + 1)*(2*Mz + 1) ) {
        xyid = tid % ((2*Mx + 1) * (2*My + 1));
        l1 = (xyid % (2*Mx + 1)) - Mx;
        l2 = (xyid / (2*Mx + 1)) - My;
        l3 = (tid / ((2*Mx + 1) * (2*My + 1))) - Mz;

        if ((l1 == 0) && (l2 == 0) && (l3 == 0)) {
          tid += blockSize;
          continue;
        }

        dl[0] = 2*PI*l1/Lx;
        dl[1] = 2*PI*l2/Ly;
        dl[2] = 2*PI*l3/Lz;
        
        vF (A, xh, a, d, dl);

        for ( int i = 0; i < 9; i++ ) { /*  */
          U[(tid & (blockSize - 1)) + (i << bitSize)] += A[i];
          A[i] = 0;
        }         /*---------- end of for loop ----------------*/
        tid += blockSize;
      }

      __syncthreads();
      
      tid =  threadIdx.x + threadIdx.y * blockDim.x;
      uint index = 0;

      for ( int i = 0; i < 9; i++ ) { /* running sum reduction on data */
        index = (tid & (blockSize - 1)) + (i << bitSize);
        // Now do the partial reduction
        if ( blockSize > 512 ) { if ( tid < 512 ) { U[index] += U[index + 512]; } __syncthreads (); } 
        if ( blockSize > 256 ) { if ( tid < 256 ) { U[index] += U[index + 256]; } __syncthreads (); } 
        if ( blockSize > 128 ) { if ( tid < 128 ) { U[index] += U[index + 128]; } __syncthreads (); }
        if ( blockSize >  64 ) { if ( tid <  64 ) { U[index] += U[index +  64]; } __syncthreads (); }
        if ( tid < 32 ) {
          warpReduce ( &U[blockSize*i], tid, blockSize );
        }
      }         /*---------- end of for loop ----------------*/

      __syncthreads();

      while (tid < 9) {
        dA ((tid % 3) + xid * 3, (tid / 3) + yid * 3) += U[tid << bitSize];
        tid += blockSize;
      }
          
      yid += ySize;
      __syncthreads();
    }
    xid += xSize;
    __syncthreads();
  }
} /*}}}*/
/* ----------------  end of CUDA kernel kernelFourier  ----------------- }}}*/

// === periodicBrinkman: FUNCTION  =========================================={{{ 
//         Name:  periodicBrinkman
//  Description:  Calculate the real space sum
// =============================================================================
void
periodicBrinkman ( /* argument list: {{{*/
   MatrixOnDevice dA, MatrixOnDevice dx, MatrixOnDevice dx0
   , const double & a
   , const double & d, const double & e
   , const int & M // numRealShells
   , const int & N // numFourierShells
   , const double* const L 
   , const int & maxBlockSize = 256
   ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* periodicBrinkman implementation: {{{*/
 assert ((dA.rows() == 3 * dx.columns()) 
     &&  (dA.columns() == 3 * dx0.columns())
     &&  (dx.rows() == 3) && (dx0.rows() == 3));
 assert ( maxBlockSize > 0 );
 uint maxBitSize = 0;
 while ((1 << maxBitSize) < maxBlockSize) {
   maxBitSize++;
 }
 if ( (1 << maxBitSize) != maxBlockSize ) {
   printf("maxBlockSize must be a power of 2 and less than 2^9 = 512 for device with compute capability <= 3.0\n");
   maxBitSize--;
   printf("maxBlockSize is reduced to 2^%d = %d\n", maxBitSize, (1 << maxBitSize));
 }
//  assert ((dx.columns() % dimBlockx == 0) && (dx0.columns() % dimBlocky == 0));
 int Mx[3] = {M, M, M};
 int Nx[3] = {N, N, N};
 int idL[3] = {0, 1, 2};
 int temp;
 
 // rearrange the periodic box size so that Lx <= Ly <= Lz
 if (L[idL[0]] > L[idL[1]]) {
   temp = idL[0];
   idL[0] = idL[1];
   idL[1] = temp;
 }
 if (L[idL[2]] <= L[idL[0]]) {
   temp = idL[2];
   idL[2] = idL[1];
   idL[1] = idL[0];
   idL[0] = temp;
 } else {
   if (L[idL[2]] < L[idL[1]]) {
     temp = idL[1];
     idL[1] = idL[2];
     idL[2] = temp;
   }
 }

 // For real space sum, we multiply by the smallest number.
 Mx[idL[2]] = ceil(M*L[idL[0]]/L[idL[2]]) ;  
 Mx[idL[1]] = ceil(M*L[idL[0]]/L[idL[1]]) ;

 // For Fourier space sum, we multiply by the largest number.
 Nx[idL[0]] = ceil(N*L[idL[2]]/L[idL[0]]) ;  
 Nx[idL[1]] = ceil(N*L[idL[2]]/L[idL[1]]) ;

 uint dimGridx   = chooseSize(dx.columns());
 uint dimGridy   = chooseSize(dx0.columns());
 dim3 dimGrid (dimGridx, dimGridy);

 uint blockSize  = chooseSize((2*Nx[0] + 1)*(2*Nx[1] + 1)*(2*Nx[2] + 1), maxBitSize);
 dim3 dimBlock (blockSize);
 uint bitSize = 0;
 while ((1 << bitSize) < blockSize) {
   bitSize++;
 }
 if (N > 0)
 kernelFourier <<< dimGrid, dimBlock, blockSize * 9 * sizeof(double) >>> 
   (dA, dx, dx0, a, d, e, Nx[0], Nx[1], Nx[2], L[0], L[1], L[2], bitSize);

 blockSize = chooseSize((2*Mx[0] + 1)*(2*Mx[1] + 1)*(2*Mx[2] + 1), maxBitSize);
 dimBlock  = dim3(blockSize);
 
 bitSize   = 0;
 while ((1 << bitSize) < blockSize) {
   bitSize++;
 }
 if (M >= 0)
 kernelReal <<< dimGrid, dimBlock, blockSize * 9 * sizeof(double) >>> 
   (dA, dx, dx0, a, d, e, Mx[0], Mx[1], Mx[2], L[0], L[1], L[2], bitSize);
} /*}}}*/
/* ---------------  end of function periodicBrinkman  -------------------- }}}*/

// === readData: FUNCTION  =========================================={{{ 
//         Name:  readData
//  Description:  Load parameters from a text file. The file is structured into
//  two columns separated by a semicolon and zero or more spaces. You can also
//  choose your own delimiters. The left
//  column contains the names of the parameters and the second column contains
//  their numerical values.
// =============================================================================
int
readData ( /* argument list: {{{*/
    double* values, const std::string names[], const int& numel
    , const char* fileName, const char* delimiters = ": "
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* readData implementation: {{{*/
  std::ifstream param(fileName);
  if (!param.good()) {
    std::cout << "Cannot open file!" << std::endl;
    return 1;
  }

  while (!param.eof()) {
    char buf[80];
    char * params;
    param.getline(buf, 80);
    params = strtok(buf, delimiters);
    if (params)
    for (int n = 0; n < numel; n++) {
      if (names[n] == std::string(params)) {
//        cout << params << ": ";
        params = strtok(NULL, delimiters);
        values[n] = atof(params);
//        cout << values[n] << std::endl;
        break;
      }
    }
  }
  param.close();
  return 0;
} /*}}}*/
/* ---------------  end of function readData  -------------------- }}}*/

// === realShells: FUNCTION  =========================================={{{ 
//         Name:  realShells
//  Description:  Compute the outer shell in real space.
//  Assume: L(1) = Lx <= Ly = L(2)
//  Since L(1) < L(2), we can increase M(1) each round and choose 
//  M(2) = ceil(Lx/Ly*M(1)).
// =============================================================================
void
realShells ( /* argument list: {{{*/
    MatrixOnHost & A, MatrixOnHost & absA
    , const MatrixOnHost & x, const MatrixOnHost & x0
    , const MatrixOnHost & newM
    , const MatrixOnHost & oldM
    , const MatrixOnHost & L // Assume that Lx <= Ly <= Lz
    , const double & a       // permeability
    , const double & d, const double & e
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* realShells implementation: {{{*/
  assert((3*x.columns() <= A.rows()) && (3*x0.columns() <= A.columns()));
  MatrixOnHost xh(3, 1), xh0(3, 1);
  MatrixOnHost tA(3, 3);
  if ((newM(0)==oldM(0)) && (newM(1)==oldM(1))) { // when there is NO shell, we use the original force location
   //std::cout << "In side initial realShells at M = " << oldM(0) << std::endl;
    for (int i = 0; i < x.columns(); i++) {
      for (int j = 0; j < x0.columns(); j++) {
          //std::cout << " inside e = " << e << std::endl;
        for (int k = 0; k < 3; k++) 
          xh(k) = x(k, i) - x0(k, j);
        vR( tA, xh, a, d, e ); 
        for (int x1 = 0; x1 < 3; x1++) {
          for (int x2 = 0; x2 < 3; x2++) {
            A(3*i + x1, 3*j + x2) = tA(x1, x2);
            absA(3*i + x1, 3*j + x2) = abs(tA(x1, x2));
          }
        }
      }
    }
    return;
  }

  //std::cout << "inside realshells " << std::endl;
  // If there is a shell of finite thickness, 
  for (int i = 0; i < x.columns(); i++) {
    for (int j = 0; j < x0.columns(); j++) {
      for (int k = 0; k < 3; k++) xh0(k) = x(k, i) - x0(k, j);
     // compute left and right cells, assuming that Lx < Ly
      //      Ly
      // |--|--|--|--|
      // | .| .| .| .|
      // |  |  |  |  |
      // |--|--|--|--|  Lx 
      // | .| .| .| .|
      // |  |  |  |  |
      // |--|--|--|--|
      // Compute top and bottom cells
      for (int l3 = oldM(2); l3 < newM(2); l3++) {
        for (int l2 = -newM(1); l2 <= newM(1); l2++) {
          for( int l1 = -newM(0); l1 <= newM(0); l1++ ) {
            xh(2) = xh0(2) - (l3 + 1)*L(2);
            xh(1) = xh0(1) - l2*L(1);
            xh(0) = xh0(0) - l1*L(0);
            vR( tA, xh, a, d, e ); 
            for (int x1 = 0; x1 < 3; x1++) {
              for (int x2 = 0; x2 < 3; x2++) {
                A(3*i + x1, 3*j + x2) += tA(x1, x2);
                absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
              }
            }

            xh(2) = xh0(2) + (l3 + 1)*L(2);
            vR( tA, xh, a, d, e ); 
            for (int x1 = 0; x1 < 3; x1++) {
              for (int x2 = 0; x2 < 3; x2++) {
                A(3*i + x1, 3*j + x2) += tA(x1, x2);
                absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
              }
            }
          }
        }
      }

      // Compute left and right cells
      for (int l2 = oldM(1) + 1; l2 <= newM(1); l2++) {
        for (int l3 = -oldM(2); l3 <= oldM(2); l3++) {
          for( int l1 = -newM(0); l1 <= newM(0); l1++ ) {
            xh(2) = xh0(2) - l3*L(2);
            xh(1) = xh0(1) - l2*L(1);
            xh(0) = xh0(0) - l1*L(0);
            vR( tA, xh, a, d, e ); 
            for (int x1 = 0; x1 < 3; x1++) {
              for (int x2 = 0; x2 < 3; x2++) {
                A(3*i + x1, 3*j + x2) += tA(x1, x2);
                absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
              }
            }

            xh(1) = xh0(1) + l2*L(1);
            vR( tA, xh, a, d, e ); 
            for (int x1 = 0; x1 < 3; x1++) {
              for (int x2 = 0; x2 < 3; x2++) {
                A(3*i + x1, 3*j + x2) += tA(x1, x2);
                absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
              }
            }
          }
        }
      }

      // Compute front and back cells
      for (int l1 = oldM(0) + 1; l1 <= newM(0); l1++) {
        for (int l3 = -oldM(2); l3 <= oldM(2); l3++) {
          for( int l2 = -oldM(0); l2 <= oldM(0); l2++ ) {
            xh(2) = xh0(2) - l3*L(2);
            xh(1) = xh0(1) - l2*L(1);
            xh(0) = xh0(0) - l1*L(0);
            vR( tA, xh, a, d, e ); 
            for (int x1 = 0; x1 < 3; x1++) {
              for (int x2 = 0; x2 < 3; x2++) {
                A(3*i + x1, 3*j + x2) += tA(x1, x2);
                absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
              }
            }

            xh(0) = xh0(0) + l1*L(0);
            vR( tA, xh, a, d, e ); 
            for (int x1 = 0; x1 < 3; x1++) {
              for (int x2 = 0; x2 < 3; x2++) {
                A(3*i + x1, 3*j + x2) += tA(x1, x2);
                absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
              }
            }
          }
        }
      }
    }
  }
} /*}}}*/
/* ---------------  end of function realShells  -------------------- }}}*/

// === fourierShells: FUNCTION  =========================================={{{ 
//         Name:  fourierShells
//  Description:  Compute the outer shell in real space.
//  Assume: L(1) = Lx <= Ly = L(2)
//  Since L(1) < L(2), we increase M(2) each round and choose 
//  M(1) = ceil(Lx/Ly*M(2)).
// =============================================================================
void
fourierShells ( /* argument list: {{{*/
    MatrixOnHost & A, MatrixOnHost & absA
    , const MatrixOnHost & x, const MatrixOnHost & x0
    , const MatrixOnHost & newM, const MatrixOnHost & oldM
    , const MatrixOnHost & L // Assume that Lx <= Ly <= Lz
    , const double & a
    , const double & d, const double & e
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* fourierShells implementation: {{{*/
  assert((3*x.columns() <= A.rows()) && (3*x0.columns() <= A.columns()));
  MatrixOnHost xh(3, 1);
  MatrixOnHost tA(3, 3);
  MatrixOnHost l(3, 1);
  double Lx = L(0), Ly = L(1), Lz = L(2);
  double tau = Lx*Ly*Lz;
  if ((newM(0)==oldM(0)) && (newM(1)==oldM(1))) { // when there is NO shell, we compute zero order term
    return;
  }
  // If there is a shell of finite thickness, 
  for (int i = 0; i < x.columns(); i++) {
    for (int j = 0; j < x0.columns(); j++) {
      for (int k = 0; k < 3; k++) xh(k) = x(k, i) - x0(k, j);
      // compute left and right cells, assuming that Lx < Ly
      //      Ly
      // |--|--|--|--|
      // | .| .| .| .|
      // |  |  |  |  |
      // |--|--|--|--|  Lx 
      // | .| .| .| .|
      // |  |  |  |  |
      // |--|--|--|--|
      // Compute top and bottom cells
      for (int l3 = oldM(2)+1; l3 <= newM(2); l3++) {
        for (int l2 = -newM(1); l2 <= newM(1); l2++) {
          for( int l1 = -newM(0); l1 <= newM(0); l1++ ) {
            l(0) = 2*PI*l1/Lx;
            l(1) = 2*PI*l2/Ly;
            l(2) = 2*PI*l3/Lz;
            vF( tA, xh, a, d, l ); 
            for (int x1 = 0; x1 < 3; x1++) {
              for (int x2 = 0; x2 < 3; x2++) {
                tA(x1, x2) /= tau;
                A(3*i + x1, 3*j + x2) += tA(x1, x2);
                absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
              }
            }

            l(2) = -l(2);
            vF( tA, xh, a, d, l ); 
            for (int x1 = 0; x1 < 3; x1++) {
              for (int x2 = 0; x2 < 3; x2++) {
                tA(x1, x2) /= tau;
                A(3*i + x1, 3*j + x2) += tA(x1, x2);
                absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
              }
            }
          }
        }
      }

      // Compute left and right cells
      for (int l2 = oldM(1) + 1; l2 <= newM(1); l2++) {
        for (int l3 = -oldM(2); l3 <= oldM(2); l3++) {
          for( int l1 = -newM(0); l1 <= newM(0); l1++ ) {
            l(0) = 2*PI*l1/Lx;
            l(1) = 2*PI*l2/Ly;
            l(2) = 2*PI*l3/Lz;
            vF( tA, xh, a, d, l ); 
            for (int x1 = 0; x1 < 3; x1++) {
              for (int x2 = 0; x2 < 3; x2++) {
                tA(x1, x2) /= tau;
                A(3*i + x1, 3*j + x2) += tA(x1, x2);
                absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
              }
            }

            l(1) = -l(1);
            vF( tA, xh, a, d, l ); 
            for (int x1 = 0; x1 < 3; x1++) {
              for (int x2 = 0; x2 < 3; x2++) {
                tA(x1, x2) /= tau;
                A(3*i + x1, 3*j + x2) += tA(x1, x2);
                absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
              }
            }
          }
        }
      }

      // Compute front and back cells
      for (int l1 = oldM(0) + 1; l1 <= newM(0); l1++) {
        for (int l3 = -oldM(2); l3 <= oldM(2); l3++) {
          for( int l2 = -oldM(1); l2 <= oldM(1); l2++ ) {
            l(0) = 2*PI*l1/Lx;
            l(1) = 2*PI*l2/Ly;
            l(2) = 2*PI*l3/Lz;
            vF( tA, xh, a, d, l ); 
            for (int x1 = 0; x1 < 3; x1++) {
              for (int x2 = 0; x2 < 3; x2++) {
                tA(x1, x2) /= tau;
                A(3*i + x1, 3*j + x2) += tA(x1, x2);
                absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
              }
            }

            l(0) = -l(0);
            vF( tA, xh, a, d, l ); 
            for (int x1 = 0; x1 < 3; x1++) {
              for (int x2 = 0; x2 < 3; x2++) {
                tA(x1, x2) /= tau;
                A(3*i + x1, 3*j + x2) += tA(x1, x2);
                absA(3*i + x1, 3*j + x2) += abs(tA(x1, x2));
              }
            }
          }
        }
      }
    }
  }
} /*}}}*/
/* ---------------  end of function fourierShells  -------------------- }}}*/

//// === optimalNumShells: FUNCTION  =========================================={{{ 
////         Name:  optimalNumShells
////  Description: Compute the optimal number of cells in x- and y-directions.
////  Input      : box size L = (Lx, Ly)
////             : blob parameter e
////             : splitting parameter d
////             : tolerance tol
////  Output     : number of cells in each direction M = (Mx, Nx)
////             : computing time for each sum  timeM  = (realSum, fourierSum)
//// =============================================================================
//void
//optimalNumShells ( /* argument list: {{{*/
//    MatrixOnHost & M, MatrixOnHost & timeM
//    , const MatrixOnHost & L
//    , const MatrixOnHost & x0
//    , const MatrixOnHost & x
//    , const MatrixOnHost & refSol
//    , const double & d, const double & e
//    , int * numPoints // number of sample points in each direction
//    , const double & tol = 1e-15
//    , const int & maxShell = 10 
////    , const double & MatrixOnHost x0 = MatrixOnHost(3, 1, 0)
//    ) /* ------------- end of argument list -----------------------------}}}*/ 
//{ /* optimalNumShells implementation: {{{*/
//  MatrixOnHost newM(2, 1), oldM(2, 1);
//  double Lx = L(0), Ly = L(1), Lz = L(2);
//
////  MatrixOnHost x(3, numPoints[0]*numPoints[1]*numPoints[2]);
////  { // set up sample points
////    double dx = Lx/numPoints[0], dy = Ly/numPoints[1], dz = Lz/numPoints[2];
////    for (int i = 0; i < numPoints[0]; i++) {
////      for (int j = 0; j < numPoints[1]; j++) {
////        for (int k = 0; k < numPoints[2]; k++) {
////          x(0, i + j * numPoints[0] + k * numPoints[0] * numPoints[1]) = i * dx;
////          x(1, i + j * numPoints[0] + k * numPoints[0] * numPoints[1]) = j * dy;
////          x(2, i + j * numPoints[0] + k * numPoints[0] * numPoints[1]) = k * dz + 0.005;
////        } 
////      }
////    }
////  }
//
//  MatrixOnHost newA(3*x.columns(), 3*x0.columns());
//  MatrixOnHost absA = newA;
//
//  // Compute the exact value using a large number of cells in each directions
////  newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
////  realShells ( refSol, absA, x, x0, newM, oldM, L, d, e );
////  fourierShells(A, absA, x, x0, newM, oldM, L, d, e);
////  refSol = refSol + A;
////  newM(0) = newM(1) = maxShell;
////  realShells ( refSol, absA, x, x0, newM, oldM, L, d, e );
////  fourierShells(refSol, absA, x, x0, newM, oldM, L, d, e);
////  newM(0) = newM(1) = maxShell;
////  realShells ( refSol, absA, x, x0, newM, oldM, L, d, e );
////  fourierShells(refSol, absA, x, x0, newM, oldM, L, d, e);
//
//  double err = 10;
//  int numLoop = 1;
//
////  boost::timer::cpu_timer timer;
////  boost::timer::cpu_times elapsed;
//  
//  clock_t start, end;
//  // compute the real space sum
////  timer.start();
////  start = clock();
////  newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
////  realShells ( newA, absA, x, x0, newM, oldM, L, d, e );
////  for (int i = 0; i < maxShell; i++) {
////    oldM(0) = newM(0); oldM(1) = newM(1);
////    newM(0)++;
////    newM(1) = ceil ( newM(0) * Lx/Ly );
////    realShells ( newA, absA, x, x0, newM, oldM, L, d, e );
////  }
////  A = newA;
////  newA.setElements(0);
////  newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
////  fourierShells ( newA, absA, x, x0, newM, oldM, L, d, e );
////  for (int i = 0; i < maxShell; i++) {
////    oldM(0) = newM(0); oldM(1) = newM(1);
////    newM(1)++;
////    newM(0) = ceil ( newM(1) * Lx/Ly );
////    fourierShells ( newA, absA, x, x0, newM, oldM, L, d, e );
////  }
////  A = A + newA;
//  
//  newA.setElements(0);
//  newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
//  realShells ( newA, absA, x, x0, newM, oldM, L,  d, e );
//  err = 10;
//  numLoop = 1;
//  while ( (err > tol) && (numLoop < maxShell) ) {
//    oldM(0) = newM(0); oldM(1) = newM(1);
//    newM(0)++;
//    newM(1) = ceil ( newM(0) * Lx/Ly );
//    absA.setElements(0);
//    realShells ( newA, absA, x, x0, newM, oldM, L, d, e );
//
//    // compute error
//    for ( int i = 0; i < absA.length(); i++ ) {
//      if ((abs(refSol(i)) > eps) || (absA(i) > eps)) {
//        absA(i) = absA(i)/abs(refSol(i));
//      }
//    }
//    err = absA.max();
////    err = 1;
//    numLoop++;
//  }
//
//  M(0) = oldM(0);     // record time and number of shells for real sum
//  std::cout << " Real err is " << err << std::endl;
//  
//  start = clock();
//  newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
//  realShells ( newA, absA, x, x0, newM, oldM, L, d, e );
//  newM(0) = M(0); newM(1) = ceil ( newM(0) * Lx/Ly );
//  realShells ( newA, absA, x, x0, newM, oldM, L, d, e );
//  end = clock();
////  refSol.print("exact sol is");
//
////  elapsed = timer.elapsed();
//  timeM(0) = 1000.0 * ((double) (end - start)) / CLOCKS_PER_SEC / x.columns();
//
////  A = newA;
////  MatrixOnHost B = newA;
//
//  // compute the fourier space sum
////  timer.start();
////  start = clock();
//  newA.setElements(0);
//  newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
//  fourierShells ( newA, absA, x, x0, newM, oldM, L, d, e );
//  err = 10;
//  numLoop = 1;
//      double tmp = 0;
//  while ( (err > tol) && (numLoop < maxShell) ) {
//    oldM(0) = newM(0); oldM(1) = newM(1);
//    newM(1)++;
//    newM(0) = ceil ( newM(1) * Lx/Ly );
//    absA.setElements(0);
//    fourierShells ( newA, absA, x, x0, newM, oldM, L, d, e );
//
//    // compute error
//    for ( int i = 0; i < absA.length(); i++ ) {
//      tmp = absA(i);
//      if ((abs(refSol(i)) > eps) || (absA(i) > eps)) {
//        absA(i) = absA(i)/abs(refSol(i));
//      }
//      //if (absA(i) > 100) {
//        //std::cout << "At numLoop = " << numLoop 
//          //<< " and newM = (" << newM(0) << ", " << newM(1) << "): "
//          //<< "err(" << i << ") = " << absA(i) 
//          //<< " for refSol = " << refSol(i) 
//          //<< " and absA = " << tmp << std::endl;
//      //}
//    }
//    err = absA.max();
////    err = 1;
//    numLoop++;
//  }
//
//  M(1) = oldM(1);  // record time and number of shells for real sum
//  std::cout << " Fourier err is " << err << std::endl;
//  
//  start = clock();
//  newM(0) = newM(1) = oldM(0) = oldM(1) = 0;
//  fourierShells ( newA, absA, x, x0, newM, oldM, L, d, e );
//  newM(1) = M(1); newM(0) = ceil ( newM(1) * Lx/Ly );
//  fourierShells ( newA, absA, x, x0, newM, oldM, L, d, e );
//  end = clock();
//  
////  elapsed = timer.elapsed();
//  timeM(1) = 1000.0 * ((double) (end - start)) / CLOCKS_PER_SEC / (double)x.columns();
//
////  refSol.print("Exact sol is");
////  A = A + newA;
////  A.print("approx is ");
//
////  B = B + newA - A;
//////  MatrixOnHost B = refSol - A;
//// std::cout << "max(refSol - approx.) = " << B.abs().max() << std::endl;
////   for (int i = 0; i < B.length(); i++) {
////     if ((A(i) > es) || (B(i) > eps)) 
////       B(i) = abs(B(i))/abs(A(i));
////   }
//// err = B.max();
//// std::cout << "error is " << err << std::endl;
//// B.print("error is ");
//
//} /*}}}*/
///* ---------------  end of function optimalNumShells  -------------------- }}}*/

#endif
