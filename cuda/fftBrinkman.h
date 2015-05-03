// ===========================================================================
//
//       Filename:  fftBrinkman.h
//        Version:  1.0
//        Created:  04/19/2015 09:03:16 PM
//       Revision:  none
//       Compiler:  nvcc, g++
//      Copyright:  Copyright (c) 2015, Hoang-Ngan Nguyen
//          Email:  zhoangngan [@] gmail
//        License:  MIT License
//
//    Description:  Compute periodic Brinkman in 3D using FFT.
//      In order to do this, we must know the strength of the force at each
//      location to get an O(NlogN) computation time.
//
// ===========================================================================

#ifndef FFT_BRINKMAN_H__ 
#define FFT_BRINKMAN_H__ 
#include "Brinkman.h"

// :TODO: -write unit test for gridded Gaussian and its Fourier transform.

// ====  fftBrinkman : Compute 3D periodic Brinkman flow using FFT for k-space sum. ={{{
// Tolerance: tol
// number of particles N = xn.columns() = f.columns();
// Real space sum:
//  sigma^2 = delta^2/2;
//  rc = mr * sigma
//  4*pi/3*rc^3*N = C  (# of points inside the ball radius rc)
//  mr^2 = 2*ln(tol/\sum||f||_n);
//
// =====================================================================================
  void
fftBrinkman ( 
          MatrixOnHost & U      // 3-by-m matrix: velocities
  , const MatrixOnHost & y      // 3-by-m matrix: observation points
  , const MatrixOnHost & f      // 3-by-N matrix: forces
  , const MatrixOnHost & xn     // 3-by-N matrix: force locations
  , const MatrixOnHost & L      // 3-by-1 matrix: box sizes
  , const int * const M        // grid sizes
  , const int * const P        // (P-1) is the number of grid points inside the 'support'
                                // P is preferably an even number
  , const int & m              // width of support = m*standard deviation
  , const int & C              // average # of particles around a particle
  , const double & e            // blob parameter
  , const double & tol          // tolerance for error in real and reciprocal space sum
  , const double & d            // splitting parameter
  , const double & alpha = 1    // inverse of permeability
  )
{
  uint N = xn.columns();                                // number of particles
  double mr = sqrt(-2*log(tol));                        // real space: width of support = mr*standard deviation
  //double d = sqrt(2)/mr*pow(3.0*C/(4*PI*N), 1.0/3.0);   // splitting parameter d > e
  std::cout << "inside fftBrinkman" << std::endl;
  std::cout << "mr = sqrt(-2*log(" << tol << ")) = " << mr << std::endl;
  std::cout << "splitting parameter = " << d << std::endl;

  fftw_complex * Hx[3], *Hk[3], *Gk[3], *Gx[3];
  uint gridSize = M[0] * M[1] * M[2];
  for( int i = 0; i < 3; i++ ) {
    // gridded Gaussian
    Hx[i] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * gridSize);
    Gx[i] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * gridSize);

    // Fourier transform of gridded Gaussian
    Hk[i] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * gridSize);
    Gk[i] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * gridSize);
  }

  // create fft plans
  fftw_plan forwardPlan[3], inversePlan[3];
  for( int i = 0; i < 3; i++ ) {
    forwardPlan[i] = fftw_plan_dft_3d(M[0], M[1], M[2], Hx[i], Hk[i], FFTW_FORWARD , FFTW_ESTIMATE);
    inversePlan[i] = fftw_plan_dft_3d(M[0], M[1], M[2], Gk[i], Gx[i], FFTW_BACKWARD, FFTW_ESTIMATE);
  }

  // When xn is inside the box, the left farthest point of the support has index
  // -P/2 and the right farthest point of the support has index P/2 + 1.
  //   |--|--x--|--|----....-----|--|--x--|--|
  // -P/2---------------------------------- P/2 + 1
  //
  // In addition, in order computation time when computing 
  //    exp(-x*x - y*y - z*z) 
  // we separate variables
  //    exp(-x*x - y*y - z*z) = exp(-x*x)*exp(-y*y)*exp(-z*z);
  // Thus, instead of evaluating the exponential function 
  //    N * Px * Py * Pz times, 
  // we only need to evaluate the exponential function
  //    N * ( Px + Py + Pz ) times, 
  // and
  //    N * Px * Py * Pz multiplications.

  // reset data to 0's
  for( int i = 0; i < gridSize; i++ ) {
    Hx[0][i][1] = Hx[0][i][0] = 0;
    Hx[1][i][1] = Hx[1][i][0] = 0;
    Hx[2][i][1] = Hx[2][i][0] = 0;
  }

  double hx  = L(0)/M[0];
  double hy  = L(1)/M[1];
  double hz  = L(2)/M[2];

  MatrixOnHost expX(P[0], N), expY(P[1], N), expZ(P[2], N);
  //int Plx = (P[0] - 1) >> 1;
  //int Ply = (P[1] - 1) >> 1;
  //int Plz = (P[2] - 1) >> 1;
  //double eta = (P[0] - 1)*hx/m/d;
  int Plx = (P[0]) >> 1;
  int Ply = (P[1]) >> 1;
  int Plz = (P[2]) >> 1;
  double eta = P[0]*hx/d/m;
  eta *= eta*d*d/2;

  // coefficient in front of the Gaussian
  double scale = 1.0/eta/PI;
  scale = scale*sqrt(scale);

  for( int n = 0; n < N; n++ ) { /* gridded Gaussian {{{ */
    // find the position of xn on the grid in k-space
    int ix = 0; while( ix*hx <= xn(0, n) ) ix++; ix--;
    int iy = 0; while( iy*hy <= xn(1, n) ) iy++; iy--;
    int iz = 0; while( iz*hz <= xn(2, n) ) iz++; iz--;

    double temp = 0;
    for( int px = 0; px < P[0]; px++ ) {
      temp = (px - Plx + ix)*hx - xn(0, n);
      expX(px, n) = exp(-temp*temp/eta);
    }
    for( int py = 0; py < P[1]; py++ ) {
      temp = (py - Ply + iy)*hy - xn(1, n);
      expY(py, n) = exp(-temp*temp/eta);
    }
    for( int pz = 0; pz < P[2]; pz++ ) {
      temp = (pz - Plz + iz)*hz - xn(2, n);
      expZ(pz, n) = exp(-temp*temp/eta);
    }

    // now compute the gridded Gaussian
    // only grid points inside the support of the Gaussian are computed
    for( int px = 0; px < P[0]; px++ ) {
      int jx = (ix + px - Plx + M[0]) % M[0];
      for( int py = 0; py < P[1]; py++ ) {
        int jy = (iy + py - Ply + M[1]) % M[1];
        for( int pz = 0; pz < P[2]; pz++ ) {
          temp = scale*expX(px, n)*expY(py, n)*expZ(pz, n);
          int jz = (iz + pz - Plz + M[2]) % M[2];
          Hx[0][(jx*M[1] + jy)*M[2] + jz][0] += f(0, n)*temp;
          Hx[1][(jx*M[1] + jy)*M[2] + jz][0] += f(1, n)*temp;
          Hx[2][(jx*M[1] + jy)*M[2] + jz][0] += f(2, n)*temp;
        }
      }
    }
  }  /* end of for loop to compute gridded Gaussian }}}*/

  // now perform FFT on the gridded Gaussian
  for( int i = 0; i < 3; i++ ) fftw_execute(forwardPlan[i]);

  // reset data to prepare for the new gridded Gaussian used in integration
  for( int i = 0; i < gridSize; i++ ) {
    Hx[0][i][0] =  0;
  }

  // Scale the computed transform by 1/gridSize to match with theoretical
  // formulation.
  for( int i = 0; i < 3; i++ ) {
    for( int j = 0; j < gridSize; j++ ) {
      Hk[i][j][0] /= gridSize;
      Hk[i][j][1] /= gridSize;
    }
  }

  // move the transformed data so that it is centered at zero wave number k = 0
  for( int i = 0; i < 3; i++ ) 
    shiftedFFT(Hk[i], Gk[i], M[0], M[1], M[2], FFTW_FORWARD);
  

  // compute new Gk
  int startKx = (M[0]) >> 1;
  int startKy = (M[1]) >> 1;
  int startKz = (M[2]) >> 1;
  double k[3] = {0, 0, 0};
  double vHk[3] = {0, 0, 0};
  double xi = alpha*d*alpha*d;
  double a1 = 1 - exp(-xi/4);
  double a0 = xi + 6*a1;
  for( int kx = 0; kx < M[0]; kx++ ) {
    k[0] = (kx - startKx)*2*PI/L[0];
    for( int ky = 0; ky < M[1]; ky++ ) {
      k[1] = (ky - startKy)*2*PI/L[1];
      for( int kz = 0; kz < M[2]; kz++ ) {
        k[2] = (kz - startKz)*2*PI/L[2];
        double kn = dot(k, k);
        double w  = d*d*kn;
        double bhat = 1/xi*(a1*(w - 6) + a0)/kn/(kn + alpha*alpha);
        bhat *= exp(-kn*d*d/4 + eta*kn/2);

        int index = (kx*M[1] + ky)*M[2] + kz;

        // zero-wave number is excluded
        if( (k[0] == 0) && (k[1] == 0) && (k[2] == 0) ) {
          for( int j = 0; j < 3; j++ ) {
            Gk[j][index][0] = Gk[j][index][1] = 0;
          }
          continue;
        }

        for( int i = 0; i < 2; i++ ) {
          for( int j = 0; j < 3; j++ )
            vHk[j] = Gk[j][index][i];
          // (1 - 2*i) since we need to take the conjugate of the Fourier
          // transform of the gridded Gaussian.
          for( int j = 0; j < 3; j++ )
            Hk[j][index][i] = (1 - 2*i)*bhat*(kn*vHk[j] - k[j]*dot(k, vHk));
        }
      }
    }
  }

  // move the 0-centered data to the one that starts at k = 0
  for( int i = 0; i < 3; i++ ) 
    shiftedFFT( Hk[i], Gk[i], M[0], M[1], M[2], FFTW_BACKWARD );

  for( int i = 0; i < 3; i++ ) fftw_execute(inversePlan[i]);

  expX = MatrixOnHost(P[0], y.columns());
  expY = MatrixOnHost(P[1], y.columns());
  expZ = MatrixOnHost(P[2], y.columns());

  for( int n = 0; n < y.columns(); n++ ) { /* gridded Gaussian {{{ */
    // find the position of xn on the grid in k-space
    int ix = 0; while( ix*hx <= y(0, n) ) ix++; ix--;
    int iy = 0; while( iy*hy <= y(1, n) ) iy++; iy--;
    int iz = 0; while( iz*hz <= y(2, n) ) iz++; iz--;

    double temp = 0;
    for( int px = 0; px < P[0]; px++ ) {
      temp = (px - Plx + ix)*hx - y(0, n);
      expX(px, n) = exp(-temp*temp/eta);
    }
    for( int py = 0; py < P[1]; py++ ) {
      temp = (py - Ply + iy)*hy - y(1, n);
      expY(py, n) = exp(-temp*temp/eta);
    }
    for( int pz = 0; pz < P[2]; pz++ ) {
      temp = (pz - Plz + iz)*hz - y(2, n);
      expZ(pz, n) = exp(-temp*temp/eta);
    }

    // compute the gridded Gaussian with respect to observation point
    // only grid points inside the support of the Gaussian are computed
    for( int px = 0; px < P[0]; px++ ) {
      int jx = (ix + px - Plx + M[0]) % M[0];
      for( int py = 0; py < P[1]; py++ ) {
        int jy = (iy + py - Ply + M[1]) % M[1];
        for( int pz = 0; pz < P[2]; pz++ ) {
          int jz = (iz + pz - Plz + M[2]) % M[2];
          Hx[0][(jx*M[1] + jy)*M[2] + jz][0] = 
            scale*expX(px, n)*expY(py, n)*expZ(pz, n);
        }
      }
    }

    // Now compute the velocity using trapezoidal rule
    // Since the function is periodic we just take the sum.
    for( int i = 0; i < 3; i++ ) {
      U(i, n) = 0;
      for( int j = 0; j < gridSize; j++ )
        U(i, n) += Hx[0][j][0]*Gx[i][j][0];
      U(i, n) *= hx*hy*hz;
    }
  }  /* end of for loop to compute gridded Gaussian }}}*/

  
  //fftw_destroy_plan(forwardPlan);
  //fftw_destroy_plan(inversePlan);
  for( int i = 0; i < 3; i++ ) {
    fftw_free(Hk[i]);
    fftw_free(Hx[i]);
    fftw_free(Gk[i]);
    fftw_free(Gx[i]);
    fftw_destroy_plan(forwardPlan[i]);
    fftw_destroy_plan(inversePlan[i]);
  }
  
}		// -----  end of function fftBrinkman  -----}}}

#endif // FFT_BRINKMAN_H__
