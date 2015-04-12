/*
 * =====================================================================================
 *
 *       Filename:  periodicBrinkman.cu
 *
 *    Description:  - Compute the suspension of a prolate spheroid.
 *                  - Compare the results for Stokes flow.
 *                  - Suspension of a helical body.
 *
 *        Version:  1.0
 *        Created:  10/12/2014 05:31:10 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Hoang-Ngan Nguyen (), zhoangngan-gmail
 *   Organization:  
 *
 * =====================================================================================
 */

#include "periodicBrinkman.h"
#include <cstdlib>  /* srand, rand */
#include <ctime>    /* time */
#include <string>
#include <fstream>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <map>
using namespace std;
using namespace Eigen;

const int D = 8;
//const int boxSize = 100;
const int maxBitSize = 8;

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
    //double* values, const std::string names[], const int& numel
    map< string, double > & params
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
    char * key;
    char * value;
    param.getline(buf, 80);
    key = strtok(buf, delimiters);
    if (key) {
      value = strtok(NULL, delimiters);
      params[key] = atof(value);
    }
  }
  param.close();
  return 0;
} /*}}}*/
/* ---------------  end of function readData  -------------------- }}}*/

int main( int argc, char *argv[] ) {
  ofstream log("log.txt");
  //string names[] = {"Lx", "Ly", "Lz", "d", "e", "M", "alpha", "boxSize", "toPrint"};
  //double params[] = {1.0, 1.0, 1.0, 0.6, 0, 4, 1, 1, 1};

  map< string, double> params;
  params["Lx"] = 1.0;
  params["Ly"] = 1.0;
  params["Lz"] = 1.0;
  params["d"]  = 0.6;
  params["e"]  = 0.0;
  params["M"]  = 4;
  params["alpha"]  = 1;
  params["boxSize"] = 1;
  params["toPrint"] = 0;
  const char * fileName = "param.txt";
  if (argc > 1) {
    fileName = argv[1];
  }
  readData (params, fileName); 

  int M, N, Mx, My, Mz;
  M = N = Mx = My = Mz = params["M"];
  double a = params["alpha"];
  double e = params["e"];
  MatrixOnHost L(3);
  double Lx, Ly, Lz;
  Lx = L(0) = params["Lx"]; 
  Ly = L(1) = params["Ly"]; 
  Lz = L(2) = params["Lz"];

  double boxSize = params["boxSize"];
  double h = 1.0/boxSize;
  uint nx, ny, nz;
  nx = ny = nz = boxSize;
  double hx, hy, hz;
  hx = h * Lx;
  hy = h * Ly;
  hz = h * Lz;

  // Create a grid inside the periodic box and put the point force at the center
  MatrixOnHost xm(3, nx * ny * nz), x0(3, 1, 0.5);
  for ( uint z = 0; z < nz; z++ ) { 
    for (uint x = 0; x < nx; x++) {
      for (uint y = 0; y < ny; y++) {
        xm(0, x + y*nx + z*nx*ny) = x * hx;
        xm(1, x + y*nx + z*nx*ny) = y * hy;
        xm(2, x + y*nx + z*nx*ny) = z * hz;
      }
    }
  }         

  xm.setRandom();

  x0.print("x0 is\n");
  xm.print("x is \n");

  // MOve data to GPU
  //MatrixOnDevice dx = xm, dx0 = x0;

  // Allocate storage for velocity matrix
  //MatrixOnDevice dA(3*xm.columns(), 3*x0.columns());
  MatrixOnHost   A(3*xm.columns(), 3*x0.columns());
  MatrixOnHost   oldA(3*xm.columns(), 3*x0.columns());
  MatrixOnHost   absA(3*xm.columns(), 3*x0.columns());
  
  bool stop = false;
  
  double dVal[D], err[D], temp;
  double d0= pow(Lx*Ly*Lz, 1.0/3) * SPI;
  d0 = params["d"];
  cout << "optimal d is " << d0 << endl;
  for (int i = 0; i < D; i++) {
    dVal[i] = (0.1 + i*0.95)*d0;
  }

  //uint dimGridx   = chooseSize(dx.columns());
  //uint dimGridy   = chooseSize(dx0.columns());
  //dim3 dimGrid (dimGridx, dimGridy);

  //uint blockSize  = (2*Mx + 1)*(2*My + 1)*(2*Mz + 1);
  //blockSize = chooseSize(blockSize, maxBitSize);

  //cout << "blockSize = " << blockSize << ", and dimGridx = " << dimGridx
    //<< ", dimGridy = " << dimGridy << endl;

  //dim3 dimBlock (blockSize);
  //uint bitSize = 0;
  //while ((1 << bitSize) < blockSize) {
    //bitSize++;
  //}

  int mVal[D] = {};

  //periodicBrinkman (dA, dx, dx0, a, .25, e, M, N, L);
  //oldA = dA;
  //for (int d = 0; d < D; d++) {
    //periodicBrinkman (dA, dx, dx0, a, dVal[d], e, M, N, L);
    //A = dA;
    //A.print("A using GPU is");
    //err[d] = abs(A - oldA).max();
  //}

  MatrixOnHost newM(3, 1, M), oldM(3, 1);
  A.setElements(0);
  newM.setElements(0);
  realShells( A, oldA, xm, x0, newM, oldM, L, a, d0, e );
  newM.setElements(M);
  realShells( A, oldA, xm, x0, newM, oldM, L, a, d0, e );
  fourierShells( A, oldA, xm, x0, newM, oldM, L, a, d0, e );
  absA = A;
  for (int d = 0; d < D; d++) {
  A.setElements(0);
  newM.setElements(0);
  realShells( A, oldA, xm, x0, newM, oldM, L, a, dVal[d], e );
  newM.setElements(M);
  realShells( A, oldA, xm, x0, newM, oldM, L, a, dVal[d], e );
  fourierShells( A, oldA, xm, x0, newM, oldM, L, a, dVal[d], e );
  //cout << "At d = " << dVal[d] << ", ";
  //A.print("Matrix A is");
    err[d] = abs(A - absA).max();
  }


  log << "For Lx = " << Lx << ", Ly = " << Ly << ", Lz = " << Lz << ", and a = "
    << a << ", we have: " << endl;
  for (int i = 0; i < D; i++) {
    log.precision(10);
    log << scientific << dVal[i] << " : " << mVal[i] 
      << " : " << err[i] << endl;
  }

  double toPrint = params["toPrint"];
  //cout << "toPrint = " << toPrint << endl;
  //cout << "boxSize = " << params[6] << endl;
  //if (toPrint > 0) A.print();

	cout << "Done!!!!!!!!!!!!!!" << endl;
  cout << "Values of parameters are " << endl;
  cout << "Lx = " << Lx << ", Ly = " << Ly
   << ", Lz = " << Lz << endl
   << "d = " << d0 << ", e = " << e << ", alpha = " << a << endl
   << "M = " << M  
   << ", boxSize = " << nx << ", toPrint = " << toPrint << endl;

	return EXIT_SUCCESS;
}				// ----------  end of function main  ----------
