#include <stdlib.h>
#include <iostream>
#include <cassert>
#include "Exception.hpp"
#include "Vector.hpp"
#include "Matrix.hpp"
#include <cstdio>
#include <ctime>
#include <chrono>
#include <math.h>
#include <fstream> //write to file
#include <iomanip> //set precision

int main(int argc, char* argv[])
{
double pi = 3.1415926535897;


// Set up matrix system

int n = 20;
Vector a1 = ones(n-1);
// disp(a1);
Matrix A = 4*eye(n) - diag(a1,1) - diag(a1,-1);
// disp(A);
Matrix B1 = eye(n);
Matrix C1 = kron(B1,A);
// disp(C1);
Matrix B2 = diag(a1,1) + diag(a1,-1);
Matrix C2 = kron(B2,-eye(n));
// disp(C2);
Matrix C = C1+C2;
// disp(C);
//RHS

Vector b(n*n);
double dx = 1.0/(n-1);
Vector xco(n);
Vector yco(n);
for (int i = 2; i<=n; i++)
{
  xco(i) = (i-1)*dx;
  yco(i) = (i-1)*dx;
}

disp(xco);

int fillcount = 1;

for(int i = 1; i<=n; i++)
{
  // std::cout<<i<<"\n";
  for (int j = 1; j<=n; j++)
  {
      b(fillcount) = ((pi*pi)*(xco(j)*xco(j))*(1-xco(j)) + 6*xco(j) - 2)*sin(pi*yco(i));
      // std::cout<<b(fillcount)<<"\n";
      fillcount += 1;
  }
}

b = dx*dx*b;
// disp(b);

// // apply and time methods

//Vector times(7);
////
//std::clock_t LUstart = std::clock();
//Vector x_lu = C/b;
//double lutime = (std::clock() - LUstart) / (double)CLOCKS_PER_SEC;
//std::cout << "Finished in " << lutime << " seconds [CPU Clock] " << std::endl;

// std::clock_t jacobistart = std::clock();
// Vector x_j = jacobi(C,b);
// double jacobitime = (std::clock() - jacobistart) / (double)CLOCKS_PER_SEC;
// std::cout << "Finished in " << jacobitime << " seconds [CPU Clock] " << std::endl;
//
// std::clock_t GSstart = std::clock();
// Vector x_gs = GS(C,b);
// double GStime = (std::clock() - GSstart) / (double)CLOCKS_PER_SEC;
// std::cout << "Finished in " << GStime << " seconds [CPU Clock] " << std::endl;
//
// std::clock_t SORstart = std::clock();
// Vector x_sor = SOR(C,b,1.7406);
// double SORtime = (std::clock() - SORstart) / (double)CLOCKS_PER_SEC;
// std::cout << "Finished in " << SORtime << " seconds [CPU Clock] " << std::endl;
//
// std::clock_t CGstart = std::clock();
// Vector x_cg = CG(C,b);
// double CGtime = (std::clock() - CGstart) / (double)CLOCKS_PER_SEC;
// std::cout << "Finished in " << CGtime << " seconds [CPU Clock] " << std::endl;
//
//
// std::clock_t GMRESstart = std::clock();
// Vector x_gmres = GMRES(C,b);
// double GMREStime = (std::clock() - GMRESstart) / (double)CLOCKS_PER_SEC;
// std::cout << "Finished in " << GMREStime << " seconds [CPU Clock] " << std::endl;
//
//
// std::clock_t MINRESstart = std::clock();
// Vector x_minres = MINRES(C,b);
// double MINREStime = (std::clock() - MINRESstart) / (double)CLOCKS_PER_SEC;
// std::cout << "Finished in " << MINREStime << " seconds [CPU Clock] " << std::endl;
// //
// // disp(x_gmres);
// times(1) = jacobitime; times(2) = GStime; times(3) = SORtime;
// times(4) = CGtime; times(5) = GMREStime; times(6) = MINREStime;
//times(7) = lutime;
//
//disp(times);

// std::ofstream CPUtimes;
// CPUtimes.open("CPUtimes.txt");
// CPUtimes << std::setprecision(15) << times << "\n";
// CPUtimes.close();



// // Convergence with mesh size


// std::ofstream MMSconvergence;
// MMSconvergence.open("MMSconvergence.txt");
//
// int n=5;
// for (int k = 1; k<=5; k++)
// {
//   Vector a1 = ones(n-1);
//   // disp(a1);
//   Matrix A = 4*eye(n) - diag(a1,1) - diag(a1,-1);
//   // disp(A);
//   Matrix B1 = eye(n);
//   Matrix C1 = kron(B1,A);
//   // disp(C1);
//   Matrix B2 = diag(a1,1) + diag(a1,-1);
//   Matrix C2 = kron(B2,-eye(n));
//   // disp(C2);
//   Matrix C = C1+C2;
//   // disp(C);
//   //RHS
//
//   Vector b(n*n);
//   double dx = 1.0/(n-1);
//   Vector xco(n);
//   Vector yco(n);
//   for (int i = 2; i<=n; i++)
//   {
//     xco(i) = (i-1)*dx;
//     yco(i) = (i-1)*dx;
//   }
//
//   disp(xco);
//
//   int fillcount = 1;
//
//   for(int i = 1; i<=n; i++)
//   {
//     std::cout<<i<<"\n";
//     for (int j = 1; j<=n; j++)
//     {
//         b(fillcount) = ((pi*pi)*(xco(j)*xco(j))*(1-xco(j)) + 6*xco(j) - 2)*sin(pi*yco(i));
//         std::cout<<b(fillcount)<<"\n";
//         fillcount += 1;
//     }
//   }
//
//   b = dx*dx*b;
//   Vector x = C/b;
//   MMSconvergence << std::setprecision(15) << x << "\n";
//   n +=5;
//   std::cout<< n << "\n";
// }
// MMSconvergence.close();


}
