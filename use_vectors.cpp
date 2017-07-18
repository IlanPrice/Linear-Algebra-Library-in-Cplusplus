// Testing //

#include <stdlib.h>
#include <iostream>
#include <cassert>
#include "Exception.hpp"
#include "Vector.hpp"
#include "Matrix.hpp"
#include <ctime>
#include <chrono>


int main(int argc, char* argv[])
{

	/*
//This would  produce a compiler warning (there is no default constructor)
  // Vector badly_formed;

  Vector a_vector(2);
  a_vector(1)=10.0;
  a_vector(2)=20.0;

  //Show that friends and methods can be used to do the same thing
  assert ( a_vector.norm() == norm(a_vector));
  assert ( a_vector.norm(3) == norm(a_vector, 3));

  std::cout << "a_vector = " << a_vector << "\n";

  Vector bigger_vector(3);
  Vector smaller_vector(1);
  std::cout << "The following produces a warning\n";
  // This produces a warning
  bigger_vector = a_vector;
  std::cout << "The following throws an exception\n";
  // This throws an exception
  try
  {
      smaller_vector = a_vector;
  }
  catch (Exception &ex)
  {
      ex.DebugPrint();
  }
  std::cout << "The following throws an exception\n";
  // This throws an exception
  try
  {
      a_vector/2.0;//Okay
      a_vector/0.0;
  }
  catch (Exception &ex)
  {
      ex.DebugPrint();
  }
  exit(0);*/



	/*
Matrix test(2,3);
std::cout<<test;
for (int i=1;i<=2;i++)
{
	for(int j=1;j<=3; j++)
		test(i,j) = i+j;
}

std::cout<<"test nows looks like:\n"<< test;


Matrix I = eye(3);

std::cout<<"I = \n"<<I;

Matrix D = test*I;
std::cout<<"\n\n\n"<<D;

D = 2*D;

std::cout<<"\n\n\n"<<D;

D = D*3;

std::cout<<"\n\n\n"<<D;

D = -D;

std::cout<<"\n\n\n"<<D<<"\n";

std::cout<<"\n"<<size(D)<<"\n";

Matrix E(2,3);
E(1,1) = 1;
E(2,2) = 2;

std::cout<<"\n\n\n"<<E<<"\n";

Matrix C = D+E;

std::cout<<"\n\n\n"<<C<<"\n";

C = D-I;

std::cout<<"\n\n\n"<<C<<"\n";

std::cout<<"Testing creating a diagonal matrix:\n";
*/
/*Vector v(6);

for (int i=0;i<6;i++)
{
	v(i+1) = i;
}

//Matrix F = diag(v);
std::cout<<v<<"\n";
//std::cout<< F<<"\n";

Matrix G = eye(6);
std::cout<<G<<"\n";

std::cout<<length(v)<<"\n"<<size(G)<<"\n";

std::cout<< G*v <<"\n\n"<<v*G<<"\n";

G(1,3) = 30;
G(4,5) = 10;

Matrix T = !G;
std::cout<< G<<"\n\n"<<T;

std::cout<<"Reached end \n";*/
/*
Matrix A(2,2);
A(1,1) = 2;
A(1,2) = 6;
A(2,1) = 6;
A(2,2) = 20;

Vector b(2);
b(1) = 1;
b(2) = -1;

std::cout<<"Matrix A is:\n"<<A<<"\n \n and vector b is: \n"<<b<<"\n";
Vector x = CG(A,b);
std::cout<<"The vector x is:\n"<<x<<"\n\n";

std::cout<<"A*b = \n"<<A*x<<"\n";

*/

/*
Vector v(6);

for (int i=0;i<6;i++)
{
	v(i+1) = i;
}

//Matrix F = diag(v);

Matrix FF = diag(v,2);

Matrix FFF = diag(v);
std::cout<< v <<"\n\n";
std::cout<<FFF<<"\n\n";

Vector newdiag = diag(FFF);

std::cout<<newdiag<<"\n\n";
*/
/*
Vector vv(5);

for (int i=0;i<5;i++)
{
	vv(i+1) = 3;
}

Vector vvv(6);

for (int i=0;i<6;i++)
{
	vvv(i+1) = 2;
}

Matrix P = diag(vvv) + diag(vv,1) + (2*diag(vv,1));
Matrix Q = diag(vvv) + (5*diag(vv,-1)) + diag(vv,1);

Matrix S = P*Q;
Matrix SS = ewisemult(P,Q);

std::cout<< P <<"\n\n";
std::cout<< Q <<"\n\n";
std::cout<< S <<"\n\n";
std::cout<<SS<<"\n\n";

//std::cout<<SS(1,4,1,4)<<"\n\n";

Vector I = SS.getElements(1,6,3,3, "Vector");

std::cout<<I<<"\n\n";

Matrix B = eye(3);
SS.setElements(2,4,2,4, B);

std::cout<<SS<<"\n\n";
*/
/*
Vector vv(4);
for (int i=1;i<=4;i++)
{
	vv(i) = i;
}

Vector vvv(4);
vvv(1) = 1; vvv(2) = 0; vvv(3) = 1; vvv(4) = 0;

Matrix Out = outer(vv,vvv);

Matrix out = outer(vvv,vv);

std::cout<< Out <<"\n\n";
std::cout<< out <<"\n\n";
*/



//int n=4;
//Matrix A = eye(n);
//
//A(1,1) = 9; A(1,2) = 3; A(1,3) = 0; A(1,4) = 0;
//A(2,1) = 3; A(2,2) = 6; A(2,3) = 5; A(2,4) = 0;
//A(3,1) = 0; A(3,2) = 5; A(3,3) = 5; A(3,4) = 9;
//A(4,1) = 0; A(4,2) = 0; A(4,3) = 9; A(4,4) = 9;
//std::cout<< A <<"\n\n";
//int n=3;
//Matrix A = eye(n);
//
//A(1,1) = 12; A(1,2) = -51; A(1,3) = 4;
//A(2,1) = 6; A(2,2) = 167; A(2,3) = -68;
//A(3,1) = -4; A(3,2) = 24; A(3,3) = -41;
//std::cout<< A <<"\n\n";

//std::pair<Matrix, Matrix> qr = QR(A);
//
//Matrix Q = qr.first;
//Matrix R = qr.second;
//std::cout<< Q <<"\n\n";
//std::cout<< R <<"\n\n";
//
//Matrix Recon = Q*R;
//Matrix IDcheck = !Q * Q;
//
//std::cout<< Recon <<"\n\n";
//
//std::cout<< IDcheck <<"\n\n";


//// Test hess(A) to get toupper hessenberg
//int n=3;
//Matrix A = eye(n);
//
//A(1,1) = -149; A(1,2) = -50; A(1,3) = -154;
//A(2,1) = 537; A(2,2) = 180; A(2,3) = 546;
//A(3,1) = -27; A(3,2) = -9; A(3,3) = -25;
//std::cout<< A <<"\n\n";
//
//Matrix H = hess(A);
//
//std::cout<< H <<"\n\n";


//std::pair<Vector,Matrix> eigs = eig(A);
//Vector evals = eigs.first;
//Matrix evecs = eigs.second;
//
//std::cout<< evals <<"\n\n";
//std::cout<< evecs <<"\n\n";
//
//Vector evec1 = evecs.getElements(1,1,1,4,"Vector");
//double eval1 = evals(1);
//
//Vector LHS_check = A*evec1;
//Vector RHS_check = eval1*evec1;
//
//std::cout<< "LHS=\n"<<LHS_check <<"\n\n";
//std::cout<< "RHS=\n"<<RHS_check <<"\n\n";

//// Testing Givens Rotations
//
//Matrix G = givens(2,1,1,A);
//
//std::cout<< "G=\n"<< G <<"\n\n";
//
//std::cout<< "Rotated=\n"<< G <<"\n\n";
//
//Matrix G2 = givens(3,2,2,G);
//
//std::cout<< "Rotated2=\n"<< G2 <<"\n\n";
//
//Matrix G3 = givens(4,3,3,G2);
//
//std::cout<< "Rotated3=\n"<< G3 <<"\n\n";

//int n=3;
//Matrix A = eye(n);
//
//A(1,1) = 1; A(1,2) = 2; A(1,3) = 3;
//A(2,1) = 0; A(2,2) = 4; A(2,3) = 5;
//A(3,1) = 0; A(3,2) = 0; A(3,3) = 6;
//std::cout<< A <<"\n\n";
//
//Vector b(n);
//b(1) = 3; b(2) = 4; b(3) = 5;
//
//Vector x = backsubst(A,b);
//
//std::cout<< x <<"\n\n";


Matrix A(4,4,
          { 1.0, 2.0 , 3.0, 5.0,
            4.0, 5.0, 6.0, 6.0,
            0.0, 7.0,8.0, 9.0,
            10.0, 4.0, 6.0, 3.0});
// A(1,1) = 2;
// A(1,2) = 6;
// A(2,1) = 6;
// A(2,2) = 20;

Vector b(4);
b(1) = 1;
b(2) = 1;
b(3) = 1;
b(4) = 1;

std::cout<<"Matrix A is:\n"<<A<<"\n \n and vector b is: \n"<<b<<"\n";
Vector x = GMRES(A,b);
std::cout<<"The vector x is:\n"<<x<<"\n\n";

std::cout<<"A*x = \n"<<A*x<<"\n";

/*

	int n=4;
	Matrix A = eye(n);

	A(1,1) = 10; A(1,2) = -1; A(1,3) = 2; A(1,4) = 0;
	A(2,1) = -1; A(2,2) = 11; A(2,3) = -1; A(2,4) = 3;
	A(3,1) = 2; A(3,2) = -1; A(3,3) = 10; A(3,4) = -1;
	A(4,1) = 0; A(4,2) = 3; A(4,3) = -1; A(4,4) = 8;
	std::cout<< A <<"\n\n";

// Matrix g = givens(3,2,1,A);
// disp(g);
//
// Matrix Ahat = g*A;
// disp(Ahat);
// Matrix g2 = givens(4,3,2,Ahat);
// Ahat = g2*Ahat;
// disp(Ahat);
//	boost::tuple<Matrix, Matrix, Matrix> LUP = LU(A);
//	Matrix L = LUP.get<0>();
//	Matrix U = LUP.get<1>();
//	Matrix P = LUP.get<2>();
//	std::cout<<"Matrix L is:\n"<<L<<"\n \n and Matrix U is: \n"<<U<<"\n";
// Vector b(n);
// b(1) = 6; b(2) = 25; b(3) = -11; b(4) = 15;

// std::cout<<"Matrix A is:\n"<<A<<"\n \n and vector b is: \n"<<b<<"\n";

Vector x = GMRES(A,b);

Vector test = A*x;

std::cout<<"A/b with jacobi is:\n"<<x<<"\n \n and A*x is: \n"<<test<<"\n";

*/

/*
Matrix C = eye(4);
std::pair <Vector,Matrix> E = eig(C);
Vector evals = E.first;
Matrix evecs = E.second;

disp(evals);
disp(evecs);
*/

/*

Matrix A = eye(3);
Matrix B(3,2);
B(1,1) = 1; B(1,2) = 2;
B(2,1) = 3; B(2,2) = 4;
B(3,1) = 5; B(3,2) = 6;

Matrix C = kron(A,B);

std::cout<<"C = \n"<<C<<"\n\n";

*/

//Test linspace
/*
Vector x = linspace(0,1,11);
std::cout<<x<<"\n\n";
*/

//Test reshape
/*int n = 20;
Vector x(n);
for (int i=1;i<=n;i++)
{
	x(i) = i;
}

Matrix A = reshape(x,4,5);
disp(A);
*/


/*
int n = 5;
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

Vector b(n*n);
Vector fill = ones(n*n);
b.setValues(1,n*n,fill);

std::chrono::high_resolution_clock::time_point start = start_timer();

Vector x = SOR(C,b,1.5);
stop_timer(start, "SOR");
// Vector x2 = C/b;
disp(x);
// Vector dif = x-x2;
//Matrix ans = reshape(x,n,n);

//disp(ans);

*/

/*Matrix I = eye(4);
Matrix A(3,3);
A(1,1) = 4; A(1,2) = -1; A(1,3) = 0;
A(2,1) = -1; A(2,2) = 4; A(2,3) = -1;
A(3,1) = 0; A(3,2) = -1; A(3,3) = 4;
*/

// Matrix H = hess(A);
// disp(H);
// std::pair <Vector,Matrix> Ev = eig(A);
// Vector E = Ev.first;
// Matrix V = Ev.second;
//
// disp(E);
// disp(V);

/* Test LU

boost::tuple<Matrix, Matrix, Matrix> LUP = LU(I);
Matrix L = LUP.get<0>();
Matrix U = LUP.get<1>();
Matrix P = LUP.get<2>();

disp(L);
disp(U);
disp(P);
disp(L*U);
Matrix S(2,2,{1,2,3.0,4});

disp(S);
*/


// std::pair <Vector,Matrix> Ev2 = eig(H);
// Vector E2 = Ev2.first;
// Matrix V2 = Ev2.second;
//
// disp(E2);
// disp(V2);
//
// Vector v = V.getElements(1,3,2,2,"Vector");
// double lambda = E(2);
//
// disp(v);
// std::cout<<lambda<<"\n";
// disp(A*v);
// disp(lambda*v);

// Matrix A(3,3,{1,2,3,0,4,5,0,0,6});
// Vector b(3); b(1) = 1; b(2) = 2; b(3) = 3;
// Vector x = backsubst(A,b);
// disp(x);
// disp(A);
// A(1,2) = 3;
// disp(A);

//
/*Matrix A(4,4,{1,2,3,4,0,0,0,0,5,6,7,8,9,10,11,12});

disp(A);

std::pair <Vector,Matrix> Ev = eig(A);
Vector E = Ev.first;
Matrix V = Ev.second;

disp(E);
disp(V);

disp(A*V);
disp(diag(E)*V);

*/
// Matrix A(4,3,{1,2,3,4,3,4,2,-1, 4, 5, 3, 6, 8, 8});
//
// disp(A);
// Vector s = size(A);
// disp(s);
// // boost::tuple<Matrix, Matrix, Matrix> LUP = LU(A);
// // Matrix L = LUP.get<0>();
// // Matrix U = LUP.get<1>();
// // Matrix P = LUP.get<2>();
// //
// // disp(L);
// // disp(U);
// // disp(P);
// // disp(L*U);
// // disp(P*A);
// //
// std::pair <Matrix,Matrix> qr = QR(A);
// Matrix E = qr.first;
// Matrix V = qr.second;
//
// disp(E);
// disp(V);
//
// disp(A*V);
// disp(diag(E)*V);

// Matrix B = hess(A);
// disp(B);

// //// Test new givens
// std::pair <double,double> giv = givens(A(2,2),A(3,2));
// double c= giv.first;
// double s= giv.second;
// A.applygivens(2,3, c,s);
//
// disp(A);


//Matrix A(3,3); // Creates a 3x3 Matrix A of zeros
//Matrix B(3,3,{1,2,3,4,5,6,7,8,9}) // creates a 3x3 matrix B with row 1 = 1,2,3; row 2 = 4,5,6; row 3 = 7,8,9
//Matrix C(B); //creates a Matrix C which is a copy of B



}
