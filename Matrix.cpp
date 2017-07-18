////////////////////////////////////////////////////////
///////// MATRIX CLASS FOR LINEAR ALGEBRA .cpp /////////
////////////////////////////////////////////////////////

// Ilan Price
// June 2017
// Produced for a 'special topic' report as part of MSc in Mathematical Modelling and Scientific Computing
// at the University of Oxford. Accompanies a written report.

//  Class written in such a way that code similar to Matlab
//  code may be written

#include <iostream>
#include "Matrix.hpp"
#include <cassert>
#include "Exception.hpp"
#include <stdlib.h> //for abs
#include <algorithm>  // for max
#include "Vector.hpp"
#include <string>
#include <cmath>
#include <math.h>
#include "boost/tuple/tuple.hpp" // for LUP factorisation
#include <initializer_list> // for constructor
#include <fstream> //writing to file
#include <iomanip> // set precision

Matrix::Matrix(int i, int j)
{
	mRows = i;
	mColumns = j;
	mElements = new double*[i];

	for (int k = 0; k<i; k++)
	{
		mElements[k]= new double[j];
		for (int l = 0; l<j; l++)
		{
			mElements[k][l] = 0;
		}
	}
	// std::cout<<"constructor called\n";
}

// creates matrix with entries specified in initialiser list
Matrix::Matrix(int i, int j, std::initializer_list<double> entries)
{

	mRows = i;
	mColumns = j;
	mElements = new double*[i];
	int count = 0;
	int len = entries.size();
	for (int k = 0; k<i; k++)
	{
		mElements[k]= new double[j];
		for (int l = 0; l<j; l++)
		{
			if (count<len){ mElements[k][l] = entries.begin()[count];}
			else {mElements[k][l] = 0.0;}
			count+=1;
		}
	}

}

// copy constructor - creates Matrix with the same entries as 'original'
Matrix::Matrix(const Matrix& original)
{
  mRows = original.mRows;
  mColumns = original.mColumns;
  mElements= new double*[mRows];
  for (int k = 0; k<mRows; k++)
  {
	  mElements[k]= new double[mColumns];
  }
  for (int k = 0; k<mRows; k++)
  {
	  for (int l = 0; l<mColumns; l++)
	  {
		  mElements[k][l] = original.mElements[k][l];
	  }
  }
	// std::cout<<"copy constructor called\n";
}

//Destructor

Matrix::~Matrix()
{
  for (int i = 0; i<mRows; i++)
	  {
	  delete[] mElements[i];

	   }
  delete[] mElements;

}

// Helper function to check dimensions of two matrices are equal

bool EqualDim(const Matrix& A, const Matrix& B)
{
	if (A.mColumns==B.mColumns && A.mRows==B.mRows) {return true;}
	else {return false;}
}

// Helper function to check two matrices can be multiplied

bool CanMultiply(const Matrix& A, const Matrix& B)
{
	if (A.mColumns==B.mRows) {return true;}
	else {return false;}
}

// Display Matrix

std::ostream& operator<<(std::ostream& output, const Matrix& A)
{
	for (int i=0; i<A.mRows; i++)
	{
		output << "[ ";
		for (int j = 0; j<A.mColumns; j++)
			{
				output <<  A.mElements[i][j];
				if (j != A.mColumns-1)
					output  << "  ";
				else
					output  << " ]";
			}
		output << "\n";
	}
	  return output;  // for multiple << operators.
}

void disp(const Matrix& A)
{
  std::cout<<A<<"\n";
}

// Getters and setters for slicing

// All indexing by users input starts from 1, not 0


Matrix Matrix:: getElements(int istart, int iend, int jstart, int jend)
{
	if (istart < 1 || jstart < 1 || iend < 1 || jend < 1)
	    {
	      throw Exception("out of range",
			  "accessing vector through () - index too small");
	    }
	  else if (iend > mRows || jend > mColumns || istart > mRows || jstart > mColumns)
	    {
	      throw Exception("length mismatch",
			  "accessing vector through () - index too high");
	    }

	int sRows = iend - istart + 1; int sColumns = jend - jstart + 1;
	Matrix slice(sRows, sColumns);
	for (int i = 0; i<sRows; i++)
	{
		for (int j = 0; j<sColumns; j++)
		{
			slice.mElements[i][j] = mElements[istart -1 + i][jstart -1 + j];
		}
	}
	return slice;
}

Vector Matrix:: getElements(int istart, int iend, int jstart, int jend, std::string vector) // when you want the output of a slice to be a vector not a matrix object
{
	if (istart < 1 || jstart < 1 || iend < 1 || jend < 1)
	{
	  throw Exception("out of range",
		  "accessing vector through () - index too small");
	}
    else if (iend > mRows || jend > mColumns || istart > mRows || jstart > mColumns)
	{
	  throw Exception("length mismatch",
		  "accessing vector through () - index too high");
	}

	int sRows = iend - istart + 1; int sColumns = jend - jstart + 1;
	Vector slice(std::max(sRows,sColumns));
	if (sRows == 1)
	{
		for (int j = 0; j<sColumns; j++)
		{
			slice(j+1) = mElements[istart -1][jstart -1 + j];
		}
	}
	else if (sColumns == 1)
	{
		for (int i = 0; i<sRows; i++)
		{
			slice(i+1) = mElements[istart -1 + i][jstart -1];
		}
	}
	else throw Exception("Incorrect dimensions",
			"One of the dimensions of the output must be 1 in order to output a vector");

	return slice;
}

void Matrix:: setElements(int istart, int iend, int jstart, int jend, Matrix A)
{
	if (istart < 1 || jstart < 1 || iend < 1 || jend < 1)
	    {
	      throw Exception("out of range",
			  "accessing vector through () - index too small");
	    }
	  else if (iend > mRows || jend > mColumns || istart > mRows || jstart > mColumns)
	    {
	      throw Exception("length mismatch",
			  "accessing vector through () - index too high");
	    }

	int sRows = iend - istart + 1; int sColumns = jend - jstart + 1;
	Matrix slice(sRows, sColumns);
	for (int i = 0; i<sRows; i++)
	{
		for (int j = 0; j<sColumns; j++)
		{
			mElements[istart -1 + i][jstart -1 + j] = A.mElements[i][j];
		}
	}
}

void Matrix:: setElements(int istart, int iend, int jstart, int jend, Vector v)
{
	if (istart < 1 || jstart < 1 || iend < 1 || jend < 1)
		    {
		      throw Exception("out of range",
				  "accessing vector through () - index too small");
		    }
		  else if (iend > mRows || jend > mColumns || istart > mRows || jstart > mColumns)
		    {
		      throw Exception("length mismatch",
				  "accessing vector through () - index too high");
		    }

		if (iend==istart) // if user is setting part of a row to the specified vector
		{
			int sColumns = jend - jstart + 1;
			for (int j = 0; j<sColumns; j++)
			{
				mElements[istart -1][jstart -1 + j] = v(j+1);
			}

		}
		else if (jend == jstart) // if user is setting part of a column to the specified vector
		{
			int sRows = iend - istart + 1;
			for (int i = 0; i<sRows; i++)
			{
				mElements[istart -1 + i][jstart-1] = v(i+1);
			}

		}

		else
		{
			throw Exception("length mismatch",
							  "trying to assign a vector to a space of the form mxn where both m and n are greater than 1");
		}

}

// Defining Matrix addition

Matrix operator+(const Matrix& A, const Matrix& B)
{
	assert(EqualDim(A,B));
	int i = A.mRows;
	int j = A.mColumns;
	Matrix ans(i,j);
	for (int k=0;k<i;k++)
	{
		for (int l=0;l<j;l++)
		{
			ans.mElements[k][l] = A.mElements[k][l] + B.mElements[k][l];
		}
	}
	return ans;
}

// Defining Matrix subtraction

Matrix operator-(const Matrix& A, const Matrix& B)
{
	assert(EqualDim(A,B));
	int i = A.mRows;
	int j = A.mColumns;
	Matrix ans(i,j);
	for (int k=0;k<i;k++)
	{
		for (int l=0;l<j;l++)
		{
			ans.mElements[k][l] = A.mElements[k][l] - B.mElements[k][l];
		}
	}
	return ans;
}

// Defining multiplication of scalar and a Matrix

Matrix operator*(double c, const Matrix& A)
{
	Matrix ans(A.mRows, A.mColumns);
	for (int i=0; i<A.mRows; i++)
	{
		for (int j=0; j< A.mColumns; j++)
		{
			ans.mElements[i][j] = c * A.mElements[i][j];
		}
	}
	return ans;

}

Matrix operator*(const Matrix& A, double c)
{
	return c*A;
}

// Multiplication of matrices

Matrix operator*(const Matrix& A, const Matrix& B)
{
	if (CanMultiply(A,B)==false) {throw Exception("Dimensions mismatch",
			  "cannot perform matrix multiplication");}

	Matrix ans(A.mRows, B.mColumns);

	for (int i = 0; i<A.mRows; i++)
			{
				for( int j=0; j<B.mColumns; j++)
				{
					double temp = 0;
					for (int k=0; k<A.mColumns; k++)
					{
						temp += A.mElements[i][k]*B.mElements[k][j];
					}
					ans.mElements[i][j]=temp;

				}
			}
	return ans;

}

// Multiplication of Matrix by vector and vector by Matrix (outputs vector)
// NOTE: We assume that if the matrix is on the left, the vector is a column, and if the vector is on the left, it is a row

Vector operator*(const Matrix& A, const Vector& v)
{
	if (A.mColumns!=length(v)) { throw Exception("Dimensions mismatch",
			  "cannot perform matrix-vector multiplication");}
	Vector ans(A.mRows);
	for (int i= 0;i<A.mRows;i++)
		{
			for (int j = 0; j<A.mColumns; j++)
			{
				ans(i+1) += A(i+1,j+1)*v(j+1);
			}
		}
	return ans;
}

Vector operator*(const Vector& v, const Matrix& A)
{
	if (A.mRows!=length(v)) { throw Exception("Dimensions mismatch",
			  "cannot perform vector-matrix multiplication");}
	Vector ans(A.mColumns);
	for (int i= 0;i<A.mColumns;i++)
		{
			for (int j = 0; j<A.mRows; j++)
			{
				ans(i+1) += A(i+1,j+1)*v(j+1);
			}
		}
	return ans;
}

// Gaussian elimination with backslash operator

Vector operator/(const Matrix& A, const Vector& b)
{
	boost::tuple<Matrix, Matrix, Matrix> LUP = LU(A);
	Matrix L = LUP.get<0>();
	Matrix U = LUP.get<1>();
	Matrix P = LUP.get<2>();
	Vector newb = P*b;
	Vector y = forwardsubst(L,newb);
	Vector x = backsubst(U,y);
	return x;
}

// definition of vector operator () for indexing
// allows A.mElements[i][j] to be written as A(i+1,j+1), as in Matlab and FORTRAN. NOTE: indexes from 1.

const double& Matrix::operator()(int i, int j) const // 'const' because then can be used in the body of other functions which take const arguments
{

  if (i < 1 || j < 1)
    {
      throw Exception("out of range",
		  "accessing vector through () - index too small");
    }
  else if (i > mRows || j > mColumns)
    {
      throw Exception("length mismatch",
		  "accessing vector through () - index too high");
    }

  return mElements[i-1][j-1];

}

double& Matrix::operator()(int i, int j) // cf. https://stackoverflow.com/questions/123758/how-do-i-remove-code-duplication-between-similar-const-and-non-const-member-func
{
  return const_cast<double&>(static_cast<const Matrix &>(*this)(i, j));
}

// Transpose operator "!". Syntax: A transpose is written as !A

Matrix operator!(const Matrix& A)
{
	Matrix T(A.mColumns, A.mRows);
	for (int i=0;i<A.mRows; i++)
	{
		for (int j=0; j<A.mColumns; j++)
		{
			T(j+1,i+1) = A.mElements[i][j];
		}
	}
	return T;
}

// Unary operator "-"

Matrix operator-(const Matrix& A)
{
  Matrix ans(A.mRows, A.mColumns);

  for (int i=0; i<A.mRows; i++)
  {
	  for (int j=0; j< A.mColumns; j++)
	  {
		  ans.mElements[i][j] = -A.mElements[i][j];
	  }
  }
  return ans;
}

// Defining Assignment

Matrix& Matrix::operator=(const Matrix& A)

{
	assert(EqualDim(*this,A));

	for (int i=0; i<A.mRows; i++)
	{
		for (int j=0; j< A.mColumns; j++)
		{
			mElements[i][j] = A.mElements[i][j];
		}
	}
	return *this;
}

// Perform element wise multiplication of matrices

Matrix ewisemult(const Matrix& A, const Matrix& B)
{
	assert(EqualDim(A,B));
	int i = A.mRows;
	int j = A.mColumns;
	Matrix ans(i,j);
	for (int k=0;k<i;k++)
	{
		for (int l=0;l<j;l++)
		{
			ans.mElements[k][l] = A.mElements[k][l] * B.mElements[k][l];
		}
	}
	return ans;
}

// Outer product

Matrix outer(Vector u, Vector v)
{
	int l1 = length(u); int l2 =length(v);
	Matrix ans(l1, l2);
	for (int i=0;i<l1;i++)
	{
		for(int j=0;j<l2;j++)
		{
			ans.mElements[i][j] = u(i+1)*v(j+1);
		}
	}
	return ans;
}

// Kronecker Product

Matrix kron(const Matrix& A, const Matrix& B)
{
	int m = A.mRows; int n = A.mColumns;
	int p = B.mRows; int q = B.mColumns;
	int r = m*p; int c = n*q;
	Matrix M(r,c);
	Vector rowpos(m);
	Vector colpos(n);
	for (int i=0;i<m;i++)
	{
		rowpos(i+1) = i*p;
	}
	for (int i=0;i<n;i++)
	{
		colpos(i+1) = i*q;
	}
	for (int i=1;i<=m;i++)
	{
		for (int j = 1;j<=n;j++)
		{
			M.setElements(rowpos(i)+1, rowpos(i)+p, colpos(j)+1, colpos(j)+q, A(i,j)*B);
		}
	}
	return M;

}

// Return the size of the Matrix in a vector, [#rows, #cols]

Vector size(const Matrix& A)
{
	Vector dimensions(2);
	dimensions(1) = A.mRows;
	dimensions(2) = A.mColumns;
	return dimensions;
}

// Identity Matrix

Matrix eye(int i)
{
	Matrix I(i,i);
	for (int k=0;k<i;k++)
	{
		I.mElements[k][k] = 1;
	}
	return I;
}

// Diagonal Matrix
// Takes in a reference to a vector, and outputs a square matrix with that vector on the diagonal, zeros elsewhere;

Matrix diag(const Vector& v)
{
	Matrix D(length(v), length(v));
	for (int k=0;k<length(v);k++)
	{
		D.mElements[k][k] = v(k+1);
	}
	return D;
}

Matrix diag(const Vector& v, int i) // lets you specify which diagonal to put the vector on
{
	int dim = length(v)+abs(i);
	Matrix D(dim, dim);
	if (i==0) {return diag(v);}
	else if (i>0) {
		for (int k=0;k<length(v);k++)
		{
			D.mElements[k][k+i] = v(k+1);
		}
	}
	else if (i<0)	{
		for (int k=0;k<length(v);k++)
		{
			D.mElements[k-i][k] = v(k+1);
		}
	}
	//std::cout<<"This comes form inside 2\n"<<D<<"\n";
	return D;
}

// Return the vector on the leading diagonal of a matrix (matrix need not be square)

Vector diag(const Matrix& A)
{
	int r =  A.mRows; int c = A.mColumns;
	int l = std::max(r,c);
	Vector diagonal(l);
	for (int i=0;i<l;i++)
	{
		diagonal(i+1) = A.mElements[i][i]; //vectors are also indexed in this notation from 1 not 0, given the overloaded index operator in the Vector class
	}
	return diagonal;
}

// Linear Solvers

// when the functions output results to file, this was used for the results produced in the report. To be removed or made optional

// Back Substitution. Input is a upper diagonal matrix and the RHS

Vector backsubst(const Matrix& A, const Vector& b, int bwidth)
{
	int n = length(b);
	int count = 0;
	Vector x(n);
	if (bwidth==0)
	{
		for (int i=n;i>0;i--)
		{
			if (A(i,i)==0)
			{
				throw Exception("Singular Matrix",
						  "no solution to equation as coefficient matrix is singular");
			}
			x(i) = b(i);
			for (int j = (i+1); j<=n; j++)
			{
				x(i) = x(i) - A(i,j)*x(j);
				count++;
			}
			x(i) = x(i)/A(i,i);
		}
	}
	else
	{
		for (int i=n;i>0;i--)
		{
			if (A(i,i)==0)
			{
				throw Exception("Singular Matrix",
										  "no solution to equation as coefficient matrix is singular");
			}
			x(i) = b(i);
			for (int j = (i+1); j<=(i+bwidth - 1); j++)
				{
					if(j<=n)
					{
						x(i) = x(i) - A(i,j)*x(j);
						count++;
					}
				}
			x(i) = x(i)/A(i,i);
		}
	}
	std::cout<<"operation count: "<<count<<"\n";
	return x;
}

// Forward Substitution

Vector forwardsubst(const Matrix& A, const Vector& b)
{
	int n = length(b);
	Vector x(n);
	for (int i=1;i<=n;i++)
	{
		if (A(i,i)==0)
		{
			throw Exception("Singular Matrix",
									  "no solution to equation as coefficient matrix is singular");
		}
		x(i) = b(i);
		for (int j = 1; j<=(i-1); j++)
		{
			x(i) = x(i) - A(i,j)*x(j);
		}
		x(i) = x(i)/A(i,i);
	}
	return x;
}


// Conjugate gradient method for solving Ax = b, given vector b and Matrix A. NOTE: only works with symmetric positive definite matrices

Vector CG(const Matrix& A, const Vector& b)
{
	std::ofstream CGres;
	std::ofstream CGIterates;
	CGres.open("CGres.txt");
	CGIterates.open("CGIterates.txt");
	Vector x(length(b));
	CGIterates << std::setprecision(15) << x << "\n";
	Vector temp1 = A*x;
	Vector r = b - temp1;
	Vector p = r;
	double temp2 = r*r;
	double temp3, beta, alpha;
	do {
		temp1 = A*p;
		alpha = (temp2)/(p*temp1);
		x = x + alpha*p;
		CGIterates << std::setprecision(15) << x << "\n";
		temp3 = temp2;
		r = r - alpha*temp1;
		temp2 = r*r;
		beta = temp2/temp3;
		p = r + beta*p;
		CGres << std::setprecision(15) << r << "\n";
	}while(norm(r)>1e-15);
	CGres.close();
	CGIterates.close();
	return x;
}

// GMRES for solving Ax = b

Vector GMRES(const Matrix& A, const Vector& b)
{
	std::ofstream GMRESres;
	GMRESres.open("GMRESres.txt");
	int n = length(b);
	Vector x(n);
	Vector temp1 = A*x;
	Vector r = b - temp1;
	double norm_r = norm(r);
	GMRESres << std::setprecision(15) << norm_r << "\n";
	double norm_r0 = norm_r;
	Vector v = r/norm_r;
	Matrix V(length(v),n);
	Matrix H(n+1,n);
	Vector r_e1(n+1);
	r_e1(1) = norm_r0;
	Matrix G = eye(n+1);
	V.setElements(1,n,1,1,v);

	double Tol = 1e-14;
	int k=0;
	while (norm_r > Tol)
		{
			k = k+1;
			Vector v_k = V.getElements(1,n,k,k,"Vector");
			Vector w = A*v_k;
			for (int j = 1; j<=k; j++)
			{
				Vector v_j = V.getElements(1,n,j,j,"Vector");
				double h = v_j*w;
				H(j,k) = h;
				w = w - h*v_j;
			}
			double hplus = norm(w);
			H(k+1,k) = hplus;
			if (hplus<1e-13)
			{break;}
			else
			{
				Vector v_kplus1 = w/hplus;
				V.setElements(1,n,k+1,k+1,v_kplus1);
				Matrix Hhat = H.getElements(1,(k+1),1,k);
				Hhat = G.getElements(1,k+1,1,k+1)*Hhat;
				std::pair<double,double> g = givens(Hhat(k,k),Hhat(k+1,k));
				double gcos = g.first; double gsin = g.second;
				G.applygivens(k,k+1,gcos,gsin);
				norm_r = std::abs(norm_r*gsin);
				GMRESres << std::setprecision(15) << norm_r << "\n";
			}

		}

		r_e1.setValues(1,k+1,G.getElements(1,k+1,1,k+1)*r_e1.getValues(1,k+1));
		H.setElements(1,k+1,1,k, G.getElements(1,k+1,1,k+1)*H.getElements(1,(k+1),1,k));
		Vector y = backsubst(H.getElements(1,k,1,k),r_e1.getValues(1,k));
		Vector xfinal = temp1 + V.getElements(1,n,1,k)*y;
		GMRESres.close();
		return xfinal;
}

Vector MINRES(const Matrix& A, const Vector& b)
{
	std::ofstream MINRESres;
	MINRESres.open("MINRESres.txt");
	int n = length(b);
	Vector x(n);
	Vector temp1 = A*x;
	Vector r = b - temp1;
	double norm_r = norm(r);
	MINRESres << std::setprecision(15) << norm_r << "\n";
	double norm_r0 = norm_r;
	Vector v = r/norm_r;
	Matrix V(length(v),n);
	Matrix H(n+1,n);
	Vector r_e1(n+1);
	r_e1(1) = norm_r0;
	Matrix G = eye(n+1);
	V.setElements(1,n,1,1,v);
	double Tol = 1e-15;
	int k=0;
	double gamma = 0;
	while (norm_r > Tol)
		{
			k = k+1;
			Vector v_k = V.getElements(1,n,k,k,"Vector");
			Vector w = A*v_k;
			double delta = v_k*w;
			H(k,k) = delta;
			if (k>1)
			{
				H(k-1,k) = gamma;
				w = w - delta*v_k - gamma*V.getElements(1,n,k-1,k-1,"Vector");
			}
			else
			{
					w = w - delta*v_k;
			}
			gamma = norm(w);
			H(k+1,k) = gamma;
			if (gamma<1e-13)
			{ break;}
			else
			{
				Vector v_kplus1 = w/gamma;
				V.setElements(1,n,k+1,k+1,v_kplus1);
				Matrix Hhat = H.getElements(1,(k+1),1,k);
				Hhat = G.getElements(1,k+1,1,k+1)*Hhat;
				std::pair<double,double> g = givens(Hhat(k,k),Hhat(k+1,k));
				double gcos = g.first; double gsin = g.second;
				G.applygivens(k,k+1,gcos,gsin);
				norm_r = std::abs(norm_r*gsin);
				MINRESres << std::setprecision(15) << norm_r << "\n";
			}

		}

		r_e1.setValues(1,k+1,G.getElements(1,k+1,1,k+1)*r_e1.getValues(1,k+1));
		H.setElements(1,k+1,1,k, G.getElements(1,k+1,1,k+1)*H.getElements(1,(k+1),1,k));
		Vector y = backsubst(H.getElements(1,k,1,k),r_e1.getValues(1,k), 3);
		Vector xfinal = temp1 + V.getElements(1,n,1,k)*y;
		MINRESres.close();
		return xfinal;
}

// Jacobi's method

Vector jacobi(const Matrix& A, const Vector& b)
{
	std::ofstream JacobiIterates;
	JacobiIterates.open("JacobiIterates.txt");
	int n = length(b);
	Vector x(n);
	Vector xnext(n);
	Vector r = b - A*x;
	double Tol = 1e-14;
	int iter = 0;
	while (norm(r) > Tol & iter < 10000)
	{
		for (int i = 1; i<=n; i++)
		{
			double sum = 0;
			for (int j=1; j<=n; j++)
			{
				if (i != j) { sum = sum + A(i,j)*x(j);}
				else if (i==j) {}
			}
			xnext(i) = (b(i) - sum)/A(i,i);
			JacobiIterates << std::setprecision(15) << xnext(i)<< ",";
		}
		r = b - A*xnext;
		x = xnext;
		JacobiIterates << "\n";
		iter++;
	}

	JacobiIterates.close();
	return xnext;
}

// Gauss-Seidel method

Vector GS(const Matrix& A, const Vector& b)
{
	std::ofstream GSIterates;
	GSIterates.open("GSIterates.txt");
	int n = length(b);
	Vector x(n);
	Vector xnext(n);
	Vector r = b - A*x;
	double Tol = 1e-14;
	int iter = 0;
	while (norm(r) > Tol & iter < 10000)
	{
		for (int i = 1; i<=n; i++)
		{
			double sum1 = 0; double sum2 = 0;
			if (i!=1){
			for (int j=1; j<i; j++)
			{
				sum1 = sum1 + A(i,j)*xnext(j);
			}
			}
			if (i!=n){
			for (int j=i+1; j<=n; j++)
			{
				sum2 = sum2 + A(i,j)*x(j);
			}
			}
			xnext(i) = (b(i) - sum2 - sum1)/A(i,i);
			GSIterates << std::setprecision(15) << xnext(i)<< ",";
		}
		r = b - A*xnext;
		x = xnext;
		GSIterates << "\n";
		iter++;
	}
	GSIterates.close();
	return xnext;
}

// Successive Over Relaxation method

Vector SOR(const Matrix& A, const Vector& b, double w)
{
	std::ofstream SORIterates;
	SORIterates.open("SORIterates.txt");
	int n = length(b);
	Vector x(n);
	Vector xnext(n);
	Vector r = b - A*x;
	double Tol = 1e-14;
	int iter = 0;
	while (norm(r) > Tol & iter < 10000)
	{
		for (int i = 1; i<=n; i++)
		{
			double sum1 = 0; double sum2 = 0;
			if (i!=1){
			for (int j=1; j<i; j++)
			{
				sum1 = sum1 + A(i,j)*xnext(j);
			}
			}
			if (i!=n){
			for (int j=i+1; j<=n; j++)
			{
				sum2 = sum2 + A(i,j)*x(j);
			}
			}
			xnext(i) = (1-w)*x(i) + (b(i) - sum2 - sum1)*(w/A(i,i));
			SORIterates << std::setprecision(15) << xnext(i)<< ",";
		}
		r = b - A*xnext;
		x = xnext;
		SORIterates << "\n";
		iter++;
	}
	SORIterates.close();
	return xnext;
}

Matrix hess(const Matrix& A)
{

	int n = A.mRows;
	Matrix Ahes = A;
	disp(A);
	Vector u = Ahes.getElements(2,n,1,1,"Vector");
	std::cout<< "u=\n"<< u <<"\n";
	Vector v(length(u));
	v(1) = norm(u,2);
	std::cout<< "v=\n"<< v <<"\n";
	Vector w = u-v;
	Matrix H = eye(n);
	Matrix K = eye(n-1);
	if (norm(w)>1e-14)
	{
		Matrix temp = outer(w,w);
		K = K - (2/(w*w))*temp;
	}
	std::cout<< "H=\n"<< H <<"\n";
	H.setElements(2,n,2,n,K);
	disp(H);
	Ahes = H*Ahes*H;
	disp(Ahes);
	for (int i=2;i<n-1;i++)
	{
		Matrix B = Ahes.getElements(i,n,i, n);
		Vector u2 = B.getElements(2,B.mRows,1,1,"Vector");
		std::cout<< "B=\n"<< B <<"\n";
		std::cout<< "u2=\n"<< u2 <<"\n";
		Vector v2(length(u2));
		v2(1) = norm(u2,2);;
		std::cout<< "v2=\n"<< v2 <<"\n";
		Vector w2 = u2-v2;
		std::cout<<"w2=\n" << w2<<"\n";
		Matrix Hhat = eye(B.mRows-1);
		if (norm(w2)>1e-14)
		{
		Matrix temp2 = outer(w2,w2);
		std::cout<< temp2 <<"\n\n";
		Hhat = Hhat - (2/(w2*w2))*temp2;
		}
		std::cout<< "Hhat=\n"<< Hhat <<"\n\n";
		Matrix H2 = eye(n);
		H2.setElements(i+1,n,i+1,n,Hhat);
		std::cout<< "H2=\n"<< H2 <<"\n\n";
		Ahes = H2*Ahes*H2;
	}

	for (int i=1;i<=n;i++)
		{
			for(int j=1;j<=n;j++)
			{
				if (std::abs(Ahes(i,j)) < 1e-12)
					Ahes(i,j) = 0;
			}
		}
	return Ahes;
}


// LU Factorisation

boost::tuple< Matrix, Matrix, Matrix> LU(const Matrix& A)
	{
		Vector dim = size(A);
		int m = dim(1);
		int n = dim(2);
		Matrix U(A);
		Matrix pivot(m,m);
		Matrix P = eye(m);
		Matrix L = eye(m);

		for (int k=1;k<m;k++)
		{
			// partial pivoting
			double max = U(k,k);
			int index = k;
			pivot = eye(m);
			for (int i = k; i<=m; i++)
			{
				if (std::abs(U(i,k)) > std::abs(max))
				{
					index = i;
					max = U(i,k);
				}
			}
			if (index != k)
			{

				Vector temp = U.getElements(k,k,1,n,"vector");
				U.setElements(k,k,1,n,U.getElements(index,index,1,n,"Vector"));
				U.setElements(index,index,1,n,temp);
				Vector temp2 = L.getElements(k,k,1,m,"vector");

				L.setElements(k,k,1,m,L.getElements(index,index,1,m,"Vector"));
				L.setElements(index,index,1,m,temp2);
				pivot(index,index)=0; pivot(k,k) = 0;
				pivot(index,k) = 1; pivot(k,index) = 1;
				P = pivot*P;
			}
			// variant of Gaussian elimination
			Matrix Lk = eye(m);
			for (int i = k+1; i<=m;i++)
			{
				Lk(i,k) = U(i,k)/U(k,k);
				for (int j=k+1;j<=n;j++)
				{
					U(i,j) = U(i,j) - Lk(i,k)*U(k,j);
				}
				U(i,k) = 0;
			}
			L = L*pivot*Lk;
		}
		boost::tuple<Matrix,Matrix,Matrix> LUP = boost::make_tuple(L, U, P);
		return LUP;
	}


// QR Factorisation (using householder matrices, see report for algorithm

std::pair <Matrix,Matrix> QR(const Matrix& A)
{
	int Rrows = A.mRows;
	int Rcolumns = A.mColumns;
	int t = std::min(Rrows,Rcolumns);
	Matrix R = A;
	Matrix Q = eye(Rrows);
	Vector u = R.getElements(1,Rrows,1,1,"Vector");
	Vector v(length(u));
	v(1) = norm(u,2);
	Vector w = u-v;
	Matrix H = eye(Rrows);
	if (norm(w,2)>1e-15)
		{
			Matrix temp = outer(w,w);
			H = H - (2/(w*w))*temp;
		}
	R = H*R;
	Q = Q*H;
	for (int i=2;i<t;i++)
	{
		Matrix B = R.getElements(i,Rrows,i, Rcolumns);
		Vector u2 = B.getElements(1,B.mRows,1,1,"Vector");
		Vector v2(length(u2));
		v2(1) = norm(u2,2);;
		Vector w2 = u2-v2;
		Matrix I2 = eye(B.mRows);
		Matrix temp2 = outer(w2,w2);
		Matrix H2 = eye(Rrows);
		if (norm(w2,2)>1e-15)
		{
		Matrix Hhat = I2 - (2/(w2*w2))*temp2;
		H2.setElements(i,Rrows,i,Rrows,Hhat);
		}
		R = H2*R;
		Q = Q*H2;
	}
	for (int i=1;i<=Rrows;i++)
	{
		for(int j=1;j<=Rcolumns;j++)
		{
			if (std::abs(R(i,j)) < 1e-14)
				{R(i,j) = 0;}
		}
	}
	std::pair<Matrix,Matrix> qr = std::make_pair(Q,R);
	return qr;

}

// Find Eigenvalues
std::pair <Vector,Matrix> eig(const Matrix& A)
{
	Vector dim = size(A);
	if (dim(1) != dim(2)) {throw Exception("dimension error",
	"computing eig(), matrix supplied not square");}
	std::pair<Matrix, Matrix> qr = QR(A);
	Matrix Q = qr.first;
	Matrix R = qr.second;
	disp(Q);
	disp(R);
	Matrix Anew = R*Q;
	Matrix Atemp = Matrix(dim(1),dim(2));
	Matrix evecs = Q;
	do {
			Atemp = Anew;
			qr = QR(Anew);
			Q = qr.first;
			R = qr.second;
			Anew = R*Q;
			evecs = evecs*Q;
			disp(Q);
			disp(evecs);
		}while((std::abs(Atemp(1,1) - Anew(1,1))) > 1e-15);

	Vector evals = diag(Anew);
	std::pair<Vector,Matrix> eigs = std::make_pair(evals,evecs);
	return eigs;
}

// givens rotation function

std::pair <double, double> givens(double a, double b)
{
	double r = sqrt((a*a) + (b*b));
	double c = a/r; double s = -b/r;
	std::pair <double,double> giv = std::make_pair(c,s);
	return giv;
}

// apply givens rotaiton to a matrix, using input from givens function
void Matrix:: applygivens(int i, int j, double c, double s)
{
	for (int k = 1; k<=mColumns; k++)
	{
		double newi = c*mElements[i-1][k-1] - s*mElements[j-1][k-1];
		double newj = s*mElements[i-1][k-1] + c*mElements[j-1][k-1];
		mElements[i-1][k-1] = newi; mElements[j-1][k-1] = newj;
	}
}
