////////////////////////////////////////////////////////
///////// MATRIX CLASS FOR LINEAR ALGEBRA .hpp /////////
////////////////////////////////////////////////////////

// Ilan Price
// June 2017
// Produced for a 'special topic' report as part of MSc in Mathematical Modelling and Scientific Computing
// at the University of Oxford. Accompanies a written report.

#ifndef MATRIXDEF__
#define MATRIXDEF__

#include <cmath>
#include <string>
#include "boost/tuple/tuple.hpp"
#include <initializer_list>

class Vector; // Forward Declaration

class Matrix
{
private:
	double **mElements;
	int mRows;
	int mColumns;

public:
	//Constructors
	//No default constructor
	Matrix(int newRows, int newCols);

	Matrix(int newRows, int newCols, std::initializer_list<double> entries);

	// Over ride copy constructor
	Matrix(const Matrix& original);

	// Destructor
	~Matrix();

	// friend declarations of operators and functions

	// Helper Functions
	friend bool EqualDim(const Matrix& A, const Matrix& B);
	friend bool CanMultiply(const Matrix& A, const Matrix& B);

	// Binary operators
	friend Matrix operator+(const Matrix& A, const Matrix& B);
	friend Matrix operator-(const Matrix& A, const Matrix& B);
	friend Matrix operator*(const Matrix& A, const Matrix& B);
	friend Matrix operator*(double c, const Matrix& A);
	friend Matrix operator*(const Matrix& A, double c);
	friend Vector operator*(const Matrix& A, const Vector& v);
	friend Vector operator*(const Vector& v, const Matrix& A);
	friend Vector operator/(const Matrix& A, const Vector& b);

	//Transpose
	friend Matrix operator!(const Matrix& A);

	// Unary '-' operator
	friend Matrix operator-(const Matrix& A);

	//assignment
	Matrix& operator=(const Matrix& A);

	//indexing
	double& operator()(int i, int j);
	const double& operator()(int i, int j) const;

	// output
	friend std::ostream& operator<<(std::ostream& output, const Matrix& A);
	friend void disp(const Matrix& A);

	// Other functions
	friend Matrix ewisemult(const Matrix& A, const Matrix& B); // Element wise multiplication
	friend Matrix outer(Vector u, Vector v); // Outer product
	friend Matrix kron(const Matrix& A, const Matrix& B); // Kronecker product
	friend Vector size(const Matrix& A);
	friend Matrix eye(int i); // Identity matrix
	friend Matrix diag(const Vector& v);
	friend Matrix diag(const Vector& v, int i);
	friend Vector diag(const Matrix& A);

	// Hessenberg
	friend Matrix hess(const Matrix& A);
	// LU Factorisation
	friend boost::tuple< Matrix, Matrix, Matrix> LU(const Matrix& A);
	// QR Factorisation
	friend std::pair <Matrix,Matrix> QR(const Matrix& A);
	// Find Eigenvalues and Vectors with QR algorithm
	friend std::pair <Vector,Matrix> eig(const Matrix& A);
	// givens rotation
	friend std::pair <double, double> givens(double a, double b);
	// Back substitution
	friend Vector backsubst(const Matrix& A, const Vector& b, int bwidth);
	//Forward Subsitution
	friend Vector forwardsubst(const Matrix& A, const Vector& b);
	// Conjugate Gradient
	friend Vector CG(const Matrix& A, const Vector& b);
	//GMRES
	friend Vector GMRES(const Matrix& A, const Vector& b);
	//MINRES
	friend Vector MINRES(const Matrix& A, const Vector& b);
	//Jacobi
	friend Vector jacobi(const Matrix& A, const Vector& b);
	//Gauss-Seidel
	friend Vector GS(const Matrix& A, const Vector& b);
	// SOR
	friend Vector SOR(const Matrix& A, const Vector& b, double w);


	// Getters and setters
	// double getElement(int i, int j) const; // index from 1

	//slicing
	Matrix getElements(int istart, int iend, int jstart, int jend);
	Vector getElements(int istart, int iend, int jstart, int jend, std::string vector);

	void setElements(int istart, int iend, int jstart, int jend, Matrix A);
	void setElements(int istart, int iend, int jstart, int jend, Vector v);

	// apply givens rotation
	void  applygivens(int i, int j, double c, double s);
};

// Prototyping the friend functions and operators

bool EqualDim(const Matrix& A, const Matrix& B);
bool CanMultiply(const Matrix& A, const Matrix& B);
Matrix operator+(const Matrix& A, const Matrix& B);
Matrix operator-(const Matrix& A, const Matrix& B);
Matrix operator*(const Matrix& A, const Matrix& B);
Matrix operator*(double c, const Matrix& A);
Matrix operator*(const Matrix& A, double c);
Vector operator*(const Matrix& A, const Vector& v);
Vector operator*(const Vector& v, const Matrix& A);
Vector operator/(const Matrix& A, const Vector& b);


Matrix operator!(const Matrix& A);

Matrix operator-(const Matrix& A);
std::ostream& operator<<(std::ostream& output, const Matrix& A);
void disp(const Matrix& A);
Matrix ewisemult(const Matrix& A, const Matrix& B);
Matrix outer(Vector u, Vector v);
Matrix kron(const Matrix& A, const Matrix& B);
Vector size(const Matrix& A);
Matrix eye(int i);
Matrix diag(const Vector& v);
Matrix diag(const Vector& v, int i);
Vector diag(const Matrix& A);

Vector backsubst(const Matrix& A, const Vector& b, int bwidth = 0);
Vector forwardsubst(const Matrix& A, const Vector& b);
Vector CG(const Matrix& A, const Vector& b);
Vector GMRES(const Matrix& A, const Vector& b);
Vector MINRES(const Matrix& A, const Vector& b);
Vector jacobi(const Matrix& A, const Vector& b);
Vector GS(const Matrix& A, const Vector& b);
Vector SOR(const Matrix& A, const Vector& b, double w);
Matrix hess(const Matrix& A);
std::pair <double, double> givens(double a, double b);
boost::tuple< Matrix, Matrix, Matrix> LU(const Matrix& A);
std::pair <Matrix,Matrix> QR(const Matrix& A);
std::pair <Vector,Matrix> eig(const Matrix& A);

#endif
