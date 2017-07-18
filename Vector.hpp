#ifndef VECTORDEF
#define VECTORDEF

// Dr Joe Pitt-Francis, edited by Ilan Price June 2017
// Accompanies a Matrix class and report produced for a 'special topic' report as part of MSc in Mathematical Modelling and Scientific Computing
// at the University of Oxford.

//  **********************
//  *  Class of vectors  *
//  **********************


//  Class written in such a way that code similar to Matlab
//  code may be written


#include <cmath>
#include "Exception.hpp"//This class throws errors using the class "error"

class Matrix;

class Vector
{
private:
   // member variables
   double* mData;   // data stored in vector
   int mSize;      // size of vector

public:
   // constructors
   // No default constructor
   // overridden copy constructor
   Vector(const Vector& v1);
   // construct vector of given size
   Vector(int sizeVal);

   // destructor
   ~Vector();


   // All "friend" external operators and functions are declared as friend inside the class (here)
   // but their actual prototype definitions occur outside the class.
   // Binary operators
   friend Vector operator+(const Vector& v1, const Vector& v2);
   friend Vector operator-(const Vector& v1, const Vector& v2);
   friend double operator*(const Vector& v1, const Vector& v2);
   friend Vector operator*(const Vector& v, const double& a);
   friend Vector operator*(const double& a, const Vector& v);
   friend Vector operator/(const Vector& v, const double& a);
   // Unary operator
   friend Vector operator-(const Vector& v);

   //other operators
   //assignment
   Vector& operator=(const Vector& v);
   //indexing
   double& operator()(int i);
   const double& operator()(int i) const;
   //output
   friend std::ostream& operator<<(std::ostream& output, const Vector& v);
   friend void disp(const Vector& u);

   //norm (as a member method)
   double norm(int p=2) const;

   // other functions
   friend Vector linspace(double start, double end, int points);
   friend Matrix reshape(Vector& u, int rows, int cols);
   // functions that are friends
   friend double norm(Vector& v, int p);
   friend int length(const Vector& v);
   friend Vector ones(int length);

   //getters and setters
  //  double getValue(int i) const;
   Vector getValues(int i,int j) const;
   void setValues(int istart, int iend, Vector v) const;
   void setValue(int loc, int set) const;
};


// All "friend" external operators and functions are declared as friend inside the class
// but their actual prototype definitions occur outside the class (here).
// Binary operators
Vector operator+(const Vector& v1, const Vector& v2);
Vector operator-(const Vector& v1, const Vector& v2);
double operator*(const Vector& v1, const Vector& v2);
Vector operator*(const Vector& v, const double& a);
Vector operator*(const double& a, const Vector& v);
Vector operator/(const Vector& v, const double& a);
// Unary operator
Vector operator-(const Vector& v);

// function prototypes
void disp(const Vector& u);
double norm(Vector& v, int p=2);
Vector linspace(int start, int end, int points);
Matrix reshape(Vector& u, int rows, int cols);
Vector ones(int length);
// Prototype signature of length() friend function
int length(const Vector& v);

#endif
