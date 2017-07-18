all: Convergence
# all: use_vectors

#For debugging
OPT=-g -Wall --std=c++11 -O
#For optimistaion
#OPT=-O

#All objects (except main) come from cpp and hpp
%.o:	%.cpp %.hpp
	g++ ${OPT} -c -o $@ $<
#use_vectors relies on objects which rely on headers
Convergence:	Convergence.cpp Vector.o Exception.o Matrix.o
		g++ ${OPT} -o Convergence Convergence.cpp Vector.o Exception.o Matrix.o

# use_vectors:	use_vectors.cpp Vector.o Exception.o Matrix.o
# 		g++ ${OPT} -o use_vectors use_vectors.cpp Vector.o Exception.o Matrix.o

# clean:
# 		rm -f *.o *~ use_vectors
clean:
	rm -f *.o *~ Convergence
