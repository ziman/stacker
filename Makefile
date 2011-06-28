CXX = g++
CXXFLAGS = -O2
LDFLAGS= -lcv

.SUFFIXES: .o .cpp

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $< -o $@

align: main.o
	$(CXX) $(LDFLAGS) main.o -o align

clean:
	-rm -f *.o *~ align
