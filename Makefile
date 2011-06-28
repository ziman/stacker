CXX = g++
CXXFLAGS = -O2 -I/usr/include/opencv
LDFLAGS= -lcv -lcxcore -lhighgui

.SUFFIXES: .o .cpp

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $< -o $@

align: main.o
	$(CXX) $(LDFLAGS) main.o -o align

clean:
	-rm -f *.o *~ align
