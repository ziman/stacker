CXX = g++
CXXFLAGS = -O0 -g3 -I/usr/local/include/opencv2 -I/usr/local/include
LDFLAGS= -L/usr/local/lib \
	 -lopencv_core -lopencv_features2d \
	 -lopencv_highgui -lopencv_imgproc \
	 -lopencv_flann

.SUFFIXES: .o .cpp

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $< -o $@

align: main.o
	$(CXX) $(LDFLAGS) main.o -o align

clean:
	-rm -f *.o *~ align
