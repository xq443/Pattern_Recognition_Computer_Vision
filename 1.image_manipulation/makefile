# MacOS compiler
CC = /usr/bin/g++

# Make it the C++ compiler too
CXX = $(CC)

# OpenCV include paths for Homebrew
CFLAGS = -std=c++11 -I/usr/local/Cellar/opencv/4.10.0_18/include/opencv4 -I../include

# Make these the C++ flags too
CXXFLAGS = $(CFLAGS)

# Library paths, update to where your OpenCV libraries are stored
LDFLAGS = -L/usr/local/Cellar/opencv/4.10.0_18/lib

# OpenCV libraries and MacOS frameworks (adjust for Homebrew)
LDLIBS = -ltiff -lpng -ljpeg -llapack -lblas -lz -ljasper -lwebp -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect

# Location to write the executables
BINDIR = .

# Generic rule to compile .cpp files to .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rules for making various executables
img: imgMod.o
	$(CXX) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

tut1: tut1.o
	$(CXX) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

tut2: tut2.o gauss.o
	$(CXX) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f *.o *~
