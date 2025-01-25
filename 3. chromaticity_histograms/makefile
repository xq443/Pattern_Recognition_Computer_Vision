# MacOS compiler
CC = /usr/bin/g++

# Make it the C++ compiler too
CXX = $(CC)

# OpenCV include paths for Homebrew
CFLAGS = -std=c++11 -I/usr/local/Cellar/opencv/4.11.0/include/opencv4 -I../include -I/usr/local/include/onnxruntime

# Make these the C++ flags too
CXXFLAGS = $(CFLAGS)

# Library paths, update to where your OpenCV libraries are stored
LDFLAGS = -L/usr/local/Cellar/opencv/4.11.0/lib -L/usr/local/include/onnxruntime/

# opencv libraries
# these settings work for macOS and macports
LDLIBS = -ltiff -lpng -ljpeg -llapack -lblas -lz -ljasper -lwebp -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect

# Location to write the executables
BINDIR = .

hist: makeHist.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

historg: buildHistogram.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f *.o *~ 
