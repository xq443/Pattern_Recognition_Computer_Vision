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

# OpenCV libraries and MacOS frameworks (adjust for Homebrew)
LDLIBS = -ltiff -lpng -ljpeg -llapack -lblas -lz -ljasper -lwebp -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lonnxruntime -lopencv_xphoto

# Location to write the executables
BINDIR = .

# Generic rule to compile .cpp files to .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rules for making various executables
main: utils.o featureVector.o readfiles.o csv_util.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

feature: featureVector.o
	$(CXX) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

dis: distanceMetric.o featureVector.o
	$(CXX) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

topN: topNGeneration.o featureVector.o distanceMetric.o csv_util.o 
	$(CXX) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

test: featureVector.o
	$(CXX) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f *.o *~