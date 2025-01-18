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
LDLIBS = -ltiff -lpng -ljpeg -llapack -lblas -lz -ljasper -lwebp -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lonnxruntime

# Location to write the executables
BINDIR = .

# Generic rule to compile .cpp files to .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rules for making various executables
vid: vidDisplay.o filter.o timeBlur.o faceDetect.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

# filter: filter.o
# 	$(CXX) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

imgDisplay: imgDisplay.o
	$(CXX) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

# vidDisplay: vidDisplay.o
# 	$(CXX) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

# timeBlur: timeBlur.o
# 	$(CXX) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

da2: da2-example.o filter.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
	install_name_tool -add_rpath /opt/local/lib ./$@

clean:
	rm -f *.o *~
