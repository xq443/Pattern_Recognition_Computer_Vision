# Project 2: Content-based Image Retrieval

- Name: Xujia Qin
- Links/URLs to any videos: no video links for this project
- OS: macos
- IDE: Visual Studio Code
- Instructions for running your executables: 
  - readfile: 
    - make main
    - ./main <image_dir> <featureset>, e.g.: ./main olympus square
  - topN: 
    - make topN
    - ./topN <featureset> <imagepath> <N>, e.g.: ./topN square olympus/pic.0018.jpg 4
- Instructions for testing any extensions you completed.
  - extension 1:
    - make topN
    - ./topN laplacianHist olympus/pic.0041.jpg 4
  - extension 2:
    - make dnn
    - ./dnn resnet18-v2-7.onnx olympus/pic.0018.jpg
    - make topN
    - ./topN OpenCVDNN olympus/pic.0018.jpg 4
  - extension 3:
    - make topN
    - ./topN getbanana olympus/pic.0343.jpg 4
- No travel days requested for this assignment