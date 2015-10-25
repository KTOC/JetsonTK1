TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += "/usr/local/lib"
LIBS += -L"/usr/local/lib"  -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video \
                            -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_flann \
                            -lopencv_imgcodecs -lopencv_cudaimgproc -lopencv_cudaoptflow -lopencv_cudalegacy \
                            -lopencv_cudawarping -lopencv_cudafeatures2d -lopencv_cudaobjdetect -lopencv_cudacodec \
                            -lopencv_cudaarithm -lopencv_videoio

INCLUDEPATH += "/opt/libjpeg-turbo/lib64"
LIBS += -L"/opt/libjpeg-turbo/lib64" -lturbojpeg

INCLUDEPATH += "/usr/local/lib"
LIBS += -L"/usr/local/lib -ljpeg" -lfreenect2 -lglog -lprotobuf

INCLUDEPATH += "/usr/local/cuda/lib64"
LIBS += -L"/usr/local/cuda/lib64" -lcudart

INCLUDEPATH += "/home/k1y0sh1/DeveloperZone/caffe/distribute/lib"
LIBS += -L"/home/k1y0sh1/DeveloperZone/caffe/distribute/lib" -lcaffe

INCLUDEPATH += "/home/k1y0sh1/DeveloperZone/Libs/hdf5-1.9.202/hdf5/lib"
LIBS += -L"/home/k1y0sh1/DeveloperZone/Libs/hdf5-1.9.202/hdf5/lib" -lhdf5 -lhdf5_hl

INCLUDEPATH += /usr/local/include
INCLUDEPATH += /usr/local/cuda/include
INCLUDEPATH += /home/k1y0sh1/DeveloperZone/caffe/distribute/include
INCLUDEPATH += /home/k1y0sh1/DeveloperZone/Libs/TinyThread++-1.1/source


SOURCES += Main.cpp \
    caffeclassifier.cpp

HEADERS += \
    caffeclassifier.h



