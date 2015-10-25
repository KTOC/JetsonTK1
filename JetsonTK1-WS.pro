TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    Main.cpp \
    caffeclassifier.cpp

include(deployment.pri)
qtcAddDeployment()

HEADERS += \
    caffeclassifier.h



