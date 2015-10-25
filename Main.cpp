
// WARNING: this sample is under construction! Use it on your own risk.
#if defined _MSC_VER && _MSC_VER >= 1400
#pragma warning(disable : 4100)
#endif


#include <iostream>
#include <iomanip>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudaarithm.hpp"
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <cstdlib>

#include "opencv2/tick_meter.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;
using namespace cv::dnn;

Rect cropRect(0,0,0,0);

/////////////// KINECT ///////////////

#include <iostream>
#include <signal.h>

#include <opencv2/opencv.hpp>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/threading.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>

bool protonect_shutdown = false;

void sigint_handler(int s)
{
  protonect_shutdown = true;
}

/////////////// END KINECT ///////////////

/////////////// HAAR ///////////////

static void help()
{
    cout << "Usage: ./cascadeclassifier_gpu \n\t--cascade <cascade_file>\n\t(<image>|--video <video>|--camera <camera_id>)\n"
            "Using OpenCV version " << CV_VERSION << endl << endl;
}

static void convertAndResizeCPU(const Mat& src, Mat& resized, double scale)
{
    Size sz(cvRound(src.cols * scale), cvRound(src.rows * scale));

    if (scale != 1)
    {
        cv::resize(src, resized, sz);
    }
    else
    {
        resized = src;
    }
}

static void convertAndResizeGPU(const GpuMat& src, GpuMat& gray, GpuMat& resized, double scale)
{
    if (src.channels() == 3)
    {
        cv::cuda::cvtColor( src, gray, COLOR_BGR2GRAY );
    }
    else
    {
        gray = src;
    }

    Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));

    if (scale != 1)
    {
        cv::cuda::resize(gray, resized, sz);
    }
    else
    {
        resized = gray;
    }
}


static void matPrint(Mat &img, int lineOffsY, Scalar fontColor, const string &ss)
{
    int fontFace = FONT_HERSHEY_DUPLEX;
    double fontScale = 0.8;
    int fontThickness = 2;
    Size fontSize = cv::getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

    Point org;
    org.x = 1;
    org.y = 3 * fontSize.height * (lineOffsY + 1) / 2;
    putText(img, ss, org, fontFace, fontScale, Scalar(0,0,0), 5*fontThickness/2, 16);
    putText(img, ss, org, fontFace, fontScale, fontColor, fontThickness, 16);
}


static void displayState(Mat &canvas, bool bHelp, bool bGpu, bool bLargestFace, bool bFilter, double fps, double scaleFactor)
{
    Scalar fontColorRed = Scalar(255,0,0);
    Scalar fontColorNV  = Scalar(118,185,0);

    ostringstream ss;
    ss << "FPS = " << setprecision(1) << fixed << fps;
    matPrint(canvas, 0, fontColorRed, ss.str());
    ss.str("");
    ss << "[" << canvas.cols << "x" << canvas.rows << "], " <<
        (bGpu ? "GPU, " : "CPU, ") <<
        (bLargestFace ? "OneFace, " : "MultiFace, ") <<
        (bFilter ? "Filter:ON" : "Filter:OFF")<<
        " Scale: " << scaleFactor;
    matPrint(canvas, 1, fontColorRed, ss.str());

    if (bHelp)
    {
        matPrint(canvas, 2, fontColorNV, "Space - switch GPU / CPU");
        matPrint(canvas, 3, fontColorNV, "M - switch OneFace / MultiFace");
        matPrint(canvas, 4, fontColorNV, "F - toggle rectangles Filter");
        matPrint(canvas, 5, fontColorNV, "H - toggle hotkeys help");
        matPrint(canvas, 6, fontColorNV, "1/Q - increase/decrease scale");
    }
    else
    {
        matPrint(canvas, 2, fontColorNV, "H - toggle hotkeys help");
    }
}

/////////////// END HAAR ///////////////

/////////////// CAFFE ////////////////
/////////////// ONLY CAFFE OR DNN ////////////
/////////////// OTHERWISE PROTOBUF ERROR ////////////
/////////////// GLOG_minloglevel=3 ./OpenCVKinect gl //////////////////////

#include "caffeclassifier.h"

using boost::shared_ptr;
using namespace caffe;

/////////////// END CAFFE ////////////////

int main(int argc, const char *argv[])
{
    if (getCudaEnabledDeviceCount() == 0)
    {
        return cerr << "No GPU found or the library is compiled without CUDA support" << endl, -1;
    }

    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    ////////////// CAFFE /////////////////////

//    Caffe::set_mode(Caffe::GPU);
//    Caffe::SetDevice(0);
//
//	caffe::Datum* datum = new caffe::Datum();
//	CVMatToDatum(cropImg, datum);
//
//    // Load net
//	// Assume you are in Caffe master directory
//	caffe::Net<float> net("/home/k1y0sh1/DeveloperZone/Project/EclipseWorkplace/OpenCV249U1404/Debug/bvlc_googlenet.prototxt", TEST);
//
//	// Load pre-trained net (binary proto)
//	// Assume you are already trained the cifar10 example.
//	net.CopyTrainedLayersFrom("/home/k1y0sh1/DeveloperZone/Project/EclipseWorkplace/OpenCV249U1404/Debug/bvlc_googlenet.caffemodel");
//
//	caffe::Blob<float>* input_blob = new caffe::Blob<float>(1, datum->channels(), datum->height(), datum->width());
//	 //get the blobproto
//	caffe::BlobProto blob_proto;
//	blob_proto.set_num(1);
//	blob_proto.set_channels(datum->channels());
//	blob_proto.set_height(datum->height());
//	blob_proto.set_width(datum->width());
//
//	const string& data = datum->data();
//	for (uint32_t i = 0; i < data.length(); ++i) {
//		blob_proto.add_data((uint8_t)data[i]);
//	}
//
//	//set data into blob
//	input_blob->FromProto(blob_proto);
//
//	std::vector<caffe::Blob<float>*> input_cnn;
//	input_cnn.push_back(input_blob);
//
//	float loss;
//	std::vector<caffe::Blob<float>*> input_blobs = net.input_blobs();
//	for (int i = 0; i < input_cnn.size(); ++i) {
//		input_blobs[i]->CopyFrom(*input_cnn[i]);
//	}
//	const std::vector<caffe::Blob<float>*>& result = net.ForwardPrefilled(&loss);
//
//	std::cout << "loss: " << loss << "\n";
//	// read the 'prob' layer and get the result
//
//	shared_ptr<caffe::Blob<float> > prob = net.blob_by_name("prob");
//
//	float maxval= 0;
//	int   maxinx= 0;
//	for (int i = 0; i < prob->count(); i++)
//	{
//		float val = (prob->cpu_data()[i]) * 100;
//		if (val> maxval)
//		{
//			maxval = val;
//			maxinx = i;
//		}
//		std::cout << "[" << i << "]" << val<< "\n";
//	}
//	std::cout << "Max value = " << maxval << ", Max index = " << maxinx<< "\n";

//    Mat cropImg = cv::imread("/home/k1y0sh1/DeveloperZone/Project/EclipseWorkplace/CaffePredictTest/Debug/80.jpg",CV_LOAD_IMAGE_COLOR);
//	imshow("crop",cropImg);
//	cv::resize(cropImg, cropImg, cv::Size(224, 224));

    string model_file   = "/home/k1y0sh1/DeveloperZone/Project/EclipseWorkplace/CaffePredictTest/Debug/deploy.prototxt";
    string trained_file = "/home/k1y0sh1/DeveloperZone/Project/EclipseWorkplace/CaffePredictTest/Debug/snapshot_iter_14640.caffemodel";
    string mean_file    = "/home/k1y0sh1/DeveloperZone/Project/EclipseWorkplace/CaffePredictTest/Debug/mean.binaryproto";
    string label_file   = "/home/k1y0sh1/DeveloperZone/Project/EclipseWorkplace/CaffePredictTest/Debug/labels.txt";

    CaffeClassifier CaffeClassifier(model_file, trained_file, mean_file, label_file);
//
//	string file = "/home/k1y0sh1/DeveloperZone/Project/EclipseWorkplace/CaffePredictTest/Debug/80.jpg";
//
//	std::cout << "---------- Prediction for "
//			<< file << " ----------" << std::endl;
//
//	std::vector<Prediction> predictions = CaffeClassifier.Classify(cropImg,1);
//
//	/* Print the top N predictions. */
//	for (size_t i = 0; i < predictions.size(); ++i)
//	{
//		Prediction p = predictions[i];
//		std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
//	}


    ////////////// END CAFFE /////////////////////

    ////////////// HAAR /////////////////////

    string cascadeName = "/home/k1y0sh1/DeveloperZone/HaarTraining/classifiers/cascade.xml";

    Ptr<cuda::CascadeClassifier> cascade_gpu = cuda::CascadeClassifier::create(cascadeName);

    Mat image;

    namedWindow("result", 1);

    Mat frame, frame_cpu, gray_cpu, resized_cpu, frameDisp;
    vector<Rect> faces;

    GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu, k_rgb_gpu;

    /* parameters */
    bool useGPU = true;
    double scaleFactor = 0.5;
    bool findLargestObject = true;
    bool filterRects = true;
    bool helpScreen = false;
    bool predictObject = false;

    ////////////// END HAAR /////////////////////

    //////////////////// KINECT /////////////////////

    std::string program_path(argv[0]);
    size_t executable_name_idx = program_path.rfind("OpenCVKinectGPU");

    std::string binpath = "/";

    if(executable_name_idx != std::string::npos)
    {
        binpath = program_path.substr(0, executable_name_idx);
    }

    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = 0;
    libfreenect2::PacketPipeline *pipeline = 0;

    if(freenect2.enumerateDevices() == 0)
    {
        std::cout << "no device connected!" << std::endl;
        return -1;
    }

    std::string serial = freenect2.getDefaultDeviceSerialNumber();

    for(int argI = 1; argI < argc; ++argI)
    {
    const std::string arg(argv[argI]);

    if(arg == "cpu")
    {
      if(!pipeline)
        pipeline = new libfreenect2::CpuPacketPipeline();
    }
    else if(arg == "gl")
    {
    #ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::OpenGLPacketPipeline();
    #else
      std::cout << "OpenGL pipeline is not supported!" << std::endl;
    #endif
    }
    else if(arg == "cl")
    {
    #ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
      if(!pipeline)
        pipeline = new libfreenect2::OpenCLPacketPipeline();
    #else
      std::cout << "OpenCL pipeline is not supported!" << std::endl;
    #endif
    }
    else if(arg.find_first_not_of("0123456789") == std::string::npos) //check if parameter could be a serial number
    {
      serial = arg;
    }
    else
    {
      std::cout << "Unknown argument: " << arg << std::endl;
    }
    }

    if(pipeline)
    {
    dev = freenect2.openDevice(serial, pipeline);
    }
    else
    {
    dev = freenect2.openDevice(serial);
    }

    if(dev == 0)
    {
    std::cout << "failure opening device!" << std::endl;
    return -1;
    }

    signal(SIGINT,sigint_handler);
    protonect_shutdown = false;

    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color);
    libfreenect2::FrameMap frames;

    dev->setColorFrameListener(&listener);
    dev->start();

    std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
    std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;

    libfreenect2::Registration* registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());

    /////////////////// END KINECT /////////////////

    while(!protonect_shutdown)
    {
        listener.waitForNewFrame(frames);
        libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];

        cv::Mat k_rgb = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);

        image = Mat(k_rgb);
        frame_gpu.upload(k_rgb);

        cuda::flip(frame_gpu,frame_gpu,1);
        cv::flip(image,image,1);
        cuda::cvtColor(frame_gpu,k_rgb_gpu,CV_BGRA2BGR);
        convertAndResizeGPU(k_rgb_gpu, gray_gpu, resized_gpu, scaleFactor);
        convertAndResizeCPU(image,image,scaleFactor);

        TickMeter tm;
        tm.start();

        //cascade_gpu->setMaxNumObjects(2);
        //cascade_gpu->setMaxObjectSize(cv::Size(224,224));
        //cascade_gpu->setMinObjectSize(cv::Size(0,0));
        cascade_gpu->setFindLargestObject(findLargestObject);
        cascade_gpu->setScaleFactor(1.2);
        cascade_gpu->setMinNeighbors((filterRects || findLargestObject) ? 4 : 0);
        cascade_gpu->detectMultiScale(resized_gpu, facesBuf_gpu);
        cascade_gpu->convert(facesBuf_gpu, faces);

        for (size_t i = 0; i < faces.size(); ++i)
        {
            //cout<< "object [" << i << "]: " << faces[i].width << " x " << faces[i].height <<endl;
            rectangle(image, faces[i], Scalar(255));
            cropRect = Rect(image.cols / 2, image.rows / 2,224,224);
            Mat cropImg = image(cropRect).clone();

            if(predictObject == true)
            {
                std::vector<Prediction> predictions = CaffeClassifier.Classify(cropImg,1);

                /* Print the top N predictions. */
                for (size_t i = 0; i < predictions.size(); ++i)
                {
                    Prediction p = predictions[i];
                    std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
                }

                predictObject = false;
            }
        }


        tm.stop();
        double detectionTime = tm.getTimeMilli();
        double fps = 1000 / detectionTime;

        displayState(image, helpScreen, useGPU, findLargestObject, filterRects, fps,scaleFactor);
        imshow("result", image);

        char key = (char)waitKey(5);
        if (key == 27)
        {
            break;
        }

        switch (key)
        {
        case ' ':
            useGPU = !useGPU;
            break;
        case 'm':
        case 'M':
            findLargestObject = !findLargestObject;
            break;
        case 'f':
        case 'F':
            filterRects = !filterRects;
            break;
        case '1':
            scaleFactor *= 1.05;
            break;
        case 'q':
        case 'Q':
            scaleFactor /= 1.05;
            break;
        case 'h':
        case 'H':
            helpScreen = !helpScreen;
            break;
        case 'p':
        case 'P':
            predictObject = !predictObject;
            break;
        }
        protonect_shutdown = protonect_shutdown || (key > 0 && ((key & 0xFF) == 27)); // shutdown on escape

        listener.release(frames);
        //libfreenect2::this_thread::sleep_for(libfreenect2::chrono::milliseconds(100));
     }

    resized_gpu.release();

    // TODO: restarting ir stream doesn't work!
    // TODO: bad things will happen, if frame listeners are freed before dev->stop() :(
    dev->stop();
    dev->close();

    delete registration;

    return 0;
}
