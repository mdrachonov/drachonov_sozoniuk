#include <iostream>
#include <string>
#include "opencv2/highgui.hpp"
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/videoio.hpp"
#include "opencv2/opencv_modules.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iomanip>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include "opencv2/calib3d/calib3d.hpp"
#include <vector>

using namespace std;
using namespace cv;

void create_positions(Mat & img, vector< Rect > & locations, const Scalar & color);

#define FILM "/home/misha/Downloads/Car_lane_sign_detection-master/video.mp4"
#define CAR_SHAPE "/home/misha/Downloads/Car_lane_sign_detection-master/cars3.xml"
#define TAB_NAME "Cars detection tab"

int main()
{
    vector<Rect> vehicle_detected;
    CascadeClassifier cars;
    Mat image, grayColor, imgRoi, grayColor1, grayColor2;
	VideoCapture videoCapture;

	cars.load(CAR_SHAPE);
    
	videoCapture.open(FILM);

	while (videoCapture.read(image))
	{
        imgRoi = image(Rect(0, image.rows / 2, image.cols, image.rows / 2));

		cvtColor(imgRoi, grayColor, COLOR_BGR2GRAY);
        cvtColor(image, grayColor2, COLOR_BGR2GRAY);

        grayColor.copyTo(grayColor1);

		cars.detectMultiScale(grayColor, vehicle_detected, 1.1, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		create_positions(image, vehicle_detected, Scalar(128, 128, 128));

		imshow(TAB_NAME, image);
        
		waitKey(10);
	}

	return 0;
}

void create_positions(Mat & img, vector< Rect > &positions, const Scalar & color) {
    Mat image, car, carCover;
    img.copyTo(image);
    string length;

	if (!positions.empty()) {
        double range = 0;
        
        for (int a = 0; a < positions.size() ; ++a) {
            positions[a].y = positions[a].y + img.rows / 2;

            range = (0.0397 * 2) / ((positions[a].width) * 0.00007);

            Size size(positions[a].width / 1.5, positions[a].height / 3);
            
            Mat roi = img.rowRange(positions[a].y - size.height, (positions[a].y + positions[a].height / 3) - size.height)
                            .colRange(positions[a].x, (positions[a].x  + positions[a].width / 1.5));

            stringstream stream;
            stream << fixed << setprecision(2) << range;
            length = stream.str() + "m";
            rectangle(img, positions[a], color, -1);
        }

        addWeighted(image, 0.8, img, 0.2, 0, img);
        
        for (int a = 0; a < positions.size(); ++a) {
            rectangle(img, positions[a], color, 2);
            
            if (std::stoi(length) < 10) {	
                putText(img, length, Point(positions[a].x, positions[a].y + positions[a].height - 5), FONT_HERSHEY_DUPLEX, 0.6, Scalar(0, 0, 255), 1);
            } else {
                putText(img, length, Point(positions[a].x, positions[a].y + positions[a].height - 5), FONT_HERSHEY_DUPLEX, 0.6, Scalar(50, 205, 50), 1);
            }

            positions[a].y = positions[a].y - img.rows / 2;
        }
	}
}
