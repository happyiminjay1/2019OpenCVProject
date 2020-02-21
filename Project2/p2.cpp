#include "cv.hpp"
#include <iostream>
#include <opencv2/dnn.hpp>
#include <fstream>

using namespace cv;
using namespace std;
using namespace dnn;

void drawLine(vector<Vec2f> lines, float x, float y, float angle_th1, float angle_th2, Mat &result);

int main() {

    int check = 0;
    Mat lineDection;
    int tempX = 0;
    int tempY = 0;
    int tempX2 = 0;
    int tempY2 = 0;
    vector<Rect> found;

    Mat frame, edge_l, edge_r, frame_roi_l, frame_roi_r, frame_roi_front, result;

    vector<Vec4i> lines_l, lines_r;

    VideoCapture cap("Pedestrian_3.mp4");

    if (!cap.isOpened()) {
        cout << "can't open video file" << endl;
        return 0;
    }

    //******************************************************************

    String modelConfiguration = "deep/yolov2.cfg";
    String modelBinary = "deep/yolov2.weights";
    Net net = readNetFromDarknet(modelConfiguration, modelBinary);

    Mat inputBlob, detectionMat;
    float confidenceThreshold = 0.24;
    vector<String> classNamesVec;
    ifstream classNamesFile("deep/coco.names");

    int maxSize_prev = 0, frame_cnt = 0, frame_freq = 10;
    float sub_size = 0;

    if (classNamesFile.is_open()) {
          string className = "";
          while (std::getline(classNamesFile, className)) classNamesVec.push_back(className);
    }

    Mat frame_roi,frame_ped;

    //******************************************************************
    HOGDescriptor hog( Size(48, 96),Size(16, 16), Size(8, 8), Size(8, 8), 9);
    hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());

    //******************************************************************

    while (1) {

        cap >> frame;
        if (frame.empty()) break;

        result = frame.clone();

        Rect rect_l(Point(100,frame.rows/2),Point(frame.cols/2,frame.rows));
        Rect rect_r(Point(frame.cols/2-100,frame.rows/2),Point(frame.cols-100,frame.rows));
        Rect rect_front(Point(210,frame.rows/2-30),Point(frame.cols-360,frame.rows));
        Rect rect_ped(Point(100,frame.rows/2-20),Point(frame.cols-200,frame.rows));

        frame_roi_l = frame(rect_l);
        frame_roi_r = frame(rect_r);
        frame_roi = frame(rect_front);

        Mat poly_roi;
        Mat poly_mask = Mat::zeros(frame.size(), frame.type());
        Point poly[1][4];
        poly[0][0] = Point(300, frame.rows/2);
        poly[0][1] = Point(frame.cols-300,frame.rows/2);
        poly[0][2] = Point(frame.cols, frame.rows);
        poly[0][3] = Point(0, frame.rows);
        const Point* ppt[1] = { poly[0] };
        int npt[] = { 4 };
        // function that draws polygon with given points
        fillPoly(poly_mask, ppt, npt, 1, Scalar(255, 255, 255), 8);
        frame.copyTo(poly_roi, poly_mask);
        imshow("polyROI", poly_roi);

        frame_ped = poly_roi;

        cvtColor(frame, frame, CV_BGR2GRAY);

//******************************************************************

        if(frame_cnt % frame_freq == 0){
            inputBlob = blobFromImage(frame_roi, 1 / 255.F, Size(416, 416), Scalar(), true, false);
            net.setInput(inputBlob, "data");
            detectionMat = net.forward("detection_out");
        }

        int maxSize = 0;

        for (int i = 0; i < detectionMat.rows; i++) {
            const int probability_index = 5;
            const int probability_size = detectionMat.cols - probability_index;
            float * prob_array_ptr = &detectionMat.at<float>(i, probability_index);
            size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;

            float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

            if (confidence > confidenceThreshold) {

                float x_center = detectionMat.at<float>(i, 0) * frame_roi.cols;
                float y_center = detectionMat.at<float>(i, 1) * frame_roi.rows;
                float width = detectionMat.at<float>(i, 2) * frame_roi.cols;
                float height = detectionMat.at<float>(i, 3) * frame_roi.rows;
                Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
                Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));
                Rect object(p1, p2);
                Scalar object_roi_color(0, 255, 0);

                Size s = object.size();
                int object_size = s.width * s.height;

                if (maxSize < object_size) {
                    maxSize = object_size;
                }

                rectangle(frame_roi, object, object_roi_color);
            }
        }

        if(frame_cnt % frame_freq == 0){
            sub_size = (maxSize - maxSize_prev) / float(maxSize);
            //printf("count : %f maxSize : %d sub : %d \n",sub_size,maxSize,maxSize - maxSize_prev);
            if(-0.1 > sub_size && sub_size > -0.8 && (maxSize - maxSize_prev)<(-2000))
            {
                check = 1;
            }
            else if(sub_size>0.1&&maxSize>10000)
            {
                check = 2;
            }
            else{
                check = 0;
            }
        }

        if (check==1)
        {
            putText(result, "Warning!: Front car departure", Point(40, 120), 0, 1, Scalar(0, 0, 255), 3);
            printf("Detected!");
        }

        else if (check==2)
        {
            putText(result, "Warning!: Collision with front car", Point(40, 120), 0, 1, Scalar(0, 0, 255), 3);
            printf("2Detected!");
        }

        maxSize_prev = maxSize;
        frame_cnt++;

        imshow("1",frame_roi);
        //******************************************************************
        // detect
        hog.detectMultiScale( frame_ped,
        found,
        1.2,
        Size(8, 8), Size(32, 32), 1.05,
        6);
        int max = -1;
        for (int i = 0; i < (int)found.size(); i++){
            Size s = found[i].size();
            int object_size = s.width * s.height;
            //printf("found[i] : %d\n",object_size);
            if(max<object_size) max = object_size;
            rectangle(frame_ped, found[i], Scalar(0, 255, 0), 2);
        }
        //printf("found[i] : %d\n",max);

        if(max>4000)
             putText(result,"Warning!: Collision with the pedestrian", Point(40, 40), 0, 1, Scalar(0, 0, 255), 3);
        //******************************************************************

        rectangle(result,rect_l, Scalar(0,0,255));
        rectangle(result,rect_r, Scalar(0,255,255));

        Canny(frame_roi_l, edge_l, 50, 150, 3);
        Canny(frame_roi_r, edge_r, 50, 150, 3);


        HoughLinesP(edge_l, lines_l, 1, CV_PI / 180, 50, 10, 400);
        HoughLinesP(edge_r, lines_r, 1, CV_PI / 180, 50, 10, 400);

        double slope = 0 , x = 0, y = 0;
        double xMax = -100, yMax =0.0;
        double xMax2 = 1000, yMax2 =0.0;
        double x1,x2,y1,y2;
        for (int i = 0; i < lines_l.size(); i++)
        {
            Vec4i l = lines_l[i];
            if((l[0]-l[2]) == 0) continue;
            x1 = l[0];
            x2 = l[2];
            y1 = l[1];
            y2 = l[3];
            slope = (y1-y2)/(x1-x2);
            if(slope<-0.4&&slope>-9)
            {
                y = y1-slope*x1;
                x = -(y/slope);
                if(xMax<x)
                {
                    xMax = x;
                    yMax = y;
                    tempX = x;
                    tempY = y;
                }
                line(result, Point(l[0]+100, l[1]+frame.rows/2), Point(l[2]+100, l[3]+frame.rows/2), Scalar(0, 0, 255), 3, 8);
            }
        }
        for (int i = 0; i < lines_r.size(); i++)
        {
            Vec4i l = lines_r[i];
            if((l[0]-l[2]) == 0) continue;
            x1 = l[0];
            x2 = l[2];
            y1 = l[1];
            y2 = l[3];
            slope = (y1-y2)/(x1-x2);
            if(slope>0.5&&slope<9)
            {
                y = y1-slope*x1;
                x = -(y/slope);
                if(xMax2>x)
                {
                    xMax2 = x;
                    yMax2 = y;
                    tempX2 = x;
                    tempY2 = y;
                }
                line(result, Point(l[0]+frame.cols/2-100, l[1]+frame.rows/2), Point(l[2]+frame.cols/2-100, l[3]+frame.rows/2), Scalar(0, 0, 255), 3, 8);
            }
        }
        if(xMax==-100)
        {
            xMax = tempX;
            yMax = tempY;
        }
        if(xMax2==1000)
        {
            xMax2 = tempX2;
            yMax2 = tempY2;
        }

        if(xMax+100<380)
        {
            putText(result,"Warning!: Lane departure", Point(40, 80), 0, 1, Scalar(0, 0, 255), 3);
        }
        if(xMax2+frame.cols/2-100>400)
        {
            putText(result,"Warning!: Lane departure", Point(40, 80), 0, 1, Scalar(0, 0, 255), 3);
        }
        line(result, Point(xMax+100,frame.rows/2), Point(0+100,yMax+frame.rows/2), Scalar(0,255,0), 3, 8);
        line(result, Point(xMax2+frame.cols/2-100,frame.rows/2), Point(frame.cols/2-100,yMax2+frame.rows/2), Scalar(255,255,255), 3, 8);
        imshow("Hough Transform", result);
        imshow("ped", frame_ped);
        waitKey(33);
    }

}

void drawLine(vector<Vec2f> lines, float x, float y, float angle_th1, float angle_th2, Mat &result){
    float rho, theta, a, b, x0, y0;
    float avr_rho=0. , avr_theta=0.;
    int count = 0;
    Point p1, p2;

    for (int i = 0; i < lines.size(); i++) {
        rho = lines[i][0];
        theta = lines[i][1];

        if (theta < CV_PI / 180 * angle_th1 || theta > CV_PI / 180 * angle_th2) continue;

        avr_rho += rho;
        avr_theta += theta;
        count++;
    }

    avr_rho /= count;
    avr_theta /= count;
    a = cos(avr_theta);
    b = sin(avr_theta);

    x0 = a * avr_rho;
    y0 = b * avr_rho;

    p1 = Point(cvRound(x0 + 1000 * (-b)) + x, cvRound(y0 + 1000 * a) + y);
    p2 = Point(cvRound(x0 - 1000 * (-b)) + x, cvRound(y0 - 1000 * a) + y);

    line(result, p1, p2, Scalar(0, 0, 255), 3, 8);
}
