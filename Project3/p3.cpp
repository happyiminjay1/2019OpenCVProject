#include "cv.hpp"
#include <iostream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;

void detect_text(string input){
    Mat rgb = imread(input);

    Mat result = rgb.clone();

    Mat extractRed;
    cvtColor(rgb, extractRed, CV_BGR2YCrCb);
    inRange(extractRed, Scalar(0, 0, 0), Scalar(255,150,255), extractRed);

    imshow("Red", extractRed);

    MatIterator_ <uchar> it, end;

    for (it = extractRed.begin<uchar>(), end = extractRed.end<uchar>(); it != end; ++it) {
        if(*it==255) *it = 0;
        else *it = 255;
    }
    imshow("RedOpposite", extractRed);

    Point p1, p2;
    vector<Vec4i> lines;

    HoughLinesP(extractRed, lines, 1, CV_PI / 180, 50, 20, 10);
    for (int i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        line(result, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, 8);
    }

    imshow("result", result);

    Mat gray;
    cvtColor(rgb, gray, CV_BGR2GRAY);

    Mat grad;
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(gray, grad, MORPH_GRADIENT, element);
    //adjust gradient
    
    Mat thresh;
    threshold(grad, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

    Mat combineWord;
    element = getStructuringElement(MORPH_CROSS, Size(10, 2));
    morphologyEx(thresh, combineWord, MORPH_CLOSE, element);
    // INVERSE로 하면 thresh 깔끔하게 글이 나옴.

    Mat thresh2 = thresh.clone();

    for (it = thresh2.begin<uchar>(), end = thresh2.end<uchar>(); it != end; ++it) {
        if(*it==255) *it = 0;
        else *it = 255;
    }

    Mat mask = Mat::zeros(thresh.size(), CV_8UC1);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(combineWord, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    //외각선 추출

    // filter contours
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0]){
        Rect rect = boundingRect(contours[idx]);
        Mat maskROI(mask, rect);
        maskROI = Scalar(0, 0, 0);
        // fill the contour
        drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
        RotatedRect rrect = minAreaRect(contours[idx]);
        //double r = (double)countNonZero(maskROI) / (rrect.size.width * rrect.size.height);

        for(int idx = 0; idx >= 0; idx = hierarchy[idx][0]){
            Rect rect = boundingRect(contours[idx]);
            Mat maskROI(mask, rect);
            maskROI = Scalar(0, 0, 0);
            // fill the contour
            drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
            RotatedRect rrect = minAreaRect(contours[idx]);


            // assume at least 25% of the area is filled if it contains text
            if ((rrect.size.height > 8 && rrect.size.width > 8))
            {
                Point2f pts[4];
                rrect.points(pts);

                int middleX, middleY;
                int left, right;

                for (int i = 0; i < lines.size(); i++) {
                    Vec4i l = lines[i];
                    middleX = ((int)pts[0].x + (int)pts[2].x)/2;
                    middleY = ((int)pts[0].y + (int)pts[2].y)/2;
                    if(l[0]<l[2]){
                        left = l[0];
                        right = l[2];
                    }
                    else{
                        left = l[2];
                        right = l[0];
                    }
                    line(rgb, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, 8);
                    if(left<middleX && middleX<right &&
                       ((abs(l[1]-middleY)<20) || (abs(l[3]-middleY)<20)))
                    {
                    rectangle(thresh2, Point((int)pts[0].x, (int)pts[0].y), Point((int)pts[2].x, (int)pts[2].y), Scalar(255,255,255),-1);
                    }

                }
            }

        }
    }

    imshow("rgb", rgb);
    imshow("00", thresh2);

    printf("finished!");
}

int main()
{
    detect_text(string("pic5.jpeg"));
    waitKey();
    return 0;
}
