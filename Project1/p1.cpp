#include "cv.hpp"
#include <iostream>

using namespace cv;
using namespace std;


int main()
{
    Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2();
    Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
    Mat frame,gray, foregroundMask, result;
    VideoCapture cap("cv_live2.mp4");
    int fps = cap.get(CV_CAP_PROP_FPS);

    while (1)
    {
        cap >> frame;
        if (frame.empty()) break;
        result = frame.clone();

        double average = 0.0;
        cvtColor(frame, gray, CV_BGR2GRAY);
        MatIterator_ <uchar> it, end;
        for(it = gray.begin<uchar>(),end=gray.end<uchar>();it != end; ++it)
        {
            average += *it;
        }
        average = average / double(frame.cols*frame.rows);
        printf("%f\n",average);
        /*
        if(average<80)
        {
            unsigned char pix[256];
            float gammaValue_dark = (sum(gray)[0] / gray.total()) / 10;

            for (int i = 0; i < 256; i++) {
                pix[i] = saturate_cast<uchar>(pow((float)(i / 255.0), gammaValue_dark) * 255.0f);
            }
            for (it = gray.begin<uchar>(), end = gray.end<uchar>(); it != end; it++) {
                *it = pix[(*it)];
            }
        }d
        */
        if (foregroundMask.empty()) foregroundMask.create(gray.size(), gray.type());

        bg_model->apply(gray, foregroundMask);
        GaussianBlur(foregroundMask, foregroundMask, Size(11, 11), 3.5, 3.5);
        threshold(foregroundMask, foregroundMask, 10, 255, THRESH_BINARY);

        vector<vector<Point>> contours;
        vector<Vec4i>hierarchy;

        erode(foregroundMask, foregroundMask, element);
        morphologyEx(foregroundMask, foregroundMask, MORPH_OPEN, element);

        findContours(foregroundMask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        imshow("0",foregroundMask);
        vector<Rect> boundRect(contours.size());
        int count = 0;
        for (int i = 0; i < contours.size(); i++)
        {
            boundRect[i] = boundingRect(contours[i]);
            Size s = boundRect[i].size();
            int object_size = s.width * s.height;
            if(object_size>4000)
            {
                rectangle(result, boundRect[i], Scalar(255, 255, 255), 2, 8);
                count ++;
            }
        }

        putText(result, format("# Rect: % d", count), Point2f(10, 30), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
        imshow("result", result);

        waitKey(1000 / fps);
    }
    return 0;
}
