#include <opencv2/highgui.hpp>
#include <aruco.h>
using namespace std;
int main(int argc,char **argv)
{
   cv::Mat im=cv::imread(argv[1]);
   aruco::FractalDetector FDetector;
   FDetector.setConfiguration("FRACTAL_4L_6");
   
    if (FDetector.detect(im)) {
        FDetector.drawMarkers(im);
    } else {
        cout << "No markers detected" << endl;
    }

   cv::imshow("image",im);
   cv::waitKey(0);
}