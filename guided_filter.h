#ifndef GUIDED_FILTER
#define GUIDED_FILTER

#ifdef __ANDROID__
#include "inc/pw_datatype.h"
#include "inc/pw_typedef.h"
#include "inc/pwerror.h"
#include "inc/pwmacro.h"
#else
#include "pw_datatype.h"
#include "pw_typedef.h"
#include "pwerror.h"
#include "pwmacro.h"
#endif

#include <opencv2/opencv.hpp>

class GuidedFilter
{
public:
    GuidedFilter(int r, float eps, float alpha, float beta);
    std::pair<cv::Mat, cv::Mat> guidedFilter(const cv::Mat &I, const cv::Mat &p, int height, int width);
    void setAlpha(float alpha);
    void setBeta(float beta);
    void setRadius(float radius);

private:
    int r;          // Radius
    float eps;      // Regularization parameter
    float alpha;    // Contrast parameter
    float beta;     // Brightness parameter
};

#endif // GUIDED_FILTER