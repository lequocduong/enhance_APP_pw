#include "guided_filter.h"

GuidedFilter::GuidedFilter(int r, float eps, float alpha, float beta) : r(r), eps(eps), alpha(alpha), beta(beta) {}

void GuidedFilter::setAlpha(float alpha) { this->alpha = alpha; }
void GuidedFilter::setBeta(float beta) { this->beta = beta; }
void GuidedFilter::setRadius(float radius) { this->r = radius; }

std::pair<cv::Mat, cv::Mat> GuidedFilter::guidedFilter(const cv::Mat &I, const cv::Mat &p, int height, int width)
{
    cv::Mat mean_I, mean_p, mean_Ip, cov_Ip, mean_II, var_I, a, b, mean_a, mean_b;

    // Step 1: Calculate mean of I and p
    cv::boxFilter(I, mean_I, CV_32F, cv::Size(r, r));
    cv::boxFilter(p, mean_p, CV_32F, cv::Size(r, r));

    // Step 2: Calculate mean of I*p and covariance
    cv::boxFilter(I.mul(p), mean_Ip, CV_32F, cv::Size(r, r));
    cov_Ip = mean_Ip - mean_I.mul(mean_p);

    // Step 3: Calculate variance of I
    cv::boxFilter(I.mul(I), mean_II, CV_32F, cv::Size(r, r));
    var_I = mean_II - mean_I.mul(mean_I);

    // Step 4: Calculate coefficients a and b
    a = cov_Ip / (var_I + eps);
    b = mean_p - a.mul(mean_I);

    // Step 5: Calculate mean of a and b
    cv::boxFilter(a, mean_a, CV_32F, cv::Size(r, r));
    cv::boxFilter(b, mean_b, CV_32F, cv::Size(r, r));

    // Step 6: Resize mean_a and mean_b to original image size
    cv::resize(mean_a, mean_a, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    cv::resize(mean_b, mean_b, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

    // Adjust mean_a and mean_b
    mean_a = mean_a.mul(alpha);
    mean_b = mean_b.mul(beta);

    // Return the pair of results
    return std::make_pair(mean_a, mean_b);
}