#include "enhancer.h"
#include "pw_base_api.h"

bool EnhanceModel::load(/*AAssetManager *mgr, */const char *model_param, const char *model_bin,
                        int input_height, int input_width)
{
    if (net_.load_param(/*mgr, */model_param) != 0 || net_.load_model(/*mgr, */model_bin) != 0)
        return false;

    input_name_ = "in0";
    output_name_ = "out0";
    input_height_ = input_height;
    input_width_ = input_width;
    return true;
}

cv::Mat EnhanceModel::forward(cv::Mat &frame)
{
    ncnn::Mat in;
    ncnn::Mat out;
    const float scal[] = {0.003915, 0.003915, 0.003915};
    const float scal2[] = {255, 255, 255};
    ncnn::Extractor extractor = net_.create_extractor();

    // Convert the frame to ncnn Mat format
    in = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_RGB, frame.cols, frame.rows);

    // Normalize the input frame to the range of 0-1
    in.substract_mean_normalize(0, scal); // 0-255  -->  0-1

    // Perform Inference
    extractor.input(input_name_, in);
    extractor.extract(output_name_, out);

    // Denormalize the output
    out.substract_mean_normalize(0, scal2);

    // Convert NCNN output back to OpenCV Mat
    cv::Mat ncnn_output_image(out.h, out.w, CV_8UC3);
    out.to_pixels(ncnn_output_image.data, ncnn::Mat::PIXEL_RGB);
    return ncnn_output_image;
}

Enhancer::Enhancer()
{
    int r = 32;
    float eps = 0.000000001f;
    float alpha = 1.5f;
    float beta = 0.8f;
    gf_ = new GuidedFilter(r, eps, alpha, beta);

    skipframe_ = 30;
    counter_ = 0;
}

Enhancer::~Enhancer()
{
    delete gf_;
}

void Enhancer::setModel(EnhanceModel *model)
{
    if (!model)
        return;

    model_ = model;
    // Adjust the radius of the Guided Filter if the input shape is too large
    if (model->input_width_ > 640)
    {
        gf_->setRadius(512);
    }
    else if (model->input_width_ > 320)
    {
        gf_->setRadius(512);
    }
    else
    {
        gf_->setRadius(64);
    }
}

void Enhancer::adjustContrast(float alpha) { gf_->setAlpha(alpha); }

void Enhancer::adjustBrightness(float beta) { gf_->setBeta(beta); }

void Enhancer::setBatchSize(int k) { skipframe_ = k; }

void Enhancer::yuv2mat(GVM_SCREEN *yuv, cv::Mat &rgb_frame)
{
    int width = yuv->lWidth;
    int height = yuv->lHeight;
    int type = yuv->ImgFormat;
    // YUV to RGB conversion using OpenCV
    cv::Mat yuv_frame(height / 2 * 3, width, CV_8UC1);

    if (type == VM_FORMAT_YUV420_P)
    {
        memcpy(yuv_frame.data, (uint8_t *)yuv->PixelArray.planar.pPixelArray[0], height * width);
        memcpy(yuv_frame.data + height * width, (uint8_t *)yuv->PixelArray.planar.pPixelArray[1], height * width / 4);
        memcpy(yuv_frame.data + height * width + (height * width / 4), (uint8_t *)yuv->PixelArray.planar.pPixelArray[2], height * width / 4);

        // Convert from YUV_420P to RGB
        cv::cvtColor(yuv_frame, rgb_frame, cv::COLOR_YUV2RGB_I420);
    }
    else if (type == VM_FORMAT_NV12)
    {
        memcpy(yuv_frame.data, (uint8_t *)yuv->PixelArray.planar.pPixelArray[0], height * width);
        memcpy(yuv_frame.data + height * width, (uint8_t *)yuv->PixelArray.planar.pPixelArray[1], height * width / 2);

        // Convert from YUV_NV12 to RGB
        cv::cvtColor(yuv_frame, rgb_frame, cv::COLOR_YUV2RGB_NV12);
    }
}

GVM_SCREEN * Enhancer::process(GVM_SCREEN *src)
{
    if (!src || !model_)
        return nullptr;

    int width = src->lWidth;
    int height = src->lHeight;
    GVM_SCREEN *dst = pwbCreateImage(width, height, VM_FORMAT_YUV420_P);
    if (!dst)
        return nullptr;

    // YUV to RGB conversion using OpenCV short version
    cv::Mat rgb_frame(height, width, CV_8UC3);
    cv::Mat enhanced_rgb_frame(height, width, CV_8UC3);

    yuv2mat(src, rgb_frame);
    enhance(rgb_frame, enhanced_rgb_frame);
    mat2yuv(dst, enhanced_rgb_frame);

    return dst;
}

void Enhancer::mat2yuv(GVM_SCREEN *yuv, const cv::Mat &rgb_frame)
{
    int width = yuv->lWidth;
    int height = yuv->lHeight;
    cv::Mat yuv_frame(height / 2 * 3, width, CV_8UC1);

    // Convert RGB back to YUV_420P format
    cv::cvtColor(rgb_frame, yuv_frame, cv::COLOR_RGB2YUV_I420);

    // Copy the channels back to GVM_SCREEN format
    memcpy((uint8_t *)yuv->PixelArray.planar.pPixelArray[0], yuv_frame.data, height * width);
    memcpy((uint8_t *)yuv->PixelArray.planar.pPixelArray[1], yuv_frame.data + height * width, height * width / 4);
    memcpy((uint8_t *)yuv->PixelArray.planar.pPixelArray[2], yuv_frame.data + height * width + (height * width / 4), height * width / 4);
}

void Enhancer::enhance(cv::Mat &frame, cv::Mat &enhanced_frame)
{
    (counter_ % skipframe_ == 0) ? enhanceFrameNCNN(frame, enhanced_frame) : enhanceFrameGuidedFilter(frame, enhanced_frame);
    counter_ = (counter_ + 1) % skipframe_;
}

void Enhancer::enhanceFrameNCNN(cv::Mat &frame, cv::Mat &enhanced_frame)
{
    // Resize the frame to the desired size
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(model_->input_width_, model_->input_height_));
    cv::Mat ncnn_output_image = model_->forward(resized_frame);

    // Convert to float
    resized_frame.convertTo(resized_frame, CV_32FC3, 1.0 / 255.0);
    frame.convertTo(frame, CV_32FC3, 1.0 / 255.0);
    ncnn_output_image.convertTo(ncnn_output_image, CV_32FC3, 1.0 / 255.0);

    // Apply the guided filter
    std::pair<cv::Mat, cv::Mat> result = gf_->guidedFilter(resized_frame, ncnn_output_image, frame.rows, frame.cols);
    matrix_A_ = result.first;
    matrix_B_ = result.second;

    // Calculate the output
    enhanced_frame = matrix_A_.mul(frame) + matrix_B_;

    // Convert the final output back to 8-bit
    enhanced_frame.convertTo(enhanced_frame, CV_8UC3, 255.0);
}

void Enhancer::enhanceFrameGuidedFilter(cv::Mat &frame, cv::Mat &enhanced_frame)
{
    // Normalize the input frame
    frame.convertTo(frame, CV_32FC3, 1.0 / 255.0);

    // Calculate the output
    enhanced_frame = matrix_A_.mul(frame) + matrix_B_;

    // Convert the final output back to 8-bit
    enhanced_frame.convertTo(enhanced_frame, CV_8UC3, 255.0);
}
