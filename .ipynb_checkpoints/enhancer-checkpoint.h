#ifndef _ENHANCER_H_
#define _ENHANCER_H_

#include <net.h>
#include <opencv2/opencv.hpp>
#include "guided_filter.h"
#include "pw_typedef.h"

class Enhancer;

class EnhanceModel
{
public:
    EnhanceModel() = default;
    ~EnhanceModel() = default;

    /// @brief Load NCNN model
    /// @param model_param  NCNN model param file
    /// @param model_bin    NCNN model bin file
    /// @param input_width  Input width of the NCNN model
    /// @param input_height Input height of the NCNN model
    bool load(/*AAssetManager *mgr, */const char *model_param, const char *model_bin,
              int input_height, int input_width);

private:
    /// @brief Run the model
    /// @param frame Input image
    cv::Mat forward(cv::Mat &frame);

private:
    ncnn::Net net_;              // NCNN model
    const char *input_name_;     // Input name of the NCNN model (preset)
    const char *output_name_;    // Output name of the NCNN (preset)
    int input_width_;            // Input width of the NCNN model (preset)
    int input_height_;           // Input height of the NCNN model (preset)

    friend class Enhancer;
};

class Enhancer
{
public:
    Enhancer();
    ~Enhancer();

    /// @brief Set the NCNN model for enhance
    /// @param model
    void setModel(EnhanceModel *model);

    /// @brief The process flow: YUV2RGB -> Enhance -> RGB2YUV
    /// @param src GVM_SCREEN in YUV420P or NV12 format
    /// @return GVM_SCREEN in YUV420P only
    GVM_SCREEN * process(GVM_SCREEN *src);

    /// @brief User adjustable contrast parameter of enhancement effects
    /// @param alpha should range between 1 to 1.5 for better experience
    void adjustContrast(float alpha);

    /// @brief User adjustable brightness parameter of enhancement effects
    /// @param beta should range between 0.7 to 1.5 for better experience
    void adjustBrightness(float beta);

    /// @brief Set the batchsize for skipping frames, only the first frame will be enhanced by NCNN model
    /// @param k number of frames in batch
    void setBatchSize(int k);

private:
    /// @brief Load Bytes from GVM_SCREEN and return cv::Mat RGB frame
    /// @param src GVM_SCREEN in YUV420P or NV12 format
    /// @param rgb_frame the returned cv::Mat in RGB format
    void yuv2mat(GVM_SCREEN *yuv, cv::Mat &rgb_frame);

    /// @brief RGB2YUV conversion using OpenCV, the output in GVM_SCREEN YUV420P format
    /// @param dst the returned GVM_SCREEN in YUV420P only
    /// @param rgb_frame source cv::Mat in RGB format
    void mat2yuv(GVM_SCREEN *yuv, const cv::Mat &rgb_frame);

    /// @brief Perform enhancement, the first frame of batch will be enhanced by NCNN, the others will be enhanced by Guided Filter
    /// @param frame the original frame in cv::Mat RGB format
    /// @param enhanced_frame the enhanced frame in cv::Mat RGB format
    void enhance(cv::Mat &frame, cv::Mat &enhanced_frame);

    /// @brief Perform enhancement with NCNN model, then apply Guided Filter
    /// @param frame the original frame in cv::Mat RGB format
    /// @param enhanced_frame the enhanced frame in cv::Mat RGB format
    void enhanceFrameNCNN(cv::Mat &frame, cv::Mat &enhanced_frame);

    /// @brief After having matrix A and matrix B, all other frames in the batch are enhanced by a Linear Transformation
    /// @param frame the original frame in cv::Mat RGB format
    /// @param enhanced_frame the enhanced frame in cv::Mat RGB format
    void enhanceFrameGuidedFilter(cv::Mat &frame, cv::Mat &enhanced_frame);

private:
    EnhanceModel *model_;
    int counter_;                // current counter of frame in the batch
    int skipframe_;              // Number of frames in the batch
    GuidedFilter *gf_;           // Guided Filter object
    cv::Mat matrix_A_;
    cv::Mat matrix_B_;
};

#endif // _ENHANCER_H_
