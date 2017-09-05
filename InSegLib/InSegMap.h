// ---------------------------------------------------------------------------
// @file       InSegMap.h
// @author     Keisuke Tateno
// @date       2015/02/01
// ---------------------------------------------------------------------------

#ifndef _H_INC_INSEG_MAP_DATA_
#define _H_INC_INSEG_MAP_DATA_

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>


namespace InSeg
{


/// Point with normal data
struct Surfel
{
    Eigen::Vector3f pos;
    Eigen::Vector3f normal;

    float radius;
    float confidence;

    int time;

    int label;
    int labelConfidence;

    bool isValid;
    bool isStable;

    uchar col[3];

    Surfel()
    : radius(0), confidence(0), time(0), isValid(false), isStable(false),  label(0), labelConfidence(0)
    {
        col[0] = col[1] = col[2] = 255;
    }

    bool operator < (const Surfel &ref) const {
        return (this->isValid == true && ref.isValid == false);
    }
    
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class WorldMap
{
public:
    WorldMap(){
        surfels.reserve(50000);
    }
    virtual ~WorldMap(){}

    std::vector<Surfel>& getSurfels()
    {
        return surfels;
    }

    /// save model as .ply
    bool saveModel(const char* filename, bool hasColor = false);
    /// loda model as .ply
    bool loadModel(const char* filename);

protected:
    // point clouds
    std::vector<Surfel> surfels;

};


class FramePyramid
{
public:

    FramePyramid();
    virtual ~FramePyramid();

    FramePyramid(FramePyramid const & other);
    FramePyramid& operator=(FramePyramid const & other);

    /// create image pyramid  of vertex map from input depth image
    void createDepthPyramid(cv::Mat& depthImg);

    void createDepthPyramid(cv::Mat& depthImg, cv::Mat& colorImg);

    ///
    void pyrDown();

public:
    /// get vertex map at the level
    cv::Mat& getDepthImage(int level = 0);

    /// get vertex map at the level
    cv::Mat& getVertexMap(int level = 0);

    /// get normal map at the level
    cv::Mat& getNormalMap(int level = 0);

    /// get normal map at the level
    cv::Mat& getLabelMap(int level = 0);

    /// get color image at the level
    cv::Mat& getColorImage(int level = 0);
    cv::Mat& getGrayImage(int level = 0);

    /// get smoothed vertex map at the level
    cv::Mat& getSmoothDepthImage(int level = 0);

    /// get smoothed vertex map at the level
    cv::Mat& getSmoothVertexMap(int level = 0);

    /// get combind map at the level
    cv::Mat& getCombMap(int level = 0);
    
    /// get index map at the level (Just only "level == INSEG_CONFIG.mainPyrLevel" is available)
    cv::Mat& getIndexMap(int level = 0);

    void setPose(Eigen::Matrix4f& pose);
    Eigen::Matrix4f& getPose();

protected:

    /// Image pyramids  of depth image
    std::vector<cv::Mat> m_pyrDepthImg;

    /// Image pyramids  of depth map
    std::vector<cv::Mat> m_pyrVertexMap;

    /// Image pyramids of normal map
    std::vector<cv::Mat> m_pyrNormalMap;

    /// Image pyramids of label map
    std::vector<cv::Mat> m_pyrLabelMap;

    /// pose of this frame
    Eigen::Matrix4f m_pose;

    /// Image pyramids of color image
    std::vector<cv::Mat> m_pyrColorImg;

    std::vector<cv::Mat> m_pyrGrayImg;

    /// Image pyramids of combind map of intensity and depth and derivatives for tracking
    std::vector<cv::Mat> m_pyrCombMap;

    /// Image pyramids of index map
    std::vector<cv::Mat> m_pyrIndexMap;

    /// Image pyramids  of smoothed depth image
    std::vector<cv::Mat> m_pyrSmoothDepthImg;

    /// Image pyramids  of smoothed depth map
    std::vector<cv::Mat> m_pyrSmoothVertexMap;
    
};
    
void calcLabelToColor(unsigned short label, cv::Vec3b& color);
    


}

#endif
