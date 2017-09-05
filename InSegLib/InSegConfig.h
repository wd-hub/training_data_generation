// ---------------------------------------------------------------------------
// @file       InSegConfig.h
// @author     Keisuke Tateno
// @date       2015/02/01
// ---------------------------------------------------------------------------

#ifndef _H_INC_INSEG_CONFIG_
#define _H_INC_INSEG_CONFIG_


#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

#define USE_LABEL_CONFIDENCE

namespace InSeg
{
    

class CamParams
{
public:
    /// Focal Length
    float fx;
    float fy;
    /// Principal Point
    float cx;
    float cy;
    
    /// Image Size
    int imgWidth;
    int imgHeight;
    
    /// Dist coeffs
    float k1;
    float k2;
    float k3;
    
    // Tangential Dist coeffs
    float p1;
    float p2;
    
    /// Dist coeffs
    float k4;
    float k5;
    float k6;
    
    float s1;
    float s2;
    float s3;
    float s4;
    
public:
    CamParams();
    CamParams(const cv::Mat* cameraMatrix, const cv::Mat* distortionCoeffs);
    virtual ~CamParams();
    
    CamParams createScaledCamParams(float scale) const;
    
    void denormalize(const Eigen::Vector2f& xy, Eigen::Vector2f& uv) const {
        uv.x() = xy.x() * fx + cx;
        uv.y() = xy.y() * fy + cy;
    }
    
    void normalize(const Eigen::Vector2f& uv, Eigen::Vector2f& xy) const {
        xy.x() = (uv.x() - cx) / fx;
        xy.y() = (uv.y() - cy) / fy;
    }
};
    
    
    

#define INSEG_CONFIG InsegConfig::getInstance()
    
// Singleton config class
class InsegConfig
{
public:
    // get instanse
    static InsegConfig& getInstance()
    {
        return InsegConfigInstance;
    }

    /// set camera paramter
    void setCamParams(const CamParams& cparam);

    /// get camera paramater of image pyramids
    CamParams& getCamParams(int level = 0);

    /// get camera paramater of index image 
    CamParams& getIndexCamParams();
    

public:

    /// camera paramaters
    std::vector<CamParams> cparams;

    /// camera paramaters of index image
    CamParams cparamIndex;

    /// number of image pyramid level
    int maxPyrLevel;
    /// main pyramid level for main process
    int mainPyrLevel;


    /// uncertainty of depth measurement
    float depthUncertaintyCoef;
    /// resultion of depth point [mm]
    float samePointThresh;
    /// Threshold of confidence in point fusion process
    int threshOfStable;

    /// culling thresh
    float nearThresh;
    float farThresh;

    /// number of valid points for tracking
    int maxTrackingPointNum;

    /// maxinum number of iteration
    int maxIterationNum;
    /// pose delta criteria on tracking 
    float criteriaDelta;

    float updatePointAngleThresh;
    
    /// iteration number of 3x3 bilateral filtering
    int bilateralFilteringNum;

    /// segmentation method (0: normal edge based, 1: minimum spanning tree based)
    int framewiseSegmentationType;
    
    /// angle threshold for normal difference between neighboring pixels (cosine value)
    float depthEdgeThresh;
    /// flag for ingnoring convex shape border
    bool bCheckConvexity;
    
    /// tuning parameter for coarseness on minimum spanning tree segmentation
    float mstSegmKParam;
    /// minimum size of segment region area
    int minRegionSize;

    int labelMerginConfidenceThresh;
    int labelupdateConfidenceMax;
    
    bool bRemoveDynamicPoint;
    
    bool bUseReferencePose;
    
public:
    /// initialize paramaters to default
    void setDefault()
    {
        mainPyrLevel = 2;
        
        maxPyrLevel = mainPyrLevel+3;
        
        // depth uncertainty of xtion pro live
        depthUncertaintyCoef = 0.0000285f;
        
        // depth resulution is 1[mm]
        samePointThresh = 1;
        //
        threshOfStable = 5;
        
        nearThresh = 400;
        farThresh  = 10000;
        
        maxTrackingPointNum = 4800;
        
        maxIterationNum = 20;
        criteriaDelta = 1.0e-5f;
        
        bilateralFilteringNum = 1;
        
        framewiseSegmentationType = 0;
        
        minRegionSize = 5;
        
        depthEdgeThresh = 0.94f;
        bCheckConvexity = true;
        
        mstSegmKParam = 0.05f;
        
        labelMerginConfidenceThresh = 3;
        labelupdateConfidenceMax = 10;
        
        bRemoveDynamicPoint = false;
        
        updatePointAngleThresh = 20;
        
        bUseReferencePose = false;
    }
    
private:
    InsegConfig();

    InsegConfig(const InsegConfig &other);
    InsegConfig &operator=(const InsegConfig &other);

private:
    static InsegConfig InsegConfigInstance;

};

}

#endif
