// ---------------------------------------------------------------------------
// @file       InSegLib.h
// @author     Keisuke Tateno
// @date       2015/02/01
// ---------------------------------------------------------------------------

#ifndef _H_INC_INSEG_INTERFACE_
#define _H_INC_INSEG_INTERFACE_


#include "opencv2/opencv.hpp"
#include <Eigen/Dense>
#include "InSegMap.h"


namespace InSeg
{


class InSegSLAM;

/// InSeg SLAM interface class
class InSegLib
{
public:
    InSegLib();
    virtual ~InSegLib();

    /// main process
    void processFrame( cv::Mat& depthMap, cv::Mat& bgrImage );

    /// initialize map data
    void initializeMap();

    /// get the current camera pose
    Eigen::Matrix4f& getCurrentPose();

    /// get the current image pyramid
    FramePyramid& getFramePyramid();
    
    /// get the rendered image pyramid from the current reconstructed map
    FramePyramid& getModelFramePyramid();
    
    /// get the current reconstructed map
    WorldMap& getMap();
    
    int findMergedLabel(int label);
    
    /// set the reference camera pose
    void setReferencePose(Eigen::Matrix4f& referencePose);
    

protected:
    InSegSLAM* m_slam;
};

}

#endif
