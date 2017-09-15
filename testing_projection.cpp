//
// Created by dong on 14.09.17.
//

/*
 * Get the fusion result, project map point to each frame and testing the accuracy
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <dirent.h>
#include <algorithm>
#include <boost/algorithm/string.hpp>

#include <pcl/PCLPointCloud2.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

//#include <Eigen/Core>
//#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "InSegLib.h"
#include "InSegConfig.h"

typedef pcl::PointXYZRGBA PointType;
int ps = 64;
int kp_num = 500;
float depthUncertaintyCoef = 0.0000285f;

struct campos{
    float tx;
    float ty;
    float tz;

    float qx, qy, qz, qw;
};

struct campam{
    float cx;
    float cy;
    float fx;
    float fy;
    float s;
};

struct frameInfo{
    cv::Mat bgr;
    cv::Mat dep;
    Eigen::Matrix4f Tcw;
};

void LoadImages(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD, std::vector<double> &vTimestamps, std::vector<campos> &vCampos);
void LoadParams(const std::string &fsSettings, campam &camInt);
void LoadCamParams(const std::string &fsSettings, InSeg::CamParams &camInt);
Eigen::Matrix4f quaternion2matrix(campos &impos);
void generatePointCloud(cv::Mat& rgb, cv::Mat& dep, campam& camera, pcl::PointCloud<PointType>::Ptr& cloud);

int main() {
    std::string fsSettings = "/home/dong/Documents/3D_Matching/3DV/orb_slam/Examples/RGB-D/TUM3.yaml";
    std::string sequence = "/home/dong/Documents/3D_Matching/Dataset/TUM/rgbd_dataset_freiburg3_long_office_household";
    std::string association = "/home/dong/Documents/3D_Matching/Dataset/TUM/rgbd_dataset_freiburg3_long_office_household/fr3_long_office_household.txt";
    std::string outputFilename = "../map.ply";

    // Retrieve paths to images
    std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<std::string> vstrImageFilenamesD;
    std::vector<double> vTimestamps;
    std::vector<campos> vCampos;
    LoadImages(association, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps, vCampos);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if (vstrImageFilenamesRGB.empty()) {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    } else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size()) {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }
    campam camInt;
    LoadParams(fsSettings, camInt);
    InSeg::CamParams cp;
    LoadCamParams(fsSettings, cp);
    InSeg::INSEG_CONFIG.setCamParams(cp);
    InSeg::INSEG_CONFIG.bUseReferencePose = true;

    bool enGeneratePointCloud = 1;
    bool enTestCorrespondence = 1;
//    pcl::visualization::PCLVisualizer viewer("Viewer Clouds");
    InSeg::InSegLib slam;
    std::vector<frameInfo> frames;
    int num_frame = 10;
    for (size_t i = 0; i < std::min(num_frame, (int) vCampos.size()); i++) {
        std::cout << "Frame--" << i << std::endl;
        cv::Mat bgr1 = cv::imread(sequence + "/" + vstrImageFilenamesRGB[i], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat dep1 = cv::imread(sequence + "/" + vstrImageFilenamesD[i], CV_LOAD_IMAGE_UNCHANGED);
        campos impos1 = vCampos[i];
        Eigen::Matrix4f Twc = quaternion2matrix(impos1);
        Twc(0, 3) = Twc(0, 3) * 1000;   // to convert the translation to milimeter
        Twc(1, 3) = Twc(1, 3) * 1000;
        Twc(2, 3) = Twc(2, 3) * 1000;

        Eigen::Matrix4f Tcw = Twc.inverse();

        frameInfo frame;
        bgr1.copyTo(frame.bgr);
        dep1.copyTo(frame.dep);
        frame.Tcw = Tcw;
        frames.push_back(frame);

//        Tcw(0,3) = Tcw(0,3)*1000;   // to convert the translation to milimeter
//        Tcw(1,3) = Tcw(1,3)*1000;
//        Tcw(2,3) = Tcw(2,3)*1000;
        slam.setReferencePose(Tcw);
        cv::Mat dep1_5 = dep1.clone() / 5.;
        slam.processFrame(dep1_5, bgr1);
        static bool isFirstFrame = true;
        if (isFirstFrame) {
            // initialize map
            slam.initializeMap();
            isFirstFrame = false;
        }

        /*
        InSeg::WorldMap map = slam.getMap();
        std::vector<InSeg::Surfel>& surfels = map.getSurfels();
        Eigen::Matrix4f Tcw_inv = Tcw.inverse();
        Eigen::Matrix3f Rcw = Tcw.block(0,0,3,3);
        Eigen::Vector3f tcw = Tcw.block(0,3,3,1);

        for (std::vector<InSeg::Surfel>::iterator it = surfels.begin(); it < surfels.end(); ++it) {
            Eigen::Vector3f x3Dw = it->pos;    //millimeter
            Eigen::Vector3f x3Dc = Rcw*x3Dw + tcw;
            const float xc = x3Dc(0);
            const float yc = x3Dc(1);
            const float zc = x3Dc(2);
            const float invzc = 1.0/zc;
//            if (invzc < 0)
//                continue;
            float u = camInt.fx*xc*invzc + camInt.cx;
            float v = camInt.fy*yc*invzc + camInt.cy;
            if (u - ps/2 > 0 && u + ps/2 < bgr1.rows && v - ps/2 > 0 && v + ps/2 < bgr1.cols)
            {
                float diff = cv::abs(dep1.at<ushort>(int(u), int(v)) - zc);
                cout<<"diff: "<<diff<<endl;
            }
        }
        */
//        slam.getMap().saveModel(outputFilename.c_str());
    }
    slam.getMap().saveModel(outputFilename.c_str());

    //------------------- project to image plane --------------------//
    InSeg::WorldMap map = slam.getMap();
//    std::vector<InSeg::Surfel>& surfels = map.getSurfels();
    int Threshold = 2;
    int num_mappoints = 50000;
    int globalNum = 0;
    int mpNum = 0;
    int wallNum = 0;
    std::vector<cv::Mat> patches_64X64_GRAY;
    std::vector<cv::Mat> patches_64X64_RGB;
    std::vector<cv::Mat> patches_64X64_DEP;
    std::vector<int> indexP_64X64;
    std::vector<InSeg::Surfel> &surfels = slam.getMap().getSurfels();
    std::random_shuffle(surfels.begin(), surfels.end());

    while(1) {
        // randomly select a map point
        int mp_idx = rand() % static_cast<int>(surfels.size());
        Eigen::Vector3f x3Dw = surfels[mp_idx].pos;

//    std::vector<cv::Mat> pointPatch_GRAY;
        std::vector<cv::Mat> pointPatch_RGB;
        std::vector<cv::Mat> pointPatch_DEP;
        std::vector<int> frame_ids;
        std::vector<cv::Point2i> points;

        // project to each frame
        for (int i = 0; i < frames.size(); ++i) {
            Eigen::Matrix4f Tcw = frames[i].Tcw;
            Eigen::Matrix3f Rcw = Tcw.block(0, 0, 3, 3);
            Eigen::Vector3f tcw = Tcw.block(0, 3, 3, 1);
            Eigen::Vector3f x3Dc = Rcw * x3Dw + tcw;
            const float xc = x3Dc(0);
            const float yc = x3Dc(1);
            const float zc = x3Dc(2);    // cameraCoordinate(meter) X 1000, note!! not 5000
            const float invzc = 1.0 / zc;
            if (invzc < 0)
                continue;
            auto u = int(camInt.fx * xc * invzc + camInt.cx);
            auto v = int(camInt.fy * yc * invzc + camInt.cy);
            // valid or not
            if (v - ps / 2 > 0 && v + ps / 2 < frames[i].bgr.rows && u - ps / 2 > 0 && u + ps / 2 < frames[i].bgr.cols) {
                points.push_back(cv::Point2i(u, v));
                float errorThresh = depthUncertaintyCoef * frames[i].dep.at<ushort>(v, u) * frames[i].dep.at<ushort>(v, u);
                if (frames[i].dep.at<ushort>(v, u) == 0 || cv::abs(frames[i].dep.at<ushort>(v, u)/5. - zc) > Threshold)
                    continue;

                int vx1 = u - ps / 2;
                int vy1 = v - ps / 2;
                cv::Mat patch_rgb = frames[i].bgr(cv::Range(vy1, vy1 + ps), cv::Range(vx1, vx1 + ps));
                cv::Mat patch_dep = frames[i].dep(cv::Range(vy1, vy1 + ps), cv::Range(vx1, vx1 + ps));

//            cv::Mat patch_gray;
//            cv::cvtColor(patch_rgb, patch_gray, CV_RGB2GRAY);
//            cv::Mat patch_g2rgb(patch_gray.size(), CV_8UC3);
//            cv::cvtColor(patch_gray, patch_g2rgb, CV_GRAY2RGB);
//            pointPatch_GRAY.push_back(patch_gray);
                pointPatch_RGB.push_back(patch_rgb);
                pointPatch_DEP.push_back(patch_dep);
                frame_ids.push_back(i);

//                cv::imshow("rgb", frames[i].bgr);
//                cv::waitKey(0);
            }
        }

        if (pointPatch_RGB.size() > 2) {
            int num_patches = (int) pointPatch_RGB.size();
            int max_cmp = 10;
            int num_cmp = std::min(num_patches, max_cmp);
            // take the first one as an anchor, select at most max_cmp number of projected patches
            cv::Mat combined_project(ps, ps * num_cmp, pointPatch_RGB[0].type());
            cv::Mat combined_transform(ps, ps * num_cmp, pointPatch_RGB[0].type());

            int count = 0;
            Eigen::Matrix4f Tcw0 = frames[frame_ids[0]].Tcw;
            Tcw0(0, 3) = Tcw0(0, 3) / 1000;   // to convert the translation to milimeter
            Tcw0(1, 3) = Tcw0(1, 3) / 1000;
            Tcw0(2, 3) = Tcw0(2, 3) / 1000;
            cv::Point2i p0 = points[frame_ids[0]];
            double c1_z = double(frames[frame_ids[0]].dep.ptr<ushort>(p0.y)[p0.x]) / camInt.s;
            double c1_x = (p0.x - camInt.cx) * c1_z / camInt.fx;
            double c1_y = (p0.y - camInt.cy) * c1_z / camInt.fy;
            Eigen::Vector4f point0H(c1_x, c1_y, c1_z, 1);

            for (int i = 0; i < num_cmp*ps; i += ps) {
                cv::Mat roi_rgb = combined_project(cv::Rect(i, 0, ps, ps));
                pointPatch_RGB[count].copyTo(roi_rgb);

                // transformation projection
                Eigen::Matrix4f Tcwi = frames[frame_ids[count]].Tcw;
                // compute the position in camera2 coordinate
                Eigen::Vector4f pointiH = Tcwi * Tcw0.inverse() * point0H;
                // compute the corresponding position in camera2 image plane
                auto cx = int(pointiH(0) * camInt.fx / pointiH(2) + camInt.cx);
                auto cy = int(pointiH(1) * camInt.fy / pointiH(2) + camInt.cy);

                if (cy - ps / 2 > 0 && cy + ps / 2 < frames[frame_ids[count]].bgr.rows && cx - ps / 2 > 0 && cx + ps / 2 < frames[frame_ids[count]].bgr.cols) {
                    int vx1 = cx - ps / 2;
                    int vy1 = cy - ps / 2;
                    cv::Mat patch_rgb = frames[frame_ids[count]].bgr(cv::Range(vy1, vy1 + ps), cv::Range(vx1, vx1 + ps));
                    cv::Mat roi_rgb_t = combined_transform(cv::Rect(i, 0, ps, ps));
                    patch_rgb.copyTo(roi_rgb_t);
                }
                count++;
            }
            cv::Mat visualMat;
            cv::vconcat(combined_project, combined_transform, visualMat);
            cv::imshow("project", visualMat);
            cv::waitKey(0);
        }
    }
    return 0;
}

void LoadImages(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD, std::vector<double> &vTimestamps, std::vector<campos> &vCampos)
{

    std::ifstream fAssociation;
//    fAssociation.open(strAssociationFilename.c_str());
    fAssociation.open(strAssociationFilename);
    if (fAssociation.fail() == true)
        return;
    while(!fAssociation.eof())
    {
        std::string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            std::stringstream ss;
            ss << s;
            double t;
            std::string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
            ss >> t;
            campos p;
            ss >> p.tx, ss >> p.ty, ss >> p.tz;
            ss >> p.qx, ss >> p.qy, ss >> p.qz, ss >> p.qw;
            vCampos.push_back(p);

        }
    }
}

void LoadParams(const std::string &fsSettings, campam &camInt)
{
    //Check settings file
    cv::FileStorage settings(fsSettings, cv::FileStorage::READ);
    if(!settings.isOpened())
    {
        cerr << "Failed to open settings file at: " << fsSettings << endl;
        exit(-1);
    }
    camInt.fx = settings["Camera.fx"];
    camInt.fy = settings["Camera.fy"];
    camInt.cx = settings["Camera.cx"];
    camInt.cy = settings["Camera.cy"];
    camInt.s  = settings["DepthMapFactor"];
}

void LoadCamParams(const std::string &fsSettings, InSeg::CamParams &camInt)
{
    cv::FileStorage settings(fsSettings, cv::FileStorage::READ);
    if(!settings.isOpened())
    {
        cerr << "Failed to open settings file at: " << fsSettings << endl;
        exit(-1);
    }
    camInt.fx = settings["Camera.fx"];
    camInt.fy = settings["Camera.fy"];
    camInt.cx = settings["Camera.cx"];
    camInt.cy = settings["Camera.cy"];
    camInt.imgHeight = settings["Camera.height"];
    camInt.imgWidth = settings["Camera.width"];
}

Eigen::Matrix4f quaternion2matrix(campos &impos)
{
    Eigen::Quaternionf q(impos.qw, impos.qx, impos.qy, impos.qz);
    Eigen::Vector3f t(impos.tx, impos.ty, impos.tz);
    Eigen::Matrix3f rotation = q.toRotationMatrix();
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block(0,0,3,3) = rotation;
    T(0,3) = impos.tx; T(1,3) = impos.ty; T(2,3) = impos.tz;
//    cv::Mat vT;
//    cv::eigen2cv(T,vT);
    return T;
}

void generatePointCloud(cv::Mat& rgb, cv::Mat& dep, campam& camera, pcl::PointCloud<PointType>::Ptr& cloud)
{
//    pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType> ());
    dep.convertTo(dep,CV_32F,5.0f/camera.s);
    for (int i = 0; i < dep.rows; ++i) {
        for (int j = 0; j < dep.cols; ++j) {
            float d = dep.ptr<float>(i)[j];
            if (d == 0)
                continue;
            PointType p;
            p.z = d;
            p.x = (j - camera.cx)*p.z/camera.fx;
            p.y = (i - camera.cy)*p.z/camera.fy;
            p.b = rgb.ptr<uchar>(i)[j*3];
            p.g = rgb.ptr<uchar>(i)[j*3+1];
            p.r = rgb.ptr<uchar>(i)[j*3+2];
            cloud->points.push_back(p);
        }
    }
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;
}