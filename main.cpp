//
// Created by dong on 04.08.17.
//
/*
 * This script is used to generate pointcloud for each frame and concatenate in world coordinate
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

#include <Eigen/Core>
#include <Eigen/Geometry>

typedef pcl::PointXYZRGBA PointType;

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

void LoadImages(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD, std::vector<double> &vTimestamps, std::vector<campos> &vCampos);
void LoadParams(const std::string &fsSettings, campam &camInt);
Eigen::Matrix4f quaternion2matrix(campos &impos);
void generatePointCloud(cv::Mat& rgb, cv::Mat& dep, campam& camera, pcl::PointCloud<PointType>::Ptr& cloud);

int main() {
    std::string fsSettings   = "/home/dong/Documents/3D_Matching/3DV/orb_slam/Examples/RGB-D/TUM1.yaml";
    std::string sequence     = "/home/dong/Documents/3D_Matching/Dataset/TUM/rgbd_dataset_freiburg1_desk";
    std::string association  = "/home/dong/Documents/3D_Matching/Dataset/TUM/rgbd_dataset_freiburg1_desk/fr1_desk.txt";

    // Retrieve paths to images
    std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<std::string> vstrImageFilenamesD;
    std::vector<double> vTimestamps;
    std::vector<campos> vCampos;
    LoadImages(association, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps, vCampos);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }
    campam camInt;
    LoadParams(fsSettings, camInt);

    bool enGeneratePointCloud = 1;
    bool enTestCorrespondence = 1;
    pcl::visualization::PCLVisualizer viewer("Viewer Clouds");


    for (size_t i = 0; i < 20; i++)
    {
        std::cout<<"Frame--"<<i<<std::endl;
        cv::Mat rgb1 = cv::imread(sequence+"/"+vstrImageFilenamesRGB[i],CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat dep1 = cv::imread(sequence+"/"+vstrImageFilenamesD[i],CV_LOAD_IMAGE_UNCHANGED);
        campos impos = vCampos[i];

        pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType> ());
        generatePointCloud(rgb1, dep1, camInt, cloud);
        pcl::PointCloud<PointType>::Ptr off_scene_scene (new pcl::PointCloud<PointType> ());
        Eigen::Matrix4f Twc = quaternion2matrix(impos);
        pcl::transformPointCloud(*cloud, *off_scene_scene, Twc);
        viewer.addPointCloud(off_scene_scene, "scene_"+std::to_string(i));
    }

//    int first = 1, second = 3, third = 5;
//    cv::Mat rgb1 = cv::imread(sequence+"/"+vstrImageFilenamesRGB[first],CV_LOAD_IMAGE_UNCHANGED);
//    cv::Mat dep1 = cv::imread(sequence+"/"+vstrImageFilenamesD[first],CV_LOAD_IMAGE_UNCHANGED);
//    campos impos1 = vCampos[first];
//    cv::Mat rgb2 = cv::imread(sequence+"/"+vstrImageFilenamesRGB[second],CV_LOAD_IMAGE_UNCHANGED);
//    cv::Mat dep2 = cv::imread(sequence+"/"+vstrImageFilenamesD[second],CV_LOAD_IMAGE_UNCHANGED);
//    campos impos2 = vCampos[second];
////    cv::Mat rgb3 = cv::imread(sequence+"/"+vstrImageFilenamesRGB[third],CV_LOAD_IMAGE_UNCHANGED);
////    cv::Mat dep3 = cv::imread(sequence+"/"+vstrImageFilenamesD[third],CV_LOAD_IMAGE_UNCHANGED);
////    campos impos3 = vCampos[third];
//
//    pcl::PointCloud<PointType>::Ptr cloud1 (new pcl::PointCloud<PointType> ());
//    pcl::PointCloud<PointType>::Ptr cloud2 (new pcl::PointCloud<PointType> ());
//    generatePointCloud(rgb1, dep1, camInt, cloud1);
//    generatePointCloud(rgb2, dep2, camInt, cloud2);
//
//    Eigen::Matrix4f Twc_1 = quaternion2matrix(impos1);
//    Eigen::Matrix4f Twc_2 = quaternion2matrix(impos2);
//    Eigen::Matrix4f transMat = Twc_2.inverse()*Twc_1;
//
//    pcl::visualization::PCLVisualizer viewer("Viewer Clouds");
//    pcl::PointCloud<PointType>::Ptr off_scene_scene1 (new pcl::PointCloud<PointType> ());
////    pcl::transformPointCloud(*cloud1, *off_scene_scene1, t1, q1);
//    pcl::transformPointCloud(*cloud1, *off_scene_scene1, Twc_1);
//    viewer.addPointCloud(off_scene_scene1, "scene1");
//
//    pcl::PointCloud<PointType>::Ptr off_scene_scene2 (new pcl::PointCloud<PointType> ());
////    pcl::transformPointCloud(*cloud2, *off_scene_scene2, t2, q2);
//    pcl::transformPointCloud(*cloud2, *off_scene_scene2, Twc_2);
//    viewer.addPointCloud(off_scene_scene2, "scene2");
    while(!viewer.wasStopped())
    {
        viewer.spinOnce();
    }

    return 0;
}

void LoadImages(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD, std::vector<double> &vTimestamps, std::vector<campos> &vCampos)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
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
    dep.convertTo(dep,CV_32F,1.0f/camera.s);
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