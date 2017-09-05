//
// Created by dong on 31.08.17.
//

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
//#include "halton.hpp"

typedef pcl::PointXYZRGBA PointType;
int ps = 64;
int kp_num = 500;

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
    std::string fsSettings   = "/home/dong/Documents/3D_Matching/3DV/orb_slam/Examples/RGB-D/TUM1.yaml";
    std::string sequence     = "/home/dong/Documents/3D_Matching/Dataset/TUM/rgbd_dataset_freiburg1_desk";
    std::string association  = "/home/dong/Documents/3D_Matching/Dataset/TUM/rgbd_dataset_freiburg1_desk/fr1_desk.txt";
    std::string outputFilename   = "../map.ply";

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
    InSeg::CamParams cp;
    LoadCamParams(fsSettings, cp);
    InSeg::INSEG_CONFIG.setCamParams(cp);
    InSeg::INSEG_CONFIG.bUseReferencePose = true;

    bool enGeneratePointCloud = 1;
    bool enTestCorrespondence = 1;
//    pcl::visualization::PCLVisualizer viewer("Viewer Clouds");
    InSeg::InSegLib slam;
    std::vector<frameInfo> frames;
    int num_frame = 2000;
    for (size_t i = 0; i < std::min(num_frame, (int)vCampos.size()); i++)
    {
        std::cout<<"Frame--"<<i<<std::endl;
        cv::Mat bgr1 = cv::imread(sequence+"/"+vstrImageFilenamesRGB[i],CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat dep1 = cv::imread(sequence+"/"+vstrImageFilenamesD[i],CV_LOAD_IMAGE_UNCHANGED)/5.;
        campos impos1 = vCampos[i];
        Eigen::Matrix4f Twc = quaternion2matrix(impos1);
        Twc(0,3) = Twc(0,3)*1000;
        Twc(1,3) = Twc(1,3)*1000;
        Twc(2,3) = Twc(2,3)*1000;

        Eigen::Matrix4f Tcw = Twc.inverse();
//        Tcw(0,3) = Tcw(0,3)*1000;
//        Tcw(1,3) = Tcw(1,3)*1000;
//        Tcw(2,3) = Tcw(2,3)*1000;
        slam.setReferencePose(Tcw);
        slam.processFrame(dep1, bgr1);
        static bool isFirstFrame = true;
        if(isFirstFrame){
            // initialize map
            slam.initializeMap();
            isFirstFrame = false;
        }
        // print current pose
//        std::cout << "gt: " << Twc << std::endl;
        std::cout << slam.getCurrentPose() << std::endl;

        /*
        cv::Mat1i mRowI(1, kp_num), mColI(1, kp_num);
        cv::randu(mRowI, cv::Scalar(patch_size), cv::Scalar(bgr1.rows-patch_size));
        cv::randu(mColI, cv::Scalar(patch_size), cv::Scalar(bgr1.cols-patch_size));

        for (int j = 0; jTwc < kp_num; j++)
        {
            cv::circle(bgr1, cv::Point(mColI.at<int>(0,j), (mRowI.at<int>(0,j))), 2, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
        cv::imshow("test", bgr1);
        cv::waitKey(0);

        pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType> ());
        generatePointCloud(bgr1, dep1, camInt, cloud);
        pcl::PointCloud<PointType>::Ptr off_scene_scene (new pcl::PointCloud<PointType> ());
//        Eigen::Matrix4f Twc = quaternion2matrix(impos1);
        pcl::transformPointCloud(*cloud, *off_scene_scene, Twc);
        viewer.addPointCloud(off_scene_scene, "scene_"+std::to_string(i));
        */
        frameInfo frame;
//        frame.bgr = bgr1.clone();
//        frame.dep = dep1.clone();
        bgr1.copyTo(frame.bgr);
        dep1.copyTo(frame.dep);
        frame.Tcw = Tcw;
        frames.push_back(frame);

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
    std::vector<std::string> str;
    boost::split(str, sequence, boost::is_any_of("/"));
    std::string savePath = "../Data/"+str[str.size()-1];
    boost::filesystem::remove_all(savePath);
    boost::filesystem::path dir(savePath);
    boost::filesystem::create_directory(dir);

    ofstream info(savePath + "/info.txt", ios::out);
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
    std::vector<InSeg::Surfel>& surfels = slam.getMap().getSurfels();
    std::random_shuffle(surfels.begin(), surfels.end());
    for (std::vector<InSeg::Surfel>::iterator it = surfels.begin(); it < std::min(surfels.end(), surfels.begin()+num_mappoints); ++it) {
//        std::cout<<"Point "<<it - surfels.begin()<<std::endl;
        Eigen::Vector3f x3Dw = it->pos;
        int kn=0;
        std::vector<cv::Mat> pointPatch_GRAY;
        std::vector<cv::Mat> pointPatch_RGB;
        std::vector<cv::Mat> pointPatch_DEP;
        std::vector<cv::Point2i> points;

        // project to each frame
        for (int i = 0; i < frames.size(); ++i) {
            Eigen::Matrix4f Tcw = frames[i].Tcw;
            Eigen::Matrix3f Rcw = Tcw.block(0,0,3,3);
            Eigen::Vector3f tcw = Tcw.block(0,3,3,1);
            Eigen::Vector3f x3Dc = Rcw*x3Dw + tcw;
            const float xc = x3Dc(0);
            const float yc = x3Dc(1);
            const float zc = x3Dc(2);
            const float invzc = 1.0/x3Dc(2);
            if (invzc < 0)
                continue;
            auto v = int(camInt.fx*xc*invzc + camInt.cx);
            auto u = int(camInt.fy*yc*invzc + camInt.cy);
            // valid or not
            if (u - ps/2 > 0 && u + ps/2 < frames[i].bgr.rows && v - ps/2 > 0 && v + ps/2 < frames[i].bgr.cols)
            {
                points.push_back(cv::Point2i(v, u));
                if (frames[i].dep.at<ushort>(u, v) == 0 || cv::abs(frames[i].dep.at<ushort>(u,v) - zc) > Threshold)
                    continue;
                kn++;
                int vx1 = v - ps/2;
                int vy1 = u - ps/2;
                cv::Mat patch_rgb = frames[i].bgr(cv::Range(vy1, vy1+ps), cv::Range(vx1, vx1+ps));
                cv::Mat patch_dep = frames[i].dep(cv::Range(vy1, vy1+ps), cv::Range(vx1, vx1+ps));

                cv::Mat patch_gray;
                cv::cvtColor(patch_rgb, patch_gray, CV_RGB2GRAY);
                cv::Mat patch_g2rgb(patch_gray.size(), CV_8UC3);
                cv::cvtColor(patch_gray, patch_g2rgb, CV_GRAY2RGB);
                pointPatch_GRAY.push_back(patch_gray);
                pointPatch_RGB.push_back(patch_rgb);
                pointPatch_DEP.push_back(patch_dep);

//                cv::imshow("rgb", frames[i].bgr);
//                cv::waitKey(0);
            }
        }

        if (kn >= 2)    // if one map point has at least two valid keypoints
        {
            globalNum+=kn;

            cv::putText(pointPatch_GRAY[0],
                        std::to_string(mpNum),
                        cv::Point(5,20), // Coordinates
                        cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
                        1.0, // Scale. 2.0 = 2x bigger
                        cv::Scalar(255,255,255) // Color
            ); // Anti-alias

            for (int m = 0; m < kn; m++)
            {
                patches_64X64_GRAY.push_back(pointPatch_GRAY[m]);
                patches_64X64_RGB.push_back(pointPatch_RGB[m]);
                patches_64X64_DEP.push_back(pointPatch_DEP[m]);
                indexP_64X64.push_back(mpNum);
            }
//                cv::Mat comI;
//                cv::hconcat((pointPatch_RGB[0], pointPatch_RGB[1]), comI);
//                cv::imshow("rgb_dep", comI);
//                cv::waitKey(0);
//            cout<<"mpNum|mp_Id: "<<mpNum<<"|"<<mp->mnId<<endl;
            mpNum++;
        }
        if (patches_64X64_RGB.size() > 256)
        {
            cv::Mat combined_gray(1024, 1024, patches_64X64_GRAY[0].type());
            cv::Mat combined_rgb(1024, 1024, patches_64X64_RGB[0].type());
            cv::Mat combined_dep(1024, 1024, patches_64X64_DEP[0].type());
            int count = 0;
            for (int i = 0; i < 1024; i+=ps) {
                for (int j = 0; j < 1024; j+=ps) {
                    cv::Mat roi_gray = combined_gray(cv::Rect(j, i, ps, ps));
                    patches_64X64_GRAY[count].copyTo(roi_gray);
                    cv::Mat roi_rgb = combined_rgb(cv::Rect(j, i, ps, ps));
                    patches_64X64_RGB[count].copyTo(roi_rgb);
                    cv::Mat roi_dep = combined_dep(cv::Rect(j, i, ps, ps));
                    patches_64X64_DEP[count].copyTo(roi_dep);

                    info<<indexP_64X64[count]<<'\n';
                    count++;
                }
            }
            std::stringstream strStr;
            strStr<< setfill('0') << setw(4) << wallNum;
            std::string strTmp(strStr.str());
            std::string grayName = savePath + "/" + "patches" + strTmp + "_gray.png";
            cv::imwrite(grayName, combined_gray);
            std::string rgbName = savePath + "/" + "patches" + strTmp + "_rgb.png";
            cv::imwrite(rgbName, combined_rgb);
            std::string depName = savePath + "/" + "patches" + strTmp + "_dep.png";
            cv::imwrite(depName, combined_dep);

            wallNum++;
            patches_64X64_GRAY.erase(patches_64X64_GRAY.begin(), patches_64X64_GRAY.begin()+count);
            patches_64X64_RGB.erase(patches_64X64_RGB.begin(), patches_64X64_RGB.begin()+count);
            patches_64X64_DEP.erase(patches_64X64_DEP.begin(), patches_64X64_DEP.begin()+count);
            indexP_64X64.erase(indexP_64X64.begin(), indexP_64X64.begin()+count);
            cout<<"wallNum: "<<wallNum<<endl;
        }
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
//    pcl::visualization::PCLVisualizer vieTwc_inv(0,3) = Twc_inv(0,3)/1000;wer("Viewer Clouds");
//    pcl::PointCloud<PointType>::Ptr off_scene_scene1 (new pcl::PointCloud<PointType> ());
////    pcl::transformPointCloud(*cloud1, *off_scene_scene1, t1, q1);
//    pcl::transformPointCloud(*cloud1, *off_scene_scene1, Twc_1);
//    viewer.addPointCloud(off_scene_scene1, "scene1");
//
//    pcl::PointCloud<PointType>::Ptr off_scene_scene2 (new pcl::PointCloud<PointType> ());
////    pcl::transformPointCloud(*cloud2, *off_scene_scene2, t2, q2);
//    pcl::transformPointCloud(*cloud2, *off_scene_scene2, Twc_2);
//    viewer.addPointCloud(off_scene_scene2, "scene2");


//    while(!viewer.wasStopped())
//    {
//        viewer.spinOnce();
//    }

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