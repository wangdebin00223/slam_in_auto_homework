#include <iostream>
#include <fstream>
#include <memory>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h> 
#include <pcl/segmentation/impl/extract_clusters.hpp> 
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;

using PointType = pcl::PointXYZ;
using PointCloudType = pcl::PointCloud<PointType>;

DEFINE_string(pcd_path, "./data/target_cloud.pcd", "点云文件路径"); // scan_example.pcd  target_cloud.pcd

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_pcd_path.empty()) {
        LOG(ERROR) << "pcd path is empty";
        return -1;
    }
    // 读取点云
    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>);
    pcl::io::loadPCDFile(FLAGS_pcd_path, *cloud);

    LOG(INFO) << "cloud points: " << cloud->size();

    if (cloud->empty()) {
        LOG(ERROR) << "cannot load cloud file";
        return -1;
    }

    pcl::StatisticalOutlierRemoval<PointType> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(30);  // 对每个点分析的邻近点个数
    sor.setStddevMulThresh(1.0);  // 标准差倍数
    sor.filter(*cloud_filtered);

    // 显示剔除离群点后的点云
    pcl::visualization::CloudViewer viewer2("after filtered");
	viewer2.showCloud(cloud_filtered, "cloud filtered0");
    while (!viewer2.wasStopped()) 
        boost::this_thread::sleep(boost::posix_time::microseconds(10000));

    pcl::search::KdTree<PointType>::Ptr kdtree_; // (new pcl::search::KdTree<PointType>)
    kdtree_ = boost::make_shared<pcl::search::KdTree<PointType>>();
    kdtree_->setInputCloud(cloud_filtered);      // 设置输入构建目标点云的KD树

    pcl::EuclideanClusterExtraction<PointType> clusterExtractor_;
    // 创建一个向量来存储聚类的结果
    std::vector<pcl::PointIndices> cluster_indices;
    clusterExtractor_.setClusterTolerance(0.02);        // 设置聚类的距离阈值
    clusterExtractor_.setMinClusterSize(10);            // 设置聚类的最小点数
    clusterExtractor_.setMaxClusterSize(1000);          // 设置聚类的最大点数 25000
    clusterExtractor_.setSearchMethod(kdtree_);         // 使用kdtree树进行加速
    clusterExtractor_.setInputCloud(cloud_filtered);    // 设置点云聚类对象的输入点云数据
    clusterExtractor_.extract(cluster_indices);         // 执行点云聚类
    LOG(INFO) << "cluster size: " << cluster_indices.size();

    // 创建可视化对象
    pcl::visualization::PCLVisualizer viewer("Cluster Viewer");
    viewer.setBackgroundColor(0.0, 0.0, 0.0);
    viewer.addPointCloud<PointType>(cloud_filtered, "cloud");
    
    int clusterNumber = 1;  // 输出聚类结果
    for (const auto& indices : cluster_indices) {
        LOG(INFO) << "Cluster " << clusterNumber << " has " << indices.indices.size() << " points.";
        pcl::PointCloud<PointType>::Ptr cluster(new pcl::PointCloud<PointType>);
        pcl::copyPointCloud(*cloud_filtered, indices, *cluster);

        // 为聚类分配不同的颜色
        double r = static_cast<double>(rand()) / RAND_MAX;
        double g = static_cast<double>(rand()) / RAND_MAX;
        double b = static_cast<double>(rand()) / RAND_MAX;

        // 将聚类点云添加到可视化对象中
        std::string clusterId = "cluster_" + std::to_string(clusterNumber);
        viewer.addPointCloud<PointType>(cluster, clusterId);
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, clusterId);
        clusterNumber++;

    }

    while (!viewer.wasStopped()) 
        viewer.spinOnce();

    return 0;
}

std::vector<Eigen::Vector2d> linecoeffs_cloud;
for (const auto& indices : cluster_indices) {

    if(indices.indices.size() < 150) {
        LOG(INFO) << "Cluster " << clusterNumber << " has " << indices.indices.size() << " points." << "点数太少，不进行直线拟合";
        clusterNumber++;
        continue;
    } 
    
    LOG(INFO) << "Cluster " << clusterNumber << " has " << indices.indices.size() << " points.";

    pcl::PointCloud<PointType>::Ptr cluster(new pcl::PointCloud<PointType>);
    pcl::copyPointCloud(*cloud_filtered, indices, *cluster);

    // 将pcl::PointXYZ类型的点云转换为Eigen::Vector3d类型的点云
    std::vector<Eigen::Vector2d> pts;
    pts.reserve(cluster->size());

    for (const PointType& pt : *cluster)
        pts.emplace_back(Eigen::Vector2d(pt.x, pt.y));

    // 拟合直线，组装J、H和误差
    Eigen::Vector3d line_coeffs;
    // 利用当前点附近的几个有效近邻点，基于SVD奇异值分解，拟合出ax+by+c=0 中的最小直线系数 a,b,c，对应公式（6.11）
    if (FitLine2D(pts, line_coeffs)) {
        if(line_coeffs[0] < 0.0)
            line_coeffs = -line_coeffs;

        double slope = -line_coeffs[0] / line_coeffs[1];
        double angle = std::atan(slope);
        LOG(INFO) << "line_coeffs: " << line_coeffs[0] << ", " << line_coeffs[1] << ", " << line_coeffs[2];
        LOG(INFO) << "slope: " << slope;
        LOG(INFO) << "angle: " << angle;
        
        linecoeffs_cloud.emplace_back(line_coeffs.head<2>());
    }

    // 为聚类分配不同的颜色
    double r = static_cast<double>(rand()) / RAND_MAX;
    double g = static_cast<double>(rand()) / RAND_MAX;
    double b = static_cast<double>(rand()) / RAND_MAX;

    // 将聚类点云添加到可视化对象中
    std::string clusterId = "cluster_" + std::to_string(clusterNumber);
    viewer.addPointCloud<PointType>(cluster, clusterId);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, clusterId);
    clusterNumber++;
}