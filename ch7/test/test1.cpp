#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h> 
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
/**
 * @description: PCL RANSAC 分割地面点云
 * @param {Ptr} cloud 点云
 * @return {*}
 */
void ExtractGroundPoints_RANSAC(PointCloudType::Ptr cloud) {
    //创建分割时所需要的模型系数对象coefficients及存储内点的点索引集合对象inliers。
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    // 创建分割对象
    pcl::SACSegmentation<pcl::PointXYZ> seg;    // 可选择配置，设置模型系数需要优化
    seg.setOptimizeCoefficients(true);          // 设置对估计的模型参数进行优化处理
    seg.setModelType(pcl::SACMODEL_PLANE);      // 设置分割的模型类型为平面
    seg.setMethodType(pcl::SAC_RANSAC);         // 设置用ransac随机参数估计方法
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.15);             // 平面厚度距离阈值，表示距离平面多少米之内的点都算平面内点inlier。
    seg.setInputCloud(cloud);                   // 输入点云
    seg.segment(*inliers, *coefficients);       // 实现分割，并存储分割结果到点集合inliers及存储平面模型系数coefficients

    PointCloudType::Ptr cloud_ground(new PointCloudType),       // 地面点云
                        cloud_no_ground(new PointCloudType);    // 非地面点云
                        
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);  // true表示剔除地面点，设置提取器的行为为“负向提取”（true）
    // 当 setNegative(true) 时，提取器会保留那些不在 inliers 中的点，即剔除掉被标记为内点的点（例如地面点）
    extract.filter(*cloud_no_ground);
    extract.setNegative(false); // false表示保留平面内点，设置提取器的行为为“正向提取”（false）
    // 当 setNegative(false) 时，提取器会保留那些在 inliers 中的点，即只保留被标记为内点的点（例如地面点）
    extract.filter(*cloud_ground);

    pcl::io::savePCDFileBinaryCompressed("./data/ch5/scan_example_ground.pcd", *cloud_ground);
    pcl::io::savePCDFileBinaryCompressed("./data/ch5/scan_example_no_ground.pcd", *cloud_no_ground);
}