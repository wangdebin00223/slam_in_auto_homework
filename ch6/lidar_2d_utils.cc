//
// Created by xiang on 2022/3/15.
//

#include "ch6/lidar_2d_utils.h"
#include <opencv2/imgproc.hpp>

namespace sad {
//pose_submap: 子地图的位姿（SE2 类型）
void Visualize2DScan(Scan2d::Ptr scan, const SE2& pose, cv::Mat& image, const Vec3b& color, int image_size,
                     float resolution, const SE2& pose_submap) {
    if (image.data == nullptr) {
        image = cv::Mat(image_size, image_size, CV_8UC3, cv::Vec3b(255, 255, 255));
    }

    for (size_t i = 0; i < scan->ranges.size(); ++i) {
        if (scan->ranges[i] < scan->range_min || scan->ranges[i] > scan->range_max) {
            continue;
        }

        double real_angle = scan->angle_min + i * scan->angle_increment;
        double x = scan->ranges[i] * std::cos(real_angle);
        double y = scan->ranges[i] * std::sin(real_angle);
        //将激光雷达的极坐标（距离和角度）转换为二维笛卡尔坐标（x 和 y）

        if (real_angle < scan->angle_min + 30 * M_PI / 180.0 || real_angle > scan->angle_max - 30 * M_PI / 180.0) {
            continue;
        }
        //过滤掉角度范围在 [angle_min + 30°, angle_max - 30°] 之外的点

        Vec2d psubmap = pose_submap.inverse() * (pose * Vec2d(x, y));
        //将当前点的坐标从机器人坐标系转换到子地图坐标系：
        //pose * Vec2d(x, y)：将点从机器人坐标系转换到世界坐标系。
        //pose_submap.inverse()：将点从世界坐标系转换到子地图坐标系。

        int image_x = int(psubmap[0] * resolution + image_size / 2);
        int image_y = int(psubmap[1] * resolution + image_size / 2);
        //将子地图坐标系中的点映射到图像坐标：
        //psubmap[0] * resolution：将点的 x 坐标从米转换为像素。
        //+ image_size / 2：将图像中心作为坐标系原点。

        if (image_x >= 0 && image_x < image.cols && image_y >= 0 && image_y < image.rows) {
            image.at<cv::Vec3b>(image_y, image_x) = cv::Vec3b(color[0], color[1], color[2]);
        }
    }
    // 同时画出pose自身所在位置
    Vec2d pose_in_image =
        pose_submap.inverse() * (pose.translation()) * double(resolution) + Vec2d(image_size / 2, image_size / 2);
    cv::circle(image, cv::Point2f(pose_in_image[0], pose_in_image[1]), 5, cv::Scalar(color[0], color[1], color[2]), 2);
}

}  // namespace sad