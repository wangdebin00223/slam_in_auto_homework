//
// Created by xiang on 2022/3/23.
//

#include "ch6/occupancy_map.h"
#include "common/eigen_types.h"
#include "common/math_utils.h"

#include <glog/logging.h>
#include <execution>

namespace sad {

OccupancyMap::OccupancyMap() {
    BuildModel();
    occupancy_grid_ = cv::Mat(image_size_, image_size_, CV_8U, 127);
}

void OccupancyMap::BuildModel() {
    for (int x = -model_size_; x <= model_size_; x++) {
        for (int y = -model_size_; y <= model_size_; y++) {
            Model2DPoint pt;
            pt.dx_ = x;
            pt.dy_ = y;
            pt.range_ = sqrt(x * x + y * y) * inv_resolution_;
            //pt.range_：点 (x, y) 的距离，转换为栅格地图中的分辨率单位
            pt.angle_ = std::atan2(y, x);
            pt.angle_ = pt.angle_ > M_PI ? pt.angle_ - 2 * M_PI : pt.angle_;  // limit in 2pi
            //如果 pt.angle_ 大于 π，减去 2π，使其落在 [-π, π] 范围内
            model_.push_back(pt);
        }
    }
}

/// 查找某个角度下的range值
//每个采样点对应一个距离值，存储在 scan->ranges
double OccupancyMap::FindRangeInAngle(double angle, Scan2d::Ptr scan) {
    math::KeepAngleInPI(angle);
    if (angle < scan->angle_min || angle > scan->angle_max) {
        return 0.0;
    }

    int angle_index = int((angle - scan->angle_min) / scan->angle_increment);

    if (angle_index < 0 || angle_index >= scan->ranges.size()) {
        return 0.0;
    }

    int angle_index_p = angle_index + 1;
    double real_angle = angle;

    // take range
    double range = 0;
    if (angle_index_p >= scan->ranges.size()) {
        range = scan->ranges[angle_index];
    } else {
        // 插值
        double s = ((angle - scan->angle_min) / scan->angle_increment) - angle_index;
        //s：插值系数，表示 angle 在 angle_index 和 angle_index_p 之间的比例
        //(一个是整数（计算的索引），另一个是浮点数)
        double range1 = scan->ranges[angle_index];
        double range2 = scan->ranges[angle_index_p];

        double real_angle1 = scan->angle_min + scan->angle_increment * angle_index;
        double real_angle2 = scan->angle_min + scan->angle_increment * angle_index_p;

        if (range2 < scan->range_min || range2 > scan->range_max) {
            range = range1;
            real_angle = real_angle1;
        } else if (range1 < scan->range_min || range1 > scan->range_max) {
            range = range2;
            real_angle = real_angle2;
        } else if (std::fabs(range1 - range2) > 0.3) {
            //如果 range1 和 range2 的差值大于 0.3，则根据插值系数 s 选择较近的距离值
            range = s > 0.5 ? range2 : range1;
            real_angle = s > 0.5 ? real_angle2 : real_angle1;
        } else {
            //否则，使用线性插值计算距离值
            range = range1 * (1 - s) + range2 * s;
        }
    }
    
    return range;
}


/// 往这个占据栅格地图中增加一个frame
void OccupancyMap::AddLidarFrame(std::shared_ptr<Frame> frame, GridMethod method) {
    auto& scan = frame->scan_;
    
    // 此处不能直接使用frame->pose_submap_，因为frame可能来自上一个地图
    // 此时frame->pose_submap_还未更新，依旧是frame在上一个地图中的pose
    SE2 pose_in_submap = pose_.inverse() * frame->pose_;
    float theta = pose_in_submap.so2().log();
    has_outside_pts_ = false;

    // 先计算末端点所在的网格
    std::set<Vec2i, less_vec<2>> endpoints;

    for (size_t i = 0; i < scan->ranges.size(); ++i) {
        if (scan->ranges[i] < scan->range_min || scan->ranges[i] > scan->range_max) {
            continue;
        }
        double real_angle = scan->angle_min + i * scan->angle_increment;
        double x = scan->ranges[i] * std::cos(real_angle);
        double y = scan->ranges[i] * std::sin(real_angle);
        endpoints.emplace(World2Image(frame->pose_ * Vec2d(x, y)));
    }

    if (method == GridMethod::MODEL_POINTS) {
        // 遍历模板，生成白色点
        std::for_each(std::execution::par_unseq, model_.begin(), model_.end(), [&](const Model2DPoint& pt) {
            Vec2i pos_in_image = World2Image(frame->pose_.translation());
            Vec2i pw = pos_in_image + Vec2i(pt.dx_, pt.dy_);  // submap下
            //确定模板点在子地图的坐标
            if (pt.range_ < closest_th_) {
                // 小距离内认为无物体
                SetPoint(pw, false);
                return;
            }

            double angle = pt.angle_ - theta;  
            //angle = pt.angle_ - theta，这是模板点的角度减去frame的旋转角度


            double range = FindRangeInAngle(angle, scan);
            //range 表示的是雷达在 angle 方向上检测到的障碍物的距离。

            if (range < scan->range_min || range > scan->range_max) {
                /// 某方向无测量值时，认为无效
                /// 但离机器比较近时，涂白
                if (pt.range_ < endpoint_close_th_) {
                    SetPoint(pw, false);
                }
                return;
            }
            //endpoints.find(pw) == endpoints.end()：表示模板点 pw 
            //不在激光雷达的末端点集合中，说明该点不是障碍物的末端点。
            if (range > pt.range_ && endpoints.find(pw) == endpoints.end()) {
                /// 末端点与车体连线上的点，涂白
                SetPoint(pw, false);
            }
        });
    } else {
        Vec2i start = World2Image(frame->pose_.translation());
        std::for_each(std::execution::par_unseq, endpoints.begin(), endpoints.end(),
                      [this, &start](const auto& pt) { BresenhamFilling(start, pt); });
    }

    /// 末端点涂黑
    std::for_each(endpoints.begin(), endpoints.end(), [this](const auto& pt) { SetPoint(pt, true); });
}

/// 设置中心点
void OccupancyMap::SetPoint(const Vec2i& pt, bool occupy) {
    int x = pt[0], y = pt[1];
    if (x < 0 || y < 0 || x >= occupancy_grid_.cols || y >= occupancy_grid_.rows) {
        if (occupy) {
            has_outside_pts_ = true;
        }

        return;
    }

    /// 这里设置了一个上下限
    uchar value = occupancy_grid_.at<uchar>(y, x);
    if (occupy) {
        if (value > 117) {
            occupancy_grid_.ptr<uchar>(y)[x] -= 1;
        }
    } else {
        if (value < 137) {
            occupancy_grid_.ptr<uchar>(y)[x] += 1;
        }
    }
}

/// 获取黑白灰形式的占据栅格，作可视化使用
cv::Mat OccupancyMap::GetOccupancyGridBlackWhite() const {
    cv::Mat image(image_size_, image_size_, CV_8UC3);
    for (int x = 0; x < occupancy_grid_.cols; ++x) {
        for (int y = 0; y < occupancy_grid_.rows; ++y) {
            if (occupancy_grid_.at<uchar>(y, x) == 127) {
                image.at<cv::Vec3b>(y, x) = cv::Vec3b(127, 127, 127);
            } else if (occupancy_grid_.at<uchar>(y, x) < 127) {
                image.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            } else if (occupancy_grid_.at<uchar>(y, x) > 127) {
                image.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
            }
        }
    }

    return image;
}

//Bresenham直线填充，给定起始点和终止点，将中间的区域填充为白色
void OccupancyMap::BresenhamFilling(const Vec2i& p1, const Vec2i& p2) {
    int dx = p2.x() - p1.x();
    int dy = p2.y() - p1.y();
    int ux = dx > 0 ? 1 : -1;
    int uy = dy > 0 ? 1 : -1;

    dx = abs(dx);
    dy = abs(dy);
    int x = p1.x();
    int y = p1.y();

    if (dx > dy) {
        // 以x为增量
        int e = -dx;
        for (int i = 0; i < dx; ++i) {
            x += ux;
            e += 2 * dy;
            if (e >= 0) {
                y += uy;
                e -= 2 * dx;
            }

            if (Vec2i(x, y) != p2) {
                SetPoint(Vec2i(x, y), false);
            }
        }
    } else {
        int e = -dy;
        for (int i = 0; i < dy; ++i) {
            y += uy;
            e += 2 * dx;
            if (e >= 0) {
                x += ux;
                e -= 2 * dy;
            }
            if (Vec2i(x, y) != p2) {
                SetPoint(Vec2i(x, y), false);
            }
        }
    }
}

}  // namespace sad