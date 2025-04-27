//
// Created by xiang on 23-2-2.
//
#include "measure_sync.h"

namespace sad {

bool MessageSync::Sync() {
    if (lidar_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }


    //lidar_pushed_ 是一个标志位，用于判断是否已经将激光雷达数据放入 measures_ 中
    if (!lidar_pushed_) {
        measures_.lidar_ = lidar_buffer_.front();
        measures_.lidar_begin_time_ = time_buffer_.front();

        lidar_end_time_ = measures_.lidar_begin_time_ + measures_.lidar_->points.back().time / double(1000);

        measures_.lidar_end_time_ = lidar_end_time_;
        lidar_pushed_ = true;
    }

    //如果最新的 IMU 数据时间戳 last_timestamp_imu_ 小于激光雷达数据的结束时间 ,
    //lidar_end_time_，表示 IMU 数据不足以覆盖激光雷达数据的时间范围，直接返回 false。
    if (last_timestamp_imu_ < lidar_end_time_) {
        return false;
    }

    double imu_time = imu_buffer_.front()->timestamp_;
    measures_.imu_.clear();
    while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_)) {
        imu_time = imu_buffer_.front()->timestamp_;
        if (imu_time > lidar_end_time_) {
            break;
        }
        measures_.imu_.push_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }

    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
    lidar_pushed_ = false;

    if (callback_) {
        callback_(measures_);
    }

    return true;
}

void MessageSync::Init(const std::string& yaml) { conv_->LoadFromYAML(yaml); }

}  // namespace sad