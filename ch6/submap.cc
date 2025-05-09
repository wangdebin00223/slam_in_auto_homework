//
// Created by xiang on 2022/3/23.
//

#include "ch6/submap.h"
#include <glog/logging.h>

namespace sad {


 /// 把另一个submap中的占据栅格复制到本地图中
 //从另一个子地图（other）中提取最近的 10 帧激光雷达数据
 //并将其添加到当前子地图的占据地图（occu_map_）中，然后更新当前子地图的似然场（field_）
void Submap::SetOccuFromOtherSubmap(std::shared_ptr<Submap> other) {
    auto frames_in_other = other->GetFrames();
    // 取最近10个帧
    for (size_t i = frames_in_other.size() - 10; i < frames_in_other.size(); ++i) {
        if (i > 0) {
            occu_map_.AddLidarFrame(frames_in_other[i]);
        }
    }
    field_.SetFieldImageFromOccuMap(occu_map_.GetOccupancyGrid());
}

/// 将frame与本submap进行匹配，计算frame->pose
bool Submap::MatchScan(std::shared_ptr<Frame> frame) {
    field_.SetSourceScan(frame->scan_);
    field_.AlignG2O(frame->pose_submap_);
    frame->pose_ = pose_ * frame->pose_submap_;  // T_w_c = T_w_s * T_s_c
    //T_w_s：子地图在世界坐标系下的位姿。
    //T_s_c：当前帧在子地图坐标系下的位姿。
    //T_w_c：当前帧在世界坐标系下的位姿。
    return true;
}

/// 在栅格地图中增加一个帧（上面是把另一submap中的占据栅格复制到本地图中）
void Submap::AddScanInOccupancyMap(std::shared_ptr<Frame> frame) {
    occu_map_.AddLidarFrame(frame, OccupancyMap::GridMethod::MODEL_POINTS);  // 更新栅格地图中的格子
    field_.SetFieldImageFromOccuMap(occu_map_.GetOccupancyGrid());           // 更新场函数图像
}

bool Submap::HasOutsidePoints() const { return occu_map_.HasOutsidePoints(); }

//子地图在世界坐标系中的位姿
void Submap::SetPose(const SE2& pose) {
    pose_ = pose;
    occu_map_.SetPose(pose);
    field_.SetPose(pose);
}

void Submap::UpdateFramePoseWorld() {
    for (auto& frame : frames_) {
        frame->pose_ = pose_ * frame->pose_submap_;
    }
}

}  // namespace sad