/**
 * @description: 仿照增量式NDT，新建一个icp_inc_3d.cc文件，实现增量式ICP
 * @return {*}
 */
void LioIEKF::AlignICP_ikdtree() {
    FullCloudPtr scan_undistort_trans(new FullPointCloudType);
    pcl::transformPointCloud(*scan_undistort_fullcloud_, *scan_undistort_trans, TIL_.matrix().cast<float>());
    scan_undistort_fullcloud_ = scan_undistort_trans;

    scan_undistort_ = ConvertToCloud<FullPointType>(scan_undistort_fullcloud_);

    // 点云降采样
    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(0.5, 0.5, 0.5);
    voxel.setInputCloud(scan_undistort_);
    voxel.filter(*scan_down_body_); // 体素滤波，降采样

    /// the first scan
    if (flg_first_scan_) {
        // ndt_.AddCloud(scan_undistort_);
        icp_.SetTarget(scan_undistort_);  // 【新增】

        first_lidar_time_ = measures_.lidar_begin_time_;
        flg_first_scan_ = false;
        return;
    }

    // 后续的scan，使用NDT配合pose进行更新
    LOG(INFO) << "=== frame " << frame_num_;

    int cur_pts = scan_down_body_->size(); // 降采样后的去畸变点云数量

    // ndt_.SetSource(scan_down_body_);
    icp_.SetSource(scan_down_body_); // 【新增】为点面icp中的ikdtree设置原始点云
    ieskf_.UpdateUsingCustomObserve([this](const SE3 &input_pose, Mat18d &HTVH, Vec18d &HTVr) {
                                        // ndt_.ComputeResidualAndJacobians(input_pose, HTVH, HTVr);
                                        icp_.ComputeResidualAndJacobians_P2Plane(input_pose, HTVH, HTVr); // 【新增】计算点面残差和雅可比
                                    });

    auto current_nav_state = ieskf_.GetNominalState();

    // 若运动了一定范围，则把点云放入地图中
    SE3 current_pose = ieskf_.GetNominalSE3();
    SE3 delta_pose = last_pose_.inverse() * current_pose;

    if (delta_pose.translation().norm() > 1.0 || delta_pose.so3().log().norm() > math::deg2rad(10.0)) {
        // 将地图合入NDT中
        CloudPtr scan_down_world_2(new PointCloudType);
        pcl::transformPointCloud(*scan_down_body_, *scan_down_world_2, current_pose.matrix());
        // ndt_.AddCloud(scan_down_world_2);
        icp_.SetTarget(scan_down_world_2); // 【新增】为点面icp中的ikdtree设置目标点云，内部际是添加新的点云到ikdtree中
        last_pose_ = current_pose;
    }

    // 放入UI
    if (ui_) {
        ui_->UpdateScan(scan_undistort_, current_nav_state.GetSE3());  // 转成Lidar Pose传给UI
        ui_->UpdateNavState(current_nav_state);
    }

    frame_num_++;
    return;
}