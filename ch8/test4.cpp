/**
 * @description: 局部地图处理
 * @return {*}
 */
void LioIEKF::MapIncremental() {
    PointVec points_to_add;
    PointVec point_no_need_downsample;

    int cur_pts = scan_down_body_->size();
    points_to_add.reserve(cur_pts);
    point_no_need_downsample.reserve(cur_pts);

    std::vector<size_t> index(cur_pts);
    for (size_t i = 0; i < cur_pts; ++i) 
        index[i] = i;

    // 并发处理
    std::for_each(  std::execution::unseq, 
                    index.begin(), index.end(), 
                    [&](const size_t &i) {
                        /* transform to world frame */
                        // 雷达系转换到世界系
                        // PointBodyToWorld(&(scan_down_body_->points[i]), &(scan_down_world_->points[i]));

                        /* decide if need add to map */
                        // 判断是否需要加入到局部地图中
                        PointType &point_world = scan_down_world_->points[i];

                        // 判断第i个点的近邻点集是否为空
                        if (!nearest_points_[i].empty() && flg_ESKF_inited_) {
                            // 取出第i个点的近邻点集
                            const PointVec &points_near = nearest_points_[i];

                            // 计算中心坐标
                            Eigen::Vector3f center = ((point_world.getVector3fMap() / filter_size_map_min_).array().floor() + 0.5) * filter_size_map_min_;

                            // 计算第i个点到中心点的L1距离
                            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

                            // 判断距离是否大于阈值
                            if (fabs(dis_2_center.x()) > 0.5 * filter_size_map_min_ &&
                                fabs(dis_2_center.y()) > 0.5 * filter_size_map_min_ &&
                                fabs(dis_2_center.z()) > 0.5 * filter_size_map_min_) {
                                // 若是，则加入到无需降采样点集中
                                point_no_need_downsample.emplace_back(point_world);
                                return; // 程序返回？因为这里是lambda函数内部，所以返回的是lambda函数，而不是MapIncremental函数
                            }

                            // 此时，标记改为需要增加
                            bool need_add = true;
                            // 计算第i个点到中心点的L2距离
                            float dist = math::calc_dist(point_world.getVector3fMap(), center); // 【在math_utils.h】中添加了两个函数实现
                            // 判断近邻点数是否多于5个
                            if (points_near.size() >= options_.NUM_MATCH_POINTS) {
                                // 遍历所有近邻点
                                for (int readd_i = 0; readd_i < options_.NUM_MATCH_POINTS; readd_i++) {
                                    // 判断这些近邻点距离中心点的距离是否小于阈值
                                    if (math::calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) {
                                        need_add = false; // 只要有一个距离很小的，就不需要增加了，直接跳出循环
                                        break;
                                    }
                                }
                            }
                            // 判断是否需要增加
                            if (need_add) 
                                // 加入到需要增加的点集中
                                points_to_add.emplace_back(point_world);
                        } else 
                            points_to_add.emplace_back(point_world);
                    });

    LOG(INFO) << "points_to_add.size: " << points_to_add.size() << " point_no_need_downsample.size: " << point_no_need_downsample.size();
    icp_.SetTarget(points_to_add);            // 【新增】为点面icp中的ikdtree设置目标点云，内部际是添加新的点云到ikdtree中
    icp_.SetTarget(point_no_need_downsample); // 【新增】为点面icp中的ikdtree设置目标点云，内部际是添加新的点云到ikdtree中
}