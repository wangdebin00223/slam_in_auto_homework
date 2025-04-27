std::vector<bool> effect_ground(ground_size, false);                  // 【新增】
std::vector<Eigen::Matrix<double, 1, 6>> jacob_ground(ground_size);  // 点面的残差是1维的
std::vector<double> errors_ground(ground_size);

/// 最近邻，地面点部分，点面ICP
if (options_.use_ground_points_) {
    std::for_each(std::execution::par_unseq, index_ground.begin(), index_ground.end(), [&](int idx) {
        Vec3d q = ToVec3d(ground->points[idx]);
        Vec3d qs = pose * q;

        // 检查最近邻
        std::vector<int> nn_indices;

        kdtree_ground_.GetClosestPoint(ToPointType(qs), nn_indices, 5);
        effect_ground[idx] = false;

        if (nn_indices.size() == 5) {
            std::vector<Vec3d> nn_eigen;
            for (auto& n : nn_indices) 
                nn_eigen.emplace_back(ToVec3d(local_map_ground_->points[n]));

            // 点面残差
            Vec4d n;
            if (!math::FitPlane(nn_eigen, n)) 
                return;

            double dis = n.head<3>().dot(qs) + n[3];
            if (fabs(dis) > options_.max_plane_distance_) 
                return;

            effect_ground[idx] = true;

            // build residual
            Eigen::Matrix<double, 1, 6> J;
            J.block<1, 3>(0, 0) = -n.head<3>().transpose() * pose.so3().matrix() * SO3::hat(q);
            J.block<1, 3>(0, 3) = n.head<3>().transpose();

            jacob_ground[idx] = J;
            errors_ground[idx] = dis;
        }
    });
}

// 海森矩阵与地面残差累加
for (const auto& idx : index_ground) {
    if (effect_ground[idx]) {
        H += jacob_ground[idx].transpose() * jacob_ground[idx];
        err += -jacob_ground[idx].transpose() * errors_ground[idx];
        effective_num++;
        total_res += errors_ground[idx] * errors_ground[idx];
    }
}