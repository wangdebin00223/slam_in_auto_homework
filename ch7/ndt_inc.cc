//
// Created by xiang on 2022/7/20.
//

#include "ch7/ndt_inc.h"
#include "common/lidar_utils.h"
#include "common/math_utils.h"
#include "common/timer/timer.h"

#include <glog/logging.h>
#include <execution>
#include <set>

namespace sad {

void IncNdt3d::AddCloud(CloudPtr cloud_world) {
    std::set<KeyType, less_vec<3>> active_voxels;  // 记录哪些voxel被更新
    for (const auto& p : cloud_world->points) {
        auto pt = ToVec3d(p);
        auto key = CastToInt(Vec3d(pt * options_.inv_voxel_size_));
        auto iter = grids_.find(key);
        if (iter == grids_.end()) {
            // 栅格不存在
            //data_：双向链表（std::list），存储体素数据
            //data_.push_front({key, {pt}})：在链表头部插入新的体素数据
            data_.push_front({key, {pt}});

            //将体素 key 和链表头部的迭代器插入哈希表
            grids_.insert({key, data_.begin()});

            //如果链表大小超过容量限制，则删除链表尾部的体素数据
            if (data_.size() >= options_.capacity_) {
                // 删除一个尾部的数据
                //从哈希表中删除尾部体素的键(key)
                grids_.erase(data_.back().first);

                //从链表中删除尾部体素数据.
                data_.pop_back();
            }
        } else {
            // 栅格存在，添加点，更新缓存
            iter->second->second.AddPoint(pt);//将点 pt 添加到体素中
            // iter：查找体素 key 的迭代器。
            // iter->second：指向 data_ 链表中体素数据的迭代器。
            // iter->second->second：体素数据对象（例如 VoxelData），包含体素的均值和协方差矩阵。
            // AddPoint(pt)：将点 pt 添加到体素中，更新体素的均值和协方差矩阵。

            data_.splice(data_.begin(), data_, iter->second);  // 更新的那个放到最前
            // data_.begin()：目标位置，表示将元素移动到 data_ 链表的头部。
            // data_：源链表，与目标链表是同一个链表。
            // iter->second：源链表中的迭代器，指向要移动的元素。

            //哈希表中体素 key 对应的链表迭代器。
            iter->second = data_.begin();                      // grids时也指向最前
        }

        active_voxels.emplace(key);//将体素 key 添加到活跃体素集合中
    }

    // 更新active_voxels
    //std::execution::par_unseq：并行执行策略，表示可以并行且无序地执行操作。
    // par：允许并行执行。
    // unseq：允许无序执行（即不保证顺序）。
    std::for_each(std::execution::par_unseq, active_voxels.begin(), active_voxels.end(),
                  [this](const auto& key) { UpdateVoxel(grids_[key]->second); });
    flag_first_scan_ = false;

}

void IncNdt3d::GenerateNearbyGrids() {
    if (options_.nearby_type_ == NearbyType::CENTER) {
        nearby_grids_.emplace_back(KeyType::Zero());
    } else if (options_.nearby_type_ == NearbyType::NEARBY6) {
        nearby_grids_ = {KeyType(0, 0, 0),  KeyType(-1, 0, 0), KeyType(1, 0, 0), KeyType(0, 1, 0),
                         KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1)};
    }
}

void IncNdt3d::UpdateVoxel(VoxelData& v) {
    if (flag_first_scan_) {
        if (v.pts_.size() > 1) {
            math::ComputeMeanAndCov(v.pts_, v.mu_, v.sigma_, [this](const Vec3d& p) { return p; });
            //[this](const Vec3d& p) { return p; }：Lambda 表达式，用于将点转换为三维向量.
            v.info_ = (v.sigma_ + Mat3d::Identity() * 1e-3).inverse();  // 避免出nan
            //v.sigma_ + Mat3d::Identity() * 1e-3：
            // 在协方差矩阵上添加一个小的正则化项（单位矩阵的缩放），以避免矩阵奇异（不可逆）。

        } else {
            v.mu_ = v.pts_[0];
            v.info_ = Mat3d::Identity() * 1e2;
        }

        v.ndt_estimated_ = true;
        v.pts_.clear();
        return;
    }

    if (v.ndt_estimated_ && v.num_pts_ > options_.max_pts_in_voxel_) {
        return;
    }

    if (!v.ndt_estimated_ && v.pts_.size() > options_.min_pts_in_voxel_) {
        // 新增的voxel
        math::ComputeMeanAndCov(v.pts_, v.mu_, v.sigma_, [this](const Vec3d& p) { return p; });
        v.info_ = (v.sigma_ + Mat3d::Identity() * 1e-3).inverse();  // 避免出nan
        v.ndt_estimated_ = true;
        v.pts_.clear();
    } else if (v.ndt_estimated_ && v.pts_.size() > options_.min_pts_in_voxel_) {
        // 已经估计，而且还有新来的点
        Vec3d cur_mu, new_mu;
        Mat3d cur_var, new_var;
        math::ComputeMeanAndCov(v.pts_, cur_mu, cur_var, [this](const Vec3d& p) { return p; });
        //[this](const Vec3d& p) { return p; }：Lambda 表达式，用于将点转换为三维向量。


        math::UpdateMeanAndCov(v.num_pts_, v.pts_.size(), v.mu_, v.sigma_, cur_mu, cur_var, new_mu, new_var);

        v.mu_ = new_mu;
        v.sigma_ = new_var;
        v.num_pts_ += v.pts_.size();
        v.pts_.clear();

        // check info
        Eigen::JacobiSVD svd(v.sigma_, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Vec3d lambda = svd.singularValues();
        if (lambda[1] < lambda[0] * 1e-3) {
            lambda[1] = lambda[0] * 1e-3;
        }

        if (lambda[2] < lambda[0] * 1e-3) {
            lambda[2] = lambda[0] * 1e-3;
        }

        Mat3d inv_lambda = Vec3d(1.0 / lambda[0], 1.0 / lambda[1], 1.0 / lambda[2]).asDiagonal();
        v.info_ = svd.matrixV() * inv_lambda * svd.matrixU().transpose();
        
        //如果 lambda[1] 或 lambda[2] 过小（小于 lambda[0] * 1e-3），则将其调整为 lambda[0] * 1e-3，避免数值不稳定。
        // 计算信息矩阵：
        // inv_lambda：奇异值的倒数构成的对角矩阵。
        // v.info_：更新信息矩阵。
    }
}

bool IncNdt3d::AlignNdt(SE3& init_pose) {
    LOG(INFO) << "aligning with inc ndt, pts: " << source_->size() << ", grids: " << grids_.size();
    assert(grids_.empty() == false);

    SE3 pose = init_pose;

    // 对点的索引，预先生成
    int num_residual_per_point = 1;
    if (options_.nearby_type_ == NearbyType::NEARBY6) {
        num_residual_per_point = 7;
    }

    std::vector<int> index(source_->points.size());
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    // 我们来写一些并发代码
    int total_size = index.size() * num_residual_per_point;

    for (int iter = 0; iter < options_.max_iteration_; ++iter) {
        std::vector<bool> effect_pts(total_size, false);
        std::vector<Eigen::Matrix<double, 3, 6>> jacobians(total_size);
        std::vector<Vec3d> errors(total_size);
        std::vector<Mat3d> infos(total_size);

        // gauss-newton 迭代
        // 最近邻，可以并发
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
            auto q = ToVec3d(source_->points[idx]);
            Vec3d qs = pose * q;  // 转换之后的q

            // 计算qs所在的栅格以及它的最近邻栅格
            Vec3i key = CastToInt(Vec3d(qs * options_.inv_voxel_size_));

            for (int i = 0; i < nearby_grids_.size(); ++i) {
                Vec3i real_key = key + nearby_grids_[i];
                auto it = grids_.find(real_key);
                int real_idx = idx * num_residual_per_point + i;
                /// 这里要检查高斯分布是否已经估计
                if (it != grids_.end() && it->second->second.ndt_estimated_) {
                    auto& v = it->second->second;  // voxel
                    Vec3d e = qs - v.mu_;

                    // check chi2 th
                    double res = e.transpose() * v.info_ * e;
                    if (std::isnan(res) || res > options_.res_outlier_th_) {
                        effect_pts[real_idx] = false;
                        continue;
                    }

                    // build residual
                    Eigen::Matrix<double, 3, 6> J;
                    J.block<3, 3>(0, 0) = -pose.so3().matrix() * SO3::hat(q);
                    J.block<3, 3>(0, 3) = Mat3d::Identity();

                    jacobians[real_idx] = J;
                    errors[real_idx] = e;
                    infos[real_idx] = v.info_;
                    effect_pts[real_idx] = true;
                } else {
                    effect_pts[real_idx] = false;
                }
            }
        });

        // 累加Hessian和error,计算dx
        double total_res = 0;

        int effective_num = 0;

        Mat6d H = Mat6d::Zero();
        Vec6d err = Vec6d::Zero();

        for (int idx = 0; idx < effect_pts.size(); ++idx) {
            if (!effect_pts[idx]) {
                continue;
            }

            total_res += errors[idx].transpose() * infos[idx] * errors[idx];
            effective_num++;

            H += jacobians[idx].transpose() * infos[idx] * jacobians[idx];
            err += -jacobians[idx].transpose() * infos[idx] * errors[idx];
        }

        if (effective_num < options_.min_effective_pts_) {
            LOG(WARNING) << "effective num too small: " << effective_num;
            init_pose = pose;
            return false;
        }

        Vec6d dx = H.inverse() * err;
        pose.so3() = pose.so3() * SO3::exp(dx.head<3>());
        pose.translation() += dx.tail<3>();

        // 更新
        LOG(INFO) << "iter " << iter << " total res: " << total_res << ", eff: " << effective_num
                  << ", mean res: " << total_res / effective_num << ", dxn: " << dx.norm()
                  << ", dx: " << dx.transpose();

        if (dx.norm() < options_.eps_) {
            LOG(INFO) << "converged, dx = " << dx.transpose();
            break;
        }
    }

    init_pose = pose;
    return true;
}

void IncNdt3d::ComputeResidualAndJacobians(const SE3& input_pose, Mat18d& HTVH, Vec18d& HTVr) {
    assert(grids_.empty() == false);
    SE3 pose = input_pose;

    // 大部分流程和前面的Align是一样的，只是会把z, H, R三者抛出去而非自己处理
    int num_residual_per_point = 1;
    if (options_.nearby_type_ == NearbyType::NEARBY6) {
        num_residual_per_point = 7;
    }

    std::vector<int> index(source_->points.size());
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    int total_size = index.size() * num_residual_per_point;

    std::vector<bool> effect_pts(total_size, false);
    std::vector<Eigen::Matrix<double, 3, 18>> jacobians(total_size);
    std::vector<Vec3d> errors(total_size);
    std::vector<Mat3d> infos(total_size);

    // gauss-newton 迭代
    // 最近邻，可以并发
    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
        auto q = ToVec3d(source_->points[idx]);
        Vec3d qs = pose * q;  // 转换之后的q

        // 计算qs所在的栅格以及它的最近邻栅格
        Vec3i key = CastToInt(Vec3d(qs * options_.inv_voxel_size_));

        for (int i = 0; i < nearby_grids_.size(); ++i) {
            Vec3i real_key = key + nearby_grids_[i];
            auto it = grids_.find(real_key);
            int real_idx = idx * num_residual_per_point + i;
            /// 这里要检查高斯分布是否已经估计
            if (it != grids_.end() && it->second->second.ndt_estimated_) {
                //检查栅格是否已经进行了NDT估计
                auto& v = it->second->second;  // voxel
                Vec3d e = qs - v.mu_;
                //计算当前点 qs 与栅格中心 v.mu_ 的误差 e

                // check chi2 th
                double res = e.transpose() * v.info_ * e;
                if (std::isnan(res) || res > options_.res_outlier_th_) {
                    effect_pts[real_idx] = false;
                    continue;
                }

                // build residual
                Eigen::Matrix<double, 3, 18> J;
                J.setZero();
                J.block<3, 3>(0, 0) = Mat3d::Identity();                   // 对p
                J.block<3, 3>(0, 6) = -pose.so3().matrix() * SO3::hat(q);  // 对R

                jacobians[real_idx] = J;
                errors[real_idx] = e;
                infos[real_idx] = v.info_;
                effect_pts[real_idx] = true;
            } else {
                effect_pts[real_idx] = false;
            }
        }
    });

    // 累加Hessian和error,计算dx
    double total_res = 0;
    int effective_num = 0;

    HTVH.setZero();
    HTVr.setZero();

    const double info_ratio = 0.01;  // 每个点反馈的info因子

    for (int idx = 0; idx < effect_pts.size(); ++idx) {
        if (!effect_pts[idx]) {
            continue;
        }

        total_res += errors[idx].transpose() * infos[idx] * errors[idx];
        effective_num++;

        HTVH += jacobians[idx].transpose() * infos[idx] * jacobians[idx] * info_ratio;
        HTVr += -jacobians[idx].transpose() * infos[idx] * errors[idx] * info_ratio;
    }

    LOG(INFO) << "effective: " << effective_num;
}

void IncNdt3d::BuildNDTEdges(sad::VertexPose* v, std::vector<EdgeNDT*>& edges) {
    assert(grids_.empty() == false);
    SE3 pose = v->estimate();

    /// 整体流程和NDT一致，只是把查询函数放到edge内部，建立和v绑定的边
    for (const auto& pt : source_->points) {
        Vec3d q = ToVec3d(pt);
        auto edge = new EdgeNDT(v, q, [this](const Vec3d& qs, Vec3d& mu, Mat3d& info) -> bool {
            Vec3i key = CastToInt(Vec3d(qs * options_.inv_voxel_size_));

            auto it = grids_.find(key);
            /// 这里要检查高斯分布是否已经估计
            if (it != grids_.end() && it->second->second.ndt_estimated_) {
                auto& v = it->second->second;  // voxel
                mu = v.mu_;
                info = v.info_;
                return true;
            } else {
                return false;
            }
        });

        if (edge->IsValid()) {
            edges.emplace_back(edge);
        } else {
            delete edge;
        }
    }
}

}  // namespace sad