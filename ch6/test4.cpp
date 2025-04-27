// 基于g2o优化器的点到线ICP-方法2-g2o内部进行kdtree近邻搜索
class EdgeSE2P2L_2 : public g2o::BaseUnaryEdge<1, double, VertexSE2> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE2P2L_2(const pcl::search::KdTree<Point2d>::Ptr kdtree, const Cloud2d::Ptr target_cloud, double range, double angle) : kdtree_(kdtree),  target_cloud_(target_cloud), range_(range), angle_(angle) {}

    bool getIsLineFitSuccess() { return isLineFitSuccess_; }

    // 直线拟合是否成功
    bool isLineFitValid() { 
        auto* pose = dynamic_cast<const VertexSE2*>(_vertices[0]);
        theta_ = pose->estimate().so2().log(); // 当前位姿的角度
        // 世界系下点的坐标 p_i^W，极坐标转笛卡尔坐标公式
        pw_ = pose->estimate() * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));

        Point2d pt;
        pt.x = pw_.x();
        pt.y = pw_.y();

        // 在目标点云的KD树中查找一个最近邻，返回该最近邻的索引和距离
        kdtree_->nearestKSearch(pt, 5, nn_idx_, dis_);

        std::vector<Vec2d> effective_pts;  // 有效点
        float max_dis = 0.3;
        // 遍历所有五个近邻点
        for (int j = 0; j < nn_idx_.size(); ++j) {
            // 判断每个近邻点的距离是否处于最远阈值距离内
            if (dis_[j] < max_dis) 
                // 若是，该近邻点符合要求，存储到向量中
                effective_pts.emplace_back(Vec2d(target_cloud_->points[nn_idx_[j]].x, target_cloud_->points[nn_idx_[j]].y));
        }
        // 判断有效近邻点是否少于三个
        if (effective_pts.size() < 3) 
            // 若少于3个，则跳过当前激光点
            return false;

        
        // 利用当前点附近的几个有效近邻点，基于SVD奇异值分解，拟合出ax+by+c=0 中的最小直线系数 a,b,c，对应公式（6.11）
        if (math::FitLine2D(effective_pts, line_coeffs_)) {
            isLineFitSuccess_ = true;
            return isLineFitSuccess_;
        } else {
            isLineFitSuccess_ = false;
            return isLineFitSuccess_;
        }
    }
    
    // 定义残差
    void computeError() override {
        // 判断最近邻集合是否非空，且最小距离是否小于最大距离阈值
        if (isLineFitValid()) 
            _error[0] = line_coeffs_[0] * pw_[0] + line_coeffs_[1] * pw_[1] + line_coeffs_[2];
        else {
            _error[0] = 0.0;
            setLevel(1);
        }
    }

    // 雅可比矩阵的解析形式
    void linearizeOplus() override {
        if (isLineFitSuccess_) {
            _jacobianOplusXi << line_coeffs_[0], 
                                line_coeffs_[1], 
                                - line_coeffs_[0] * range_ * std::sin(angle_ + theta_) 
                                + line_coeffs_[1] * range_ * std::cos(angle_ + theta_);        
        } else {
            _jacobianOplusXi.setZero();
            setLevel(1);
        }                   
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

private:
    double range_ = 0;  // 距离
    double angle_ = 0;  // 角度

    // 【新增】
    double theta_ = 0;
    Vec2d pw_, qw_;
    const pcl::search::KdTree<Point2d>::Ptr kdtree_; 
    const Cloud2d::Ptr target_cloud_;
    std::vector<int> nn_idx_;    // 最近邻的索引
    std::vector<float> dis_;     // 最近邻的距离

    Vec3d line_coeffs_;  // 拟合直线，组装J、H和误差

    bool isLineFitSuccess_ = false;
};
bool Icp2d::AlignG2OP2L_2(SE2& init_pose) {
    int iterations = 10;        // 迭代次数
    double rk_delta = 0.8;
    float max_dis = 0.3;       // 最近邻时的最远距离（平方）
    int min_effect_pts = 20;    // 最小有效点数
    
    SE2 current_pose = init_pose;   // 当前位姿

    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
    using LinearSolverType = g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType>;
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    auto* v = new VertexSE2();      // 新建SE2位姿顶点
    v->setId(0);                    // 设置顶点的id
    v->setEstimate(current_pose);   // 设置顶点的估计值为初始位姿
    optimizer.addVertex(v);         // 将顶点添加到优化器中
    int effective_num = 0;          // 有效点数

    // 遍历源始点云
    for (size_t i = 0; i < source_scan_->ranges.size(); ++i) {
        double range = source_scan_->ranges[i]; // 源始点云的距离
        // 判断每个点的距离是否越界
        if (range < source_scan_->range_min || range > source_scan_->range_max) 
            continue;

        // 当前激光点的角度
        double angle = source_scan_->angle_min + i * source_scan_->angle_increment;
        
        auto *edge = new EdgeSE2P2L_2(kdtree_2d, target_cloud_2d, range, angle);

        edge->setVertex(0, v);                  // 设置边的第一个顶点为SE2位姿顶点
        

        // 利用当前点附近的几个有效近邻点，基于SVD奇异值分解，拟合出ax+by+c=0 中的最小直线系数 a,b,c，对应公式（6.11）
        if (edge->isLineFitValid()) {
            effective_num++; // 有效点数
            edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());// 观测为2维点坐标，因此信息矩阵需设为2x2单位矩阵
            auto rk = new g2o::RobustKernelHuber;   // Huber鲁棒核函数
            rk->setDelta(rk_delta);                 // 设置阈值
            edge->setRobustKernel(rk);              // 为边设置鲁棒核函数
            optimizer.addEdge(edge);                // 将约束边添加到优化器中
        }
    }

    // 判断有效激光点数是否少于最小有效点数阈值
    if (effective_num < min_effect_pts) 
        return false;

    optimizer.setVerbose(false);        // 不输出优化过程
    optimizer.initializeOptimization(); // 初始化优化器
    optimizer.optimize(10);              // g2o内部仅非线性优化求解一次

    // 取出优化后的SE2位姿，更新当前位姿，用于下一次迭代
    current_pose = v->estimate();
    
    init_pose = current_pose;
    LOG(INFO) << "estimated pose: " << current_pose.translation().transpose() << ", theta: " << current_pose.so2().log();
    // LOG(INFO) << "g2o: estimated pose: " << init_pose.translation().transpose() << ", theta: " << init_pose.so2().log();
    return true;
}


// 在http://test_2d_icp_s2s.cc中增加上述几种方法的选项，便于终端切换：
SE2 pose;
if (fLS::FLAGS_method == "point2point") {
    icp.AlignGaussNewton(pose);
} else if (fLS::FLAGS_method == "point2line") {
    icp.AlignGaussNewtonPoint2Plane(pose);
} else if (fLS::FLAGS_method == "point2point_g2o") {
    LOG(INFO) << "icp.AlignG2OP2P(pose): ";
    icp.AlignG2OP2P(pose);
} else if (fLS::FLAGS_method == "point2point_g2o_2") {
    LOG(INFO) << "icp.AlignG2OP2P_2(pose): ";
    icp.AlignG2OP2P_2(pose);
} else if (fLS::FLAGS_method == "point2point_g2o_3") {
    LOG(INFO) << "icp.AlignG2OP2P_3(pose): ";
    icp.AlignG2OP2P_3(pose);
} else if (fLS::FLAGS_method == "point2line_g2o") {
    LOG(INFO) << "icp.AlignG2OP2L(pose): ";
    icp.AlignG2OP2L(pose);
} else if (fLS::FLAGS_method == "point2line_g2o_2") {
    LOG(INFO) << "icp.AlignG2OP2L_2(pose): ";
    icp.AlignG2OP2L_2(pose);
}




