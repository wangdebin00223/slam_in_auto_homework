// 基于g2o优化器的点到线ICP-方法1-g2o外部进行kdtree近邻搜索
class EdgeSE2P2L : public g2o::BaseUnaryEdge<1, double, VertexSE2> {    // 测量值为2维；SE2类型位姿顶点
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE2P2L(double range, double angle, Vec3d line_coeffs) : range_(range), angle_(angle), line_coeffs_(line_coeffs) {}
    
    // 定义残差
    void computeError() override {
        auto* pose = dynamic_cast<const VertexSE2*>(_vertices[0]);
        Vec2d pw = pose->estimate() * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
        _error[0] = line_coeffs_[0] * pw[0] + line_coeffs_[1] * pw[1] + line_coeffs_[2]; 
    }

    // 雅可比矩阵的解析形式
    void linearizeOplus() override {
        auto* pose = dynamic_cast<const VertexSE2*>(_vertices[0]);
        float theta = pose->estimate().so2().log(); // 当前位姿的角度
        _jacobianOplusXi <<   line_coeffs_[0], 
                              line_coeffs_[1], 
                            - line_coeffs_[0] * range_ * std::sin(angle_ + theta) 
                            + line_coeffs_[1] * range_ * std::cos(angle_ + theta);  
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

private:
    double range_ = 0;  // 距离
    double angle_ = 0;  // 角度
    Vec3d line_coeffs_; // 直线拟合系数 A,B,C
};


bool Icp2d::AlignG2OP2L(SE2& init_pose) {
    int iterations = 10;        // 迭代次数
    double rk_delta = 0.8;
    float max_dis = 0.3;       // 最近邻时的最远距离（平方）
    int min_effect_pts = 20;    // 最小有效点数
    
    SE2 current_pose = init_pose;   // 当前位姿
    for (int iter = 0; iter < iterations; ++iter) {
        using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
        using LinearSolverType = g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType>;
        auto* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        auto* v = new VertexSE2();      // 新建SE2位姿顶点
        v->setId(0);                    // 设置顶点的id
        v->setEstimate(current_pose);   // 设置顶点的估计值为初始位姿
        optimizer.addVertex(v);         // 将顶点添加到优化器中
        int effective_num = 0;  // 有效点数
        // 遍历源始点云
        for (size_t i = 0; i < source_scan_->ranges.size(); ++i) {
            double range = source_scan_->ranges[i]; // 源始点云的距离
            // 判断每个点的距离是否越界
            if (range < source_scan_->range_min || range > source_scan_->range_max) 
                continue;

            // 当前激光点的角度
            double angle = source_scan_->angle_min + i * source_scan_->angle_increment;
            // 从上一次迭代得到的位姿 T_wb 的2x2旋转矩阵中，利用对数映射获取对应的旋转角度
            double theta = current_pose.so2().log();
            // 机器人坐标系下的极坐标转换为笛卡尔坐标，并转为世界坐标系下的坐标 p_i^W，
            Vec2d pw = current_pose * Vec2d(range * std::cos(angle), range * std::sin(angle));
            Point2d pt;
            pt.x = pw.x();
            pt.y = pw.y();

            // 查找5个最近邻
            std::vector<int> nn_idx;    // 最近邻的索引
            std::vector<float> dis;     // 最近邻的距离
            kdtree_2d->nearestKSearch(pt, 5, nn_idx, dis);

            std::vector<Vec2d> effective_pts;  // 有效点
            // 遍历所有五个近邻点
            for (int j = 0; j < nn_idx.size(); ++j) {
                // 判断每个近邻点的距离是否处于最远阈值距离内
                if (dis[j] < max_dis) 
                    // 若是，该近邻点符合要求，存储到向量中
                    effective_pts.emplace_back(Vec2d(target_cloud_2d->points[nn_idx[j]].x, target_cloud_2d->points[nn_idx[j]].y));
            }
            // 判断有效近邻点是否少于三个
            if (effective_pts.size() < 3) 
                // 若少于3个，则跳过当前激光点
                continue;

            // 拟合直线，组装J、H和误差
            Vec3d line_coeffs;
            // 利用当前点附近的几个有效近邻点，基于SVD奇异值分解，拟合出ax+by+c=0 中的最小直线系数 a,b,c，对应公式（6.11）
            if (math::FitLine2D(effective_pts, line_coeffs)) {
                effective_num++; // 有效点数
                auto *edge = new EdgeSE2P2L(range, angle, line_coeffs);
                edge->setVertex(0, v);                  // 设置边的第一个顶点为SE2位姿顶点
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
        optimizer.optimize(1);              // g2o内部仅非线性优化求解一次

        // 取出优化后的SE2位姿，更新当前位姿，用于下一次迭代
        current_pose = v->estimate();
    }
    init_pose = current_pose;
    LOG(INFO) << "estimated pose: " << current_pose.translation().transpose() << ", theta: " << current_pose.so2().log();
    // LOG(INFO) << "g2o: estimated pose: " << init_pose.translation().transpose() << ", theta: " << init_pose.so2().log();
    return true;
}