// 1.2 基于g2o优化器的点到点ICP-方法2-g2o内部进行kdtree近邻搜索
bool Icp2d::AlignG2OP2P_2(SE2& init_pose) {
    int iterations = 10;                // 迭代次数
    double rk_delta = 0.8;
    float max_dis2 = 0.01;        // 最近邻时的最远距离（平方）
    int min_effect_pts = 20;      // 最小有效点数

    using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
    using LinearSolverType = g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType>;
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    
    auto* v = new VertexSE2();      // 新建SE2位姿顶点
    v->setId(0);                    // 设置顶点的id
    v->setEstimate(init_pose);   // 设置顶点的估计值为初始位姿
    optimizer.addVertex(v);         // 将顶点添加到优化器中
    int effective_num = 0;  // 有效点数
    // 遍历源始点云
    for (size_t i = 0; i < source_scan_->ranges.size(); ++i) {
        double range = source_scan_->ranges[i]; // 源始点云的距离
        // 判断每个点的距离是否越界
        if (range < source_scan_->range_min || range > source_scan_->range_max) 
            continue;

        // 根据最小角度和角分辨率计算每个点的角度
        double angle = source_scan_->angle_min + i * source_scan_->angle_increment;
        
        auto *edge = new EdgeSE2P2P_2(kdtree_2d, target_cloud_2d, range, angle);   

        edge->setVertex(0, v);                  // 设置边的第一个顶点为SE2位姿顶点
        if (edge->isPointValid()){
            effective_num++; 
            edge->setInformation(Mat2d::Identity());// 观测为2维点坐标，信息矩阵需设为2x2单位矩阵
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
    init_pose = v->estimate();
    LOG(INFO) << "estimated pose: " << v->estimate().translation().transpose() << ", theta: " << v->estimate().so2().log();
    return true;
}


class EdgeSE2P2P_2 : public g2o::BaseUnaryEdge<2, Vec2d, VertexSE2> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE2P2P_2(const pcl::search::KdTree<Point2d>::Ptr kdtree, const Cloud2d::Ptr target_cloud, double range, double angle) : kdtree_(kdtree),  target_cloud_(target_cloud), range_(range), angle_(angle) {}

    // 判断当前激光点的最近邻集合是否为空，或者最小距离是否大于最大距离阈值
    bool isPointValid() { 
        auto* pose = dynamic_cast<const VertexSE2*>(_vertices[0]);
        theta_ = pose->estimate().so2().log(); // 当前位姿的角度
        // 世界系下点的坐标 p_i^W，极坐标转笛卡尔坐标公式
        pw_ = pose->estimate() * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));

        Point2d pt;
        pt.x = pw_.x();
        pt.y = pw_.y();

        // 在目标点云的KD树中查找一个最近邻，返回该最近邻的索引和距离
        kdtree_->nearestKSearch(pt, 1, nn_idx_, dis_);
        float max_dis2 = 0.01;
        // 判断最近邻集合是否非空，且最小距离是否小于最大距离阈值
        if (nn_idx_.size() > 0 && dis_[0] < max_dis2) {
            // 当前激光点在目标点云中的最近邻点坐标
            qw_ = Vec2d(target_cloud_->points[nn_idx_[0]].x, target_cloud_->points[nn_idx_[0]].y);   
            return true;
        }
        else 
            return false;
    }
    
    // 定义残差
    void computeError() override {
        // 判断最近邻集合是否非空，且最小距离是否小于最大距离阈值
        if (isPointValid()) 
            _error =  pw_ - qw_; 
        else {
            _error = Vec2d(0, 0);
            setLevel(1);
        }
    }

    // 雅可比矩阵的解析形式
    void linearizeOplus() override {
        if (isPointValid()) {
            _jacobianOplusXi <<  1, 0, 0, 1,  // de / dx， de / dy
                                -range_ * std::sin(angle_ + theta_), range_ * std::cos(angle_ + theta_);  //  de / dtheta       
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
};