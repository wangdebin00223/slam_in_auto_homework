/// 【新增】使用g2o进行点到点的ICP配准
bool AlignG2OP2P(SE2& init_pose);   // 外部for循环迭代10次，内部for循环遍历scan点云，kdtree搜索current_pose转换的pw点得到qw，optimize(1)，每次迭代后更新current_pose
bool AlignG2OP2P_2(SE2& init_pose); // for循环遍历scan点云，kdtree传入位姿边构造函数，g2o内部每次迭代更新的位姿转换得到pw点进行近邻搜索得到qw，optimize(10)

bool AlignG2OP2P_3(SE2& init_pose); // for循环遍历scan点云，每个点都使用initial_pose转换的pw点，去进行近邻搜索得到qw，optimize(10)

/// 【新增】使用g2o进行点到线的ICP配准（2D激光场景中，并不存在面，可以将点到线的形式看成低维的点到面）
bool AlignG2OP2L(SE2& init_pose);   // 外部for循环迭代10次，内部for循环遍历scan点云，kdtree搜索current_pose转换的pw点得到qw，optimize(1)，每次迭代后更新current_pose
bool AlignG2OP2L_2(SE2& init_pose); // for循环遍历scan点云，kdtree传入位姿边构造函数，g2o内部每次迭代更新的位姿转换得到pw点进行近邻搜索得到qw，optimize(10)


// 基于g2o优化器的点到点ICP-方法1-g2o外部进行kdtree近邻搜索
/**
 * @description: 基于G2O优化器的2D 点到点ICP算法
 * @param {SE2&} init_pose 初始位姿
 * @return {*}
 */
bool Icp2d::AlignG2OP2P(SE2& init_pose) {
    int iterations = 10;                // 迭代次数
    double rk_delta = 0.8;
    float max_dis2 = 0.01;        // 最近邻时的最远距离（平方）
    int min_effect_pts = 20;      // 最小有效点数
    SE2 current_pose = init_pose;   // 当前位姿
    for (int iter = 0; iter < iterations; ++iter) {
        using BlockSolverType = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
        // 第一个参数 3 表示顶点（Vertex）的状态向量维度。对于 SE(2) 位姿，状态向量包含 3 个元素：x, y 和 θ
        // 第二个参数 1 表示边（Edge）的测量值维度。这里表示的是 1 维的测量值。
        // 使用了 using 关键字为复杂的模板类型创建一个简短的别名
        using LinearSolverType = g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType>;

        // 这是 g2o 库中的一个线性求解器类，基于 Cholesky 分解（Cholmod）实现。它用于解决稀疏线性系统，通常在非线性优化问题中作为底层求解器。
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

            // 根据最小角度和角分辨率计算每个点的角度
            double angle = source_scan_->angle_min + i * source_scan_->angle_increment;
            double theta = current_pose.so2().log(); // 当前位姿的角度

            // 世界系下点的坐标 p_i^W，极坐标转笛卡尔坐标公式
            Vec2d pw = current_pose * Vec2d(range * std::cos(angle), range * std::sin(angle));

            Point2d pt;
            pt.x = pw.x();
            pt.y = pw.y();

            // 最近邻
            std::vector<int> nn_idx;    // 最近邻的索引
            std::vector<float> dis;     // 最近邻的距离
            // 在目标点云的KD树中查找一个最近邻，返回该最近邻的索引和距离
            kdtree_2d->nearestKSearch(pt, 1, nn_idx, dis);

            // 判断最近邻集合是否非空，且最小距离是否小于最大距离阈值
            if (nn_idx.size() > 0 && dis[0] < max_dis2) {
                effective_num++;    // 有效点数自增一
                Vec2d qw = Vec2d(target_cloud_2d->points[nn_idx[0]].x, target_cloud_2d->points[nn_idx[0]].y);   // 当前激光点在目标点云中的最近邻点坐标
                auto *edge = new EdgeSE2P2P(range, angle, qw, theta);   // 构建约束边，参数为：激光点的距离、角度、近邻点坐标、当前旋转角度
                edge->setVertex(0, v);                      // 设置边的第一个顶点为SE2位姿顶点
                edge->setInformation(Mat2d::Identity());    // 观测为2维点坐标，因此信息矩阵需设为2x2单位矩阵
                auto rk = new g2o::RobustKernelHuber;       // Huber鲁棒核函数
                rk->setDelta(rk_delta);                     // 设置阈值
                edge->setRobustKernel(rk);                  // 为边设置鲁棒核函数
                optimizer.addEdge(edge);                    // 将约束边添加到优化器中
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



/**
 * @description: 一元边，定义了SE2位姿的残差，以及残差相对于（x,y,theta）雅可比矩阵的解析形式
 * @return {*}
 */
class EdgeSE2P2P : public g2o::BaseUnaryEdge<2, Vec2d, VertexSE2> {    // 测量值为2维；SE2类型位姿顶点
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE2P2P(double range, double angle, Vec2d qw, double theta) : range_(range), angle_(angle), qw_(qw), theta_(theta) {}
    
    // 定义残差
    void computeError() override {
        auto* pose = dynamic_cast<const VertexSE2*>(_vertices[0]);
        _error = pose->estimate() * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_)) - qw_; // pw - qw
    }

    // 雅可比矩阵的解析形式
    void linearizeOplus() override {
        _jacobianOplusXi <<  1, 0, 0, 1,  // de / dx， de / dy
                            -range_ * std::sin(angle_ + theta_), range_ * std::cos(angle_ + theta_);  //  de / dtheta
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

private:
    double range_ = 0;  // 距离
    double angle_ = 0;  // 角度
    double theta_ = 0;
    Vec2d qw_;          // 世界系下的近邻点坐标
};