//
// Created by xiang on 2021/9/22.
//

#include "ch5/kdtree.h"
#include "common/math_utils.h"

#include <glog/logging.h>
#include <execution>
#include <set>

namespace sad {

bool KdTree::BuildTree(const CloudPtr &cloud) {
    if (cloud->empty()) {
        return false;
    }

    cloud_.clear();
    cloud_.resize(cloud->size());
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        cloud_[i] = ToVec3f(cloud->points[i]);
    }

    Clear();
    Reset();

    IndexVec idx(cloud->size());
    for (int i = 0; i < cloud->points.size(); ++i) {
        idx[i] = i;
    }

    Insert(idx, root_.get());
    return true;
}

void KdTree::Insert(const IndexVec &points, KdTreeNode *node) {
    //points 点的索引集合
    //node 当前 Kd-Tree 节点

    //将当前节点 node 插入到 nodes_ 容器中
    nodes_.insert({node->id_, node});
    if (points.empty()) {
        return;
    }

    //如果点集中只有一个点，将当前节点标记为叶子节点
    //size_++：增加 Kd-Tree 的节点计数
    //node->point_idx_ = points[0]：将当前节点的 point_idx_ 设置为该点的索引
    if (points.size() == 1) {
        size_++;
        node->point_idx_ = points[0];
        return;
    }

    IndexVec left, right;
    if (!FindSplitAxisAndThresh(points, node->axis_index_, node->split_thresh_, left, right)) {
        size_++;
        node->point_idx_ = points[0];
        return;
    }

    //定义一个 lambda 函数，用于递归插入非空点集
    const auto create_if_not_empty = [&node, this](KdTreeNode *&new_node, const IndexVec &index) {
        if (!index.empty()) {
            new_node = new KdTreeNode;
            new_node->id_ = tree_node_id_++;
            Insert(index, new_node);
            //创建一个新节点 new_node
            //分配唯一 ID：new_node->id_ = tree_node_id_++
        }
    };
    //递归插入左右子树
    create_if_not_empty(node->left_, left);
    create_if_not_empty(node->right_, right);
}

bool KdTree::GetClosestPoint(const PointType &pt, std::vector<int> &closest_idx, int k) {
    if (k > size_) {
        LOG(ERROR) << "cannot set k larger than cloud size: " << k << ", " << size_;
        return false;
    }
    k_ = k;

    std::priority_queue<NodeAndDistance> knn_result;
    //初始化一个优先队列，用于存储最近邻点std::priority_queue<NodeAndDistance> knn_result；

    Knn(ToVec3f(pt), root_.get(), knn_result);

    // 排序并返回结果
    //堆顶元素是距离最大的点
    //堆中的其他元素没有严格的顺序
    closest_idx.resize(knn_result.size());
    for (int i = closest_idx.size() - 1; i >= 0; --i) {
        // 倒序插入
        closest_idx[i] = knn_result.top().node_->point_idx_;
        knn_result.pop();
    }
    //如果我们直接使用 knn_result 中的数据，输出的顺序会是 距离从大到小，而不是我们期望的 从小到大
    return true;
}

bool KdTree::GetClosestPointMT(const CloudPtr &cloud, std::vector<std::pair<size_t, size_t>> &matches, int k) {
    matches.resize(cloud->size() * k);

    // 索引
    std::vector<int> index(cloud->size());
    for (int i = 0; i < cloud->points.size(); ++i) {
        index[i] = i;
    }

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [this, &cloud, &matches, &k](int idx) {
        //这个idx其实就是index.begin(), index.end()中的一个对吧？[this, &cloud, &matches, &k]是指捕获该函数的这些参数
        std::vector<int> closest_idx;
        GetClosestPoint(cloud->points[idx], closest_idx, k);
        for (int i = 0; i < k; ++i) {
            matches[idx * k + i].second = idx;
            if (i < closest_idx.size()) {
                matches[idx * k + i].first = closest_idx[i];
            } else {
                matches[idx * k + i].first = math::kINVALID_ID;
                //如果 i 大于等于 closest_idx 的大小，说明没有找到足够的最近邻点。
                //将 matches 中的值设置为 math::kINVALID_ID，表示无效匹配。
            }
        }
    });
    return true;
}

void KdTree::Knn(const Vec3f &pt, KdTreeNode *node, std::priority_queue<NodeAndDistance> &knn_result) const {
    if (node->IsLeaf()) {
        // 如果是叶子，检查叶子是否能插入
        ComputeDisForLeaf(pt, node, knn_result);
        return;
    }

    // 看pt落在左还是右，优先搜索pt所在的子树
    // 然后再看另一侧子树是否需要搜索
    KdTreeNode *this_side, *that_side;
    ////查询点，在某个轴的值，与我们阈值的关系,如果小于阈值那么就在左边否则就在右边
    if (pt[node->axis_index_] < node->split_thresh_) {
        this_side = node->left_;
        that_side = node->right_;
    } else {
        this_side = node->right_;
        that_side = node->left_;
    }

    Knn(pt, this_side, knn_result);
    if (NeedExpand(pt, node, knn_result)) {  // 注意这里是跟自己比
        Knn(pt, that_side, knn_result);
    }

}

bool KdTree::NeedExpand(const Vec3f &pt, KdTreeNode *node, std::priority_queue<NodeAndDistance> &knn_result) const {
    if (knn_result.size() < k_) {
        return true;
    }

    if (approximate_) {
        float d = pt[node->axis_index_] - node->split_thresh_;
        //查询点，在某个轴的值，与我们阈值的关系
        //如果查询点到分割面的距离小于当前最近邻点中的最大距离，返回 true，表示需要扩展子树。
        if ((d * d) < knn_result.top().distance2_ * alpha_) {
            return true;
        } else {
            return false;
        }
    } else {
        // 检测切面距离，看是否有比现在更小的
        float d = pt[node->axis_index_] - node->split_thresh_;
        if ((d * d) < knn_result.top().distance2_) {
            return true;
        } else {
            return false;
        }
    }
}


//对叶子节点，计算它和查询点的距离，尝试放入结果中
//计算查询点 pt 与叶子节点 node 中存储的点的距离，并更新 K 近邻结果
void KdTree::ComputeDisForLeaf(const Vec3f &pt, KdTreeNode *node,
                               std::priority_queue<NodeAndDistance> &knn_result) const {
    // 比较与结果队列的差异，如果优于最远距离，则插入
    //计算查询点 pt 与叶子节点 node 中存储的点的平方距离
    //叶子节点只有一个点存储，所以可以直接用索引计算距离
    float dis2 = Dis2(pt, cloud_[node->point_idx_]);
    
    //
    if (knn_result.size() < k_) {
        // results 不足k
        knn_result.emplace(node, dis2);
    } else {
        // results等于k，比较current与max_dis_iter之间的差异
        if (dis2 < knn_result.top().distance2_) {
            //knn_result 是一个最大堆（默认行为），堆顶元素是当前结果中距离最大的点。
            knn_result.emplace(node, dis2);
            //优先队列是基于堆实现的，插入新元素后会自动调整堆结构
            //确保堆顶元素是最大值（最大堆）或最小值（最小堆）。
            knn_result.pop();
        }
    }
}

bool KdTree::FindSplitAxisAndThresh(const IndexVec &point_idx, int &axis, float &th, IndexVec &left, IndexVec &right) {
    // 计算三个轴上的散布情况，我们使用math_utils.h里的函数.
    //确定分割轴和分割阈值，并将点集分为左右两部分.
    Vec3f var;
    Vec3f mean;

    //var：存储点集在三个轴上的方差.
    //mean：存储点集在三个轴上的均值.


    //计算均值和协方差.
    //该函数会遍历 point_idx 中的所有索引，使用提供的lambda表达式获取每个点的坐标.
    //计算所有点的均值 mean.
    //计算每个维度上的方差 var，即协方差矩阵的对角线元素.
    math::ComputeMeanAndCovDiag(point_idx, mean, var, [this](int idx) { return cloud_[idx]; });
    
    int max_i, max_j;
    var.maxCoeff(&max_i, &max_j);
    //var.maxCoeff(&max_i, &max_j)：找到方差最大的轴.
    //max_i：方差最大的轴的索引（0 表示 x 轴，1 表示 y 轴，2 表示 z 轴）.
    //max_j：未使用（因为 var 是一维向量）.
    //将分割轴设置为方差最大的轴.


    //axis：输出参数，表示选择的分割轴（0 表示 x 轴，1 表示 y 轴，2 表示 z 轴）.
    axis = max_i;

    //th：输出参数，表示分割阈值.
    //将分割阈值设置为该轴的均值.
    th = mean[axis];

    for (const auto &idx : point_idx) {
        if (cloud_[idx][axis] < th) {
            // 中位数可能向左取整.
            left.emplace_back(idx);
        } else {
            right.emplace_back(idx);
        }
    }

    // 边界情况检查：输入的points等于同一个值，上面的判定是>=号，所以都进了右侧.
    // 这种情况不需要继续展开，直接将当前节点设为叶子就行.
    //如果点集大小大于 1，但 left 或 right 为空，说明所有点在分割轴上的值相同，无法分割.
    //返回 false，表示分割失败.
    if (point_idx.size() > 1 && (left.empty() || right.empty())) {
        return false;
    }

    return true;
}

void KdTree::Reset() {
    tree_node_id_ = 0;
    root_.reset(new KdTreeNode());
    root_->id_ = tree_node_id_++;
    size_ = 0;
}

void KdTree::Clear() {
    for (const auto &np : nodes_) {
        if (np.second != root_.get()) {
            delete np.second;
        }
    }

    nodes_.clear();
    root_ = nullptr;
    size_ = 0;
    tree_node_id_ = 0;
}

void KdTree::PrintAll() {
    for (const auto &np : nodes_) {
        auto node = np.second;
        if (node->left_ == nullptr && node->right_ == nullptr) {
            LOG(INFO) << "leaf node: " << node->id_ << ", idx: " << node->point_idx_;
        } else {
            LOG(INFO) << "node: " << node->id_ << ", axis: " << node->axis_index_ << ", th: " << node->split_thresh_;
        }
    }
}

}  // namespace sad
