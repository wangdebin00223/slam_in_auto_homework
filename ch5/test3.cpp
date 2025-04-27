// 在最上方定义了一个适配nanoflann的点云结构体，便于其函数调用
template <typename T>
struct PointCloud_NanoFlann
{
    struct Point
    {
        T x, y, z;
    };
    using coord_t = T;  //!< The type of each coordinate
    std::vector<Point> pts;
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else
            return pts[idx].z;
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }
};


/////////////////////////// kdtree nanoflann测试 1////////////////////////////////
LOG(INFO) << "building kdtree nanflann";
using kdtree_nano = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud_NanoFlann<float>>, 
                                                        PointCloud_NanoFlann<float>, 3>;
kdtree_nano* my_kdtree_nano;
// nanoflann tree
PointCloud_NanoFlann<float> first_cloud_flann;
first_cloud_flann.pts.resize(first->points.size());
for (int i = 0; i < first->points.size(); i++)
{
    first_cloud_flann.pts[i].x = first->points[i].x;
    first_cloud_flann.pts[i].y = first->points[i].y;
    first_cloud_flann.pts[i].z = first->points[i].z;
}
sad::evaluate_and_call([&first, &my_kdtree_nano, &first_cloud_flann]() { 
                            my_kdtree_nano = new kdtree_nano(3, first_cloud_flann, nanoflann::KDTreeSingleIndexAdaptorParams(10));
                            my_kdtree_nano->buildIndex();
                        }, "Kd Tree build", 1);

LOG(INFO) << "searching nanoflann";
matches.clear();
int k = 5; 
std::vector<std::vector<uint32_t>> ret_index_all;
std::vector<std::vector<float>> out_dist_sqr_all;
ret_index_all.resize(second->size());
out_dist_sqr_all.resize(second->size());
sad::evaluate_and_call([&second, &my_kdtree_nano, &matches, &k, &ret_index_all, &out_dist_sqr_all]() {
        // 索引
        std::vector<int> index(second->size());
        for (int i = 0; i < second->points.size(); ++i) {
            index[i] = i;
        }
        
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&second, &my_kdtree_nano, &k, &ret_index_all, &out_dist_sqr_all](int idx) {
            std::vector<uint32_t> ret_index(k);          // 返回的点的索引
            std::vector<float> out_dist_sqr(k);         // 返回的点的距离
            // 获取第二个点云中的每个点，作为当前待查询点
            float query_p[3] = { second->points[idx].x, second->points[idx].y, second->points[idx].z};
            
            // 调用knnSearch()函数，返回最近的5个点的索引和距离
            int num_results = my_kdtree_nano->knnSearch(&query_p[0], k, &ret_index[0], &out_dist_sqr[0]);
            
            ret_index.resize(num_results);
            out_dist_sqr.resize(num_results);

            ret_index_all[idx] = ret_index;
            out_dist_sqr_all[idx] = out_dist_sqr;
            
        });
        // 遍历每个点，获取最近的5个点的索引和距离
        for (int i = 0; i < second->points.size(); i++) {
            // 遍历每个点的最近的5个点
            for (int j = 0; j < ret_index_all[i].size(); ++j) {
                int m = ret_index_all[i][j];         // 最近的5个点的索引
                double d = out_dist_sqr_all[i][j];   // 最近的5个点的距离
                matches.push_back({m, i});          // 将最近的5个点的索引和距离存入matches
            }
        }

    },
    "Kd Tree 5NN in nanoflann", 1);
EvaluateMatches(true_matches, matches);
/////////////////////////// kdtree nanoflann测试 2 ////////////////////////////////
LOG(INFO) << "building kdtree nanflann 2";

kdtree_nano* my_kdtree_nano2;
sad::evaluate_and_call([&first, &my_kdtree_nano2, &first_cloud_flann]() { 
                            my_kdtree_nano2 = new kdtree_nano(3, first_cloud_flann, nanoflann::KDTreeSingleIndexAdaptorParams(10));
                            my_kdtree_nano2->buildIndex();
                        }, "Kd Tree build", 1);
LOG(INFO) << "searching nanoflann 2";
matches.clear();
std::vector<std::vector<size_t>> ret_index_all2;
std::vector<std::vector<float>> out_dist_sqr_all2;
ret_index_all2.resize(second->size());
out_dist_sqr_all2.resize(second->size());
sad::evaluate_and_call([&second, &my_kdtree_nano2, &matches, &k, &ret_index_all2, &out_dist_sqr_all2]() {
        // 索引
        std::vector<int> index(second->size());
        for (int i = 0; i < second->points.size(); ++i) {
            index[i] = i;
        }
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&second, &my_kdtree_nano2, &k, &ret_index_all2, &out_dist_sqr_all2](int idx) {
            
            size_t ret_index[k];          // 返回的点的索引
            float out_dist_sqr[k];         // 返回的点的距离
            // 获取第二个点云中的每个点，作为当前待查询点
            float query_p[3] = { second->points[idx].x, second->points[idx].y, second->points[idx].z};
            nanoflann::KNNResultSet<float> resultSet(k);
            resultSet.init(ret_index, out_dist_sqr);
            my_kdtree_nano2->findNeighbors(resultSet, query_p);

            std::vector<size_t> ret_index_(ret_index, ret_index+k);          // 返回的点的索引
            std::vector<float> out_dist_sqr_(out_dist_sqr, out_dist_sqr+k);  // 返回的点的距离

            ret_index_all2[idx] = ret_index_;
            out_dist_sqr_all2[idx] = out_dist_sqr_;
            
        });
        // 遍历每个点，获取最近的5个点的索引和距离
        for (int i = 0; i < second->points.size(); i++) {
            // 遍历每个点的最近的5个点
            for (int j = 0; j < ret_index_all2[i].size(); ++j) {
                int m = ret_index_all2[i][j];         // 最近的5个点的索引
                double d = out_dist_sqr_all2[i][j];   // 最近的5个点的距离
                matches.push_back({m, i});          // 将最近的5个点的索引和距离存入matches
            }
        }

    },
    "Kd Tree 5NN in nanoflann", 1);
EvaluateMatches(true_matches, matches);