// 从github上克隆代码到本地，其实只有一个hpp文件，
// 将其与第5章的代码放在同一级目录下，使用起来非常方便，只需在http://test_nn.cc文件中引入该头文件即可使用：
// #include "ch5/nanoflann.hpp"
template <typename num_t>
void kdtree_demo(const size_t N)
{
    PointCloud<num_t> cloud;    // 创建点云对象
    // Generate points:
    // 生成随机点云
    generateRandomPointCloud(cloud, N);
    // construct a kd-tree index 使用using定义kd树类型
    using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<num_t, PointCloud<num_t>>,
                                                             PointCloud<num_t>, 3 /* dim */>;
    // 构建kd树索引
    my_kd_tree_t index(3 /*dim*/,               // 3维空间
                       cloud,                   // 点云对象 
                       {10 /* max leaf */});    // 最大叶子节点数
    // 待查询点
    const num_t query_pt[3] = {0.5, 0.5, 0.5};
    // ----------------------------------------------------------------
    // knnSearch():  Perform a search for the N closest points 
    //               执行搜索以查找最近的N个点
    // ----------------------------------------------------------------
    {
        size_t                num_results = 5;              // 最近的5个点
        std::vector<uint32_t> ret_index(num_results);       // 返回的点的索引
        std::vector<num_t>    out_dist_sqr(num_results);    // 返回的点的距离
        // 调用knnSearch()函数，返回最近的5个点的索引和距离
        num_results = index.knnSearch(&query_pt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
        // In case of less points in the tree than requested: 如果树中的点少于请求的点，则重新调整返回的点的索引和距离容器的大小
        ret_index.resize(num_results);
        out_dist_sqr.resize(num_results);
        cout << "knnSearch(): num_results=" << num_results << "\n";
        for (size_t i = 0; i < num_results; i++)
            cout << "idx[" << i << "]=" << ret_index[i] << " dist[" << i << "]=" << out_dist_sqr[i] << endl;
        cout << "\n";
    }
    // ----------------------------------------------------------------
    // radiusSearch(): Perform a search for the points within search_radius
    //                  执行搜索以查找search_radius半径内的点
    // ----------------------------------------------------------------
    {
        const num_t search_radius = static_cast<num_t>(0.1);
        std::vector<nanoflann::ResultItem<uint32_t, num_t>> ret_matches;
        // nanoflanSearchParamsameters params;
        // params.sorted = false;

        // 调用radiusSearch()函数，返回指定搜索半径内的点的索引和距离
        const size_t nMatches = index.radiusSearch(&query_pt[0], search_radius, ret_matches);
        cout << "radiusSearch(): radius=" << search_radius << " -> " << nMatches << " matches\n";
        for (size_t i = 0; i < nMatches; i++)
            cout << "idx[" << i << "]=" << ret_matches[i].first << " dist[" << i << "]=" << ret_matches[i].second << endl;
        cout << "\n";
    }
}


// 在另一个例程pointcloud_example.cpp中，给出了另一种近邻搜索的使用方法，就是利用findNeighbors()函数，对应函数说明如下图所示：
template <typename num_t>
void kdtree_demo(const size_t N)
{
    static_assert(std::is_standard_layout<nanoflann::ResultItem<num_t, size_t>>::value, "Unexpected memory layout for nanoflann::ResultItem");
    PointCloud<num_t> cloud;
    // Generate points:
    generateRandomPointCloud(cloud, N);
    num_t query_pt[3] = {0.5, 0.5, 0.5};
    // construct a kd-tree index:
    using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<num_t, PointCloud<num_t>>,
        PointCloud<num_t>, 3 /* dim */>;
    dump_mem_usage();
    my_kd_tree_t index(3 /*dim*/, cloud, {10 /* max leaf */});
    dump_mem_usage();
    {
        // do a knn search
        const size_t                   num_results = 1;
        size_t                         ret_index;
        num_t                          out_dist_sqr;
        nanoflann::KNNResultSet<num_t> resultSet(num_results);
        resultSet.init(&ret_index, &out_dist_sqr);
        index.findNeighbors(resultSet, &query_pt[0]);
        std::cout << "knnSearch(nn=" << num_results << "): \n";
        std::cout << "ret_index=" << ret_index
                  << " out_dist_sqr=" << out_dist_sqr << std::endl;
    }
    {
        // radius search:
        const num_t                                       squaredRadius = 1;
        std::vector<nanoflann::ResultItem<size_t, num_t>> indices_dists;
        nanoflann::RadiusResultSet<num_t, size_t>         resultSet(squaredRadius, indices_dists);
        index.findNeighbors(resultSet, query_pt);
        // Get worst (furthest) point, without sorting:
        nanoflann::ResultItem<size_t, num_t> worst_pair = resultSet.worst_item();
        std::cout << "Worst pair: idx=" << worst_pair.first
                  << " squaredDist=" << worst_pair.second << std::endl;
    }
}