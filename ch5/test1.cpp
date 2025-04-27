// src/ch5/gridnn.hpp
// 枚举类型，定义近邻关系
enum class NearbyType {
    CENTER,  // 只考虑中心

    // for 2D
    NEARBY4,  // 上下，左右           
    NEARBY8,  // 上下左右四角         

    // for 3D
    NEARBY6,  // 上下左右前后，不包含8个角
    NEARBY14, // 上下左右前后，包含8个角【  新增】
};


// src/ch5/gridnn.hpp
template <>
void GridNN<3>::GenerateNearbyGrids() {
    if (nearby_type_ == NearbyType::CENTER) {
        nearby_grids_.emplace_back(KeyType::Zero());
    } else if (nearby_type_ == NearbyType::NEARBY6) {
        nearby_grids_ = {KeyType( 0,  0,  0),  
                         KeyType(-1,  0,  0),   KeyType(1,  0,  0),   // 左右
                         KeyType( 0,  1,  0),   KeyType(0, -1,  0),   // 前后
                         KeyType( 0,  0, -1),   KeyType(0,  0,  1)};  // 上下
    } else if (nearby_type_ == NearbyType::NEARBY14) {
        // 【新增】
        nearby_grids_ = {KeyType( 0,  0,  0),  
                         KeyType(-1,  0,  0),   KeyType( 1,  0,  0),   // 左右
                         KeyType( 0,  1,  0),   KeyType( 0, -1,  0),   // 前后
                         KeyType( 0,  0, -1),   KeyType( 0,  0,  1),   // 上下
                         KeyType(-1, -1, -1),   KeyType(-1,  1, -1),   // 八个角
                         KeyType(-1, -1,  1),   KeyType(-1,  1,  1), 
                         KeyType( 1, -1, -1),   KeyType( 1,  1, -1), 
                         KeyType( 1, -1,  1),   KeyType( 1,  1,  1)};
    }
}


// src/ch5/http://test_nn.cc
    ...
    // 对比不同种类的grid
    sad::GridNN<2>  grid2_0(0.1, sad::GridNN<2>::NearbyType::CENTER), 
                    grid2_4(0.1, sad::GridNN<2>::NearbyType::NEARBY4),
                    grid2_8(0.1, sad::GridNN<2>::NearbyType::NEARBY8);
    sad::GridNN<3>  grid3_0(0.1, sad::GridNN<3>::NearbyType::CENTER),       // 【新增 by ClarkWang】
                    grid3_6(0.1, sad::GridNN<3>::NearbyType::NEARBY6),
                    grid3_14(0.1, sad::GridNN<3>::NearbyType::NEARBY14);    // 【新增 by ClarkWang】

    grid2_0.SetPointCloud(first);
    grid2_4.SetPointCloud(first);
    grid2_8.SetPointCloud(first);
    
    grid3_0.SetPointCloud(first);       // 【新增 by ClarkWang】
    grid3_6.SetPointCloud(first);
    grid3_14.SetPointCloud(first);      // 【新增 by ClarkWang】
    ...

    LOG(INFO) << "===================";
    sad::evaluate_and_call(
        [&first, &second, &grid3_0, &matches]() { grid3_0.GetClosestPointForCloud(first, second, matches); },
        "Grid0 3D 单线程", 10); // 【新增 by ClarkWang】
    EvaluateMatches(truth_matches, matches);
    LOG(INFO) << "===================";
    sad::evaluate_and_call( 
        [&first, &second, &grid3_0, &matches]() { grid3_0.GetClosestPointForCloudMT(first, second, matches); },
        "Grid0 3D 多线程", 10); // 【新增 by ClarkWang】
    EvaluateMatches(truth_matches, matches);
    LOG(INFO) << "===================";
    sad::evaluate_and_call(
        [&first, &second, &grid3_6, &matches]() { grid3_6.GetClosestPointForCloud(first, second, matches); },
        "Grid6 3D 单线程", 10); 
    EvaluateMatches(truth_matches, matches);
    LOG(INFO) << "===================";
    sad::evaluate_and_call(
        [&first, &second, &grid3_6, &matches]() { grid3_6.GetClosestPointForCloudMT(first, second, matches); },
        "Grid6 3D 多线程", 10);
    EvaluateMatches(truth_matches, matches);
    LOG(INFO) << "===================";
    sad::evaluate_and_call(
        [&first, &second, &grid3_14, &matches]() { grid3_14.GetClosestPointForCloud(first, second, matches); },
        "Grid14 3D 单线程", 10);    // 【新增 by ClarkWang】
    EvaluateMatches(truth_matches, matches);
    LOG(INFO) << "===================";
    sad::evaluate_and_call(
        [&first, &second, &grid3_14, &matches]() { grid3_14.GetClosestPointForCloudMT(first, second, matches); },
        "Grid14 3D 多线程", 10);    // 【新增 by ClarkWang】
    EvaluateMatches(truth_matches, matches);
    ...