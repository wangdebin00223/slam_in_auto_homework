std::vector<CloudPtr> scans_in_each_line;  // 分线数的点云
CloudPtr full_cloud(new PointCloudType),            // 全点云
         range_cloud(new PointCloudType),           // 距离图像  range image projection
         current_ground(new PointCloudType),        // 当前帧的地面点云【新增】  PCL RANSAC方法
         current_no_ground(new PointCloudType),     // 当前帧的非地面点云【新增】 PCL RANSAC方法
         groundCloud_legoloam(new PointCloudType);  // 地面点云【新增】 LegoLOAM方法

// 遍历所有激光点，按照分配到对应的激光线束中
int k = 0;
unsigned int rowIdn, columnIdn, index;
float verticalAngle, horizonAngle, range;
float azimuth_resolution_deg = 0.3; // 方位角分辨率
float sensorMinimumRange = 1.0;
int Horizon_SCAN = int(360 / azimuth_resolution_deg);  // 水平为360度，按分辨率切分即可，360/0.3=1200

range_cloud->points.resize(num_scans*Horizon_SCAN);

cv::Mat groundMat(num_scans, Horizon_SCAN, CV_8S, cv::Scalar::all(0));

for (auto &pt : pc_in->points) {
    // 点云中已经携带了每个点的线束信息，无需计算，直接读取即可。
    assert(pt.ring >= 0 && pt.ring < num_scans);
    if (rowIdn < 0 || rowIdn >= num_scans) // 0~15
            continue;

    LOG(INFO) << "pt.ring: " << unsigned(pt.ring); // pt.ring 是uint8_t类型，范围是0~255，打印时需要转换为unsigned类型
    PointType p;
    p.x = pt.x; // x,y,z坐标
    p.y = pt.y;
    p.z = pt.z;
    p.intensity = pt.intensity; // intensity存储的是强度
    // 将点按照线束分组，第ring行激光线束的点
    scans_in_each_line[pt.ring]->points.emplace_back(p);
    full_cloud->points.emplace_back(p);  // 用于PCL RANSAC方法

    /// 【LegoLOAM】
    rowIdn = unsigned(pt.ring); // 如果当前点的ring值[0~15]已知，行索引就是ring值

    horizonAngle = atan2(pt.x, pt.y) * 180 / M_PI;
    // LOG(INFO) << "horizonAngle: " << horizonAngle; 

    // 列索引
    columnIdn = -round((horizonAngle-90.0)/azimuth_resolution_deg) + Horizon_SCAN/2;
    // LOG(INFO) << "columnIdn: " << columnIdn; 
    if (columnIdn >= Horizon_SCAN)
        columnIdn -= Horizon_SCAN;

    if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
            continue;

    
    // 计算距离
    range = sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
    if (range < sensorMinimumRange)
            continue;


    // LOG(INFO) << "range: " << range; 
    // 计算出当前点在距离图像中的索引
    index = columnIdn  + rowIdn * Horizon_SCAN;
    // LOG(INFO) << "index: " << index; 
    range_cloud->points[index] = p; // 用于LegoLOAM
    k++;
}

size_t lowerInd, upperInd;
float diffX, diffY, diffZ, angle;

float sensorMountAngle = 0.0;
// groundMat
// -1, no valid info to check if ground of not  无效点
//  0, initial value, after validation, means not ground  表示不是地面点
//  1, ground 地面点标记
for (size_t j = 0; j < Horizon_SCAN; ++j){ // 一根线束上共1800个点
    for (size_t i = 0; i < 7; ++i){ // 对于VLP16线来说，下面7根为地面线束，[0-6]

        lowerInd = j + ( i )*Horizon_SCAN;  // 第 i   根线束的第j个点的索引
        upperInd = j + (i+1)*Horizon_SCAN;  // 第 i+1 根线束的第j个点的索引

        if (range_cloud->points[lowerInd].intensity == -1 ||
            range_cloud->points[upperInd].intensity == -1){
            // no info to check, invalid points
            groundMat.at<int8_t>(i,j) = -1; // 无效点
            continue;
        }
            
        // 上下两根激光线束扫到的垂直相邻点的坐标差
        diffX = range_cloud->points[upperInd].x - range_cloud->points[lowerInd].x;
        diffY = range_cloud->points[upperInd].y - range_cloud->points[lowerInd].y;
        diffZ = range_cloud->points[upperInd].z - range_cloud->points[lowerInd].z;

        // 计算相邻两根线束的点云来计算水平夹角，水平夹角在特定范围内，则认为是地面点
        angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;
        if (abs(angle - sensorMountAngle) <= 10){
            // 那么，这两根相邻线束上相同索引处扫到点，就是地面点，标记为1
            groundMat.at<int8_t>(i,j) = 1;      // 第i根线束的第j个点是地面点
            groundMat.at<int8_t>(i+1,j) = 1;    // 第i+1根线束的第j个点是地面点
        }
    }
}

// extract ground cloud (groundMat == 1) 
// mark entry that doesn't need to label (ground and invalid point) for segmentation
// note that ground remove is from 0~N_SCAN-1, need rangeMat for mark label matrix for the 16th scan
// 遍历16线激光线束的底下的7根线束，将这7根线束上标记为1的地面点提取出来，放到groundCloud中
for (size_t i = 0; i <= 7; ++i){    // 7根地面线束
    for (size_t j = 0; j < Horizon_SCAN; ++j){  // 1800个点
        // 判断第i根线束上的第j个点是否是地面点
        if (groundMat.at<int8_t>(i,j) == 1)
            // 如果是地面点，则放到groundCloud中
            groundCloud_legoloam->push_back(range_cloud->points[j + i*Horizon_SCAN]);
    }
}

LOG(INFO) << "ground: " << groundCloud_legoloam->size();