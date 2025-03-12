// error state 递推
// 计算运动过程雅可比矩阵 F，见公式(3.47)
// F实际上是稀疏矩阵，也可以不用矩阵形式进行相乘而是写成散装形式，这里为了教学方便，使用矩阵形式
Mat18T F = Mat18T::Identity();                                                 // 主对角线
F.template block<3, 3>(0, 3) = Mat3T::Identity() * dt;                         // p 对 v
F.template block<3, 3>(3, 6) = -R_.matrix() * SO3::hat(imu.acce_ - ba_) * dt;  // v 对 theta
F.template block<3, 3>(3, 12) = -R_.matrix() * dt;                             // v 对 ba
F.template block<3, 3>(3, 15) = Mat3T::Identity() * dt;                        // v 对 g
F.template block<3, 3>(6, 6) = SO3::exp(-(imu.gyro_ - bg_) * dt).matrix();     // theta 对 theta
F.template block<3, 3>(6, 9) = -Mat3T::Identity() * dt;                        // theta 对 bg

// 【第二题的实现】
Vec18T dx_new_ = Vec18T::Zero();
auto starttime = system_clock::now(); 
dx_new_.template block<3, 1>(0, 0)  = dx_.template block<3, 1>(0, 0) 
                                    + dx_.template block<3, 1>(3, 0) * dt;

dx_new_.template block<3, 1>(3, 0)  = dx_.template block<3, 1>(3, 0) 
                                    + F.template block<3, 3>(3, 6) * dx_.template block<3, 1>(6, 0)
                                    + F.template block<3, 3>(3, 12) * dx_.template block<3, 1>(12, 0)
                                    + dx_.template block<3, 1>(15, 0) * dt; 

dx_new_.template block<3, 1>(6, 0)  = F.template block<3, 3>(6, 6) * dx_.template block<3, 1>(6, 0)
                                    - dx_.template block<3, 1>(9, 0) * dt;

// dx_的其余三个状态（delta bg, delta ba, delta g）保持不变
dx_new_.template block<3, 1>(9, 0)  = dx_.template block<3, 1>(9, 0);
dx_new_.template block<3, 1>(12, 0) = dx_.template block<3, 1>(12, 0);
dx_new_.template block<3, 1>(15, 0) = dx_.template block<3, 1>(15, 0);

duration<double> diff = system_clock::now()- starttime; // 【计算分块计算形式的耗时】                          
LOG(INFO) << "\ntiming_dx_new_ = "<<diff.count();       
// mean and cov prediction
starttime = system_clock::now();
dx_ = F * dx_;  // 这行其实没必要算，dx_在重置之后应该为零，因此这步可以跳过，但F需要参与Cov部分计算，所以保留
diff = system_clock::now()- starttime;                  // 【计算矩阵相乘形式的耗时】
LOG(INFO) << "\ntiming_dx_ = "<<diff.count();  