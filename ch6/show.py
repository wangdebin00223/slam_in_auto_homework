import numpy as np
import matplotlib.pyplot as plt

# 假设数据
pt_angle = 30.0  # 模板点相对于某个固定方向（比如车体的前方）的角度（α）
theta = 45.0     # 激光雷达帧相对于同一固定方向的旋转角度（θ）

# 激光雷达扫描数据
num_points = 360  # 扫描点数
scan_angles = np.linspace(0, 360 - 360/num_points, num_points)  # 扫描角度，从0到359度（闭环）
# 为了简化，我们假设所有扫描点的距离都是相同的（形成一个圆），但实际应用中距离会不同
scan_radius = 10.0  # 假设扫描半径为10单位
scan_ranges = np.full(num_points, scan_radius)  # 距离数组，全部填充为扫描半径

# 计算相对角度
relative_angle = pt_angle - theta

# 将角度转换为弧度，因为matplotlib的极坐标图使用弧度
relative_angle_rad = np.deg2rad(relative_angle)
scan_angles_rad = np.deg2rad(scan_angles)

# 找到扫描数据中与相对角度最接近的点
closest_index = np.argmin(np.abs(scan_angles_rad - relative_angle_rad))
closest_angle_deg = np.rad2deg(scan_angles_rad[closest_index])
closest_range = scan_ranges[closest_index]

# 绘制图形
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# 绘制激光雷达扫描数据
ax.plot(scan_angles_rad, scan_ranges, label='激光雷达扫描数据')

# 绘制模板点的相对位置（以雷达为中心，但角度是相对的）
# 注意：这里我们只是为了示意，所以画了一个从雷达中心出发的线段，长度并不代表实际距离
ax.plot([0, relative_angle_rad], [0, 1], 'ro-', label='模板点相对位置')  # 'ro-'表示红色圆点和线

# 添加文本注释
ax.text(relative_angle_rad, 1.2, f'相对角度: {relative_angle:.1f}°', ha='center', va='bottom')
ax.text(np.deg2rad(closest_angle_deg), closest_range + 0.5, f'最接近点: {closest_angle_deg:.1f}°, 距离: {closest_range:.1f}单位', ha='center', va='bottom')

# 设置图形标题和标签
ax.set_title('激光雷达扫描与模板点关系示意图')
ax.set_xlabel('角度 (度)')
ax.set_ylabel('距离 (单位)')

# 显示图例
ax.legend()

# 显示网格线
ax.grid(True)

# 显示图形
plt.show()