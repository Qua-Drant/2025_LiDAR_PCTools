#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <memory>
#include <string>
#include <set>
#include <map>

#include <Eigen/Dense> // Eigen核心功能，用于矩阵和向量操作

#include <pcl/io/pcd_io.h> // PCL点云文件输入输出
#include <pcl/point_types.h> // PCL点类型定义
#include <pcl/point_cloud.h> // PCL点云数据结构
#include <pcl/filters/voxel_grid.h> // PCL体素下采样滤波器
#include <pcl/features/normal_3d_omp.h> // PCL法线估计 (OMP版本，用于潜在加速)
#include <pcl/segmentation/conditional_euclidean_clustering.h> // PCL条件欧几里得聚类
#include <pcl/segmentation/extract_clusters.h> // PCL从索引中提取点集（用于聚类提取）
#include <pcl/common/centroid.h> // PCL计算点云质心
#include <pcl/filters/extract_indices.h> // PCL按索引提取点云
#include <pcl/common/common.h> // PCL通用功能，例如 getMinMax3D

#include <opencv2/opencv.hpp> // OpenCV主要头文件 (包含了下面几个)
// #include <opencv2/imgproc.hpp> // OpenCV图像处理功能
// #include <opencv2/highgui.hpp>  // OpenCV图像显示和保存功能

// 如果未定义M_PI (例如，在MSVC中没有定义_USE_MATH_DEFINES)，则定义它
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 使用标准命名空间和OpenCV命名空间
// using namespace std; // 在.cpp文件中可以接受，但大型项目中需谨慎
// using namespace cv;  // 同上

// 定义处理上下文结构体，用于管理全局参数
struct ProcessingContext {
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    float min_z_orig = std::numeric_limits<float>::max(); // 点云原始最小Z值
    float chm_resolution = 0.2f; // CHM分辨率，默认为0.2米

    int chm_width = 0;
    int chm_height = 0;

    void updateBounds(const pcl::PointCloud<pcl::PointXYZL>& cloud) {
        pcl::PointXYZL min_pt, max_pt;
        pcl::getMinMax3D(cloud, min_pt, max_pt);
        min_x = min_pt.x;
        max_x = max_pt.x;
        min_y = min_pt.y;
        max_y = max_pt.y;
        min_z_orig = min_pt.z; // 注意：这里假设所有点的Z值都是有效的，getMinMax3D会找到实际的最小Z
    }

    void calculateChmDimensions() {
        if (max_x > min_x && max_y > min_y && chm_resolution > 0) {
            chm_width = static_cast<int>(std::ceil((max_x - min_x) / chm_resolution));
            chm_height = static_cast<int>(std::ceil((max_y - min_y) / chm_resolution));
        }
        else {
            chm_width = 0;
            chm_height = 0;
        }
    }
};


/**
 * @brief 条件欧几里得聚类的自定义条件函数。
 * 目的是聚类属于垂直杆状结构（树干）的点。
 * @param pointA 当前簇中的点（或种子点）。
 * @param pointB 候选邻近点。
 * @param squared_distance pointA 和 pointB 之间的欧氏距离的平方。
 * @return 如果pointB应添加到pointA的簇中，则返回true，否则返回false。
 */
bool trunkCondition(const pcl::PointXYZINormal& pointA, const pcl::PointXYZINormal& pointB, float squared_distance) {
    Eigen::Map<const Eigen::Vector3f> normalA = pointA.getNormalVector3fMap();
    Eigen::Map<const Eigen::Vector3f> normalB = pointB.getNormalVector3fMap();

    // 常量定义
    const float MAX_NORMAL_Z_COMPONENT = std::sin(25.0f * M_PI / 180.0f); // sin(25 deg) approx 0.422
    const float MIN_NORMAL_DOT_PRODUCT = std::cos(30.0f * M_PI / 180.0f); // cos(30 deg) approx 0.866
    const float MAX_COS_ANGLE_CONNECTION_NORMAL = std::cos(70.0f * M_PI / 180.0f); // cos(70 deg) approx 0.342
    const float MIN_SQUARED_NORM_FOR_CONNECTION = 1e-6f;


    // 条件1：法线应主要为水平方向（表面是垂直的）。
    // 法线的Z分量应该很小。例如，与XY平面的夹角 < 25度。
    if (std::abs(normalA[2]) > MAX_NORMAL_Z_COMPONENT || std::abs(normalB[2]) > MAX_NORMAL_Z_COMPONENT) {
        return false;
    }

    // 条件2：法线应大致平行。
    // 法线之间的角度 < 30度 (cos(angle) > cos(30 deg))。
    if (normalA.dot(normalB) < MIN_NORMAL_DOT_PRODUCT) {
        return false;
    }

    // 条件3：点之间的连接向量应与表面法线大致正交。
    // (连接向量与normalA之间的角度 > 70度，即 cos(angle) < cos(70 deg))。
    // 这鼓励沿着表面生长，而不是垂直于表面。
    Eigen::Vector3f connection_vector = pointB.getVector3fMap() - pointA.getVector3fMap();
    if (connection_vector.squaredNorm() < MIN_SQUARED_NORM_FOR_CONNECTION) { // 避免点完全相同导致的问题
        return true; // 如果点相同，可以认为满足条件
    }
    connection_vector.normalize(); // 对于点积作为余弦值至关重要

    // 我们使用abs，因为normalA可能指向内部或外部。
    // 我们希望连接向量和法线之间的角度较大。
    // 因此，它们的点积（角度的余弦值）应该较小。
    if (std::abs(connection_vector.dot(normalA)) > MAX_COS_ANGLE_CONNECTION_NORMAL) {
        return false;
    }

    return true;
}


/**
 * @brief 在CHM中围绕给定的栅格坐标查找最高点。
 * @param topPointsMat 冠层高度模型 (CV_32F, 归一化高度)。
 * @param base_grid_coords 树干基部在CHM中的栅格坐标 (列, 行)。
 * @param highest_point 输出：找到的最高点的世界坐标 (x, y, 归一化高度)。
 * @param context 包含CHM分辨率和全局边界的上下文信息。
 * @param search_radius_m 围绕base_grid_coords的搜索半径（米）。
 */
void get_highest_point_from_chm(const cv::Mat& topPointsMat,
    const Eigen::Vector2i& base_grid_coords,
    Eigen::Vector3f& highest_point,
    const ProcessingContext& context,
    float search_radius_m = 2.0f) {
    int chm_h = topPointsMat.rows;
    int chm_w = topPointsMat.cols;
    float max_h_val = std::numeric_limits<float>::lowest();

    int x0_grid = base_grid_coords[0]; // 列
    int y0_grid = base_grid_coords[1]; // 行

    // 将搜索半径从米转换为栅格单位
    int search_radius_grid = static_cast<int>(std::ceil(search_radius_m / context.chm_resolution));

    highest_point = Eigen::Vector3f(0, 0, 0); // 如果未找到合适的点，则为默认值

    for (int r_offset = -search_radius_grid; r_offset <= search_radius_grid; ++r_offset) {
        for (int c_offset = -search_radius_grid; c_offset <= search_radius_grid; ++c_offset) {
            if (r_offset * r_offset + c_offset * c_offset > search_radius_grid * search_radius_grid) {
                continue; // 超出圆形搜索半径
            }

            int r = y0_grid + r_offset;
            int c = x0_grid + c_offset;

            if (r >= 0 && r < chm_h && c >= 0 && c < chm_w) {
                float current_h_val = topPointsMat.at<float>(r, c);
                if (current_h_val > max_h_val) {
                    max_h_val = current_h_val;
                    highest_point[0] = context.min_x + (c + 0.5f) * context.chm_resolution; // 世界坐标X
                    highest_point[1] = context.min_y + (r + 0.5f) * context.chm_resolution; // 世界坐标Y
                    highest_point[2] = current_h_val; // CHM中的归一化高度
                }
            }
        }
    }
    // 如果在半径内未找到点（例如在边缘，或所有CHM值都为0/很低），
    // 并且base_grid_coords本身有效，则使用其CHM高度。
    if (max_h_val == std::numeric_limits<float>::lowest() &&
        y0_grid >= 0 && y0_grid < chm_h && x0_grid >= 0 && x0_grid < chm_w) {
        max_h_val = topPointsMat.at<float>(y0_grid, x0_grid);
        highest_point[0] = context.min_x + (x0_grid + 0.5f) * context.chm_resolution;
        highest_point[1] = context.min_y + (y0_grid + 0.5f) * context.chm_resolution;
        highest_point[2] = max_h_val;
    }
}


int main() {
    // --- 配置参数 ---
    std::string pcd_input_path = "D:/Experiments_2025_spring/2025LiDAR/CourseWorkData/Test/trees.pcd"; // 输入: 您的PCD文件路径
    std::string pcd_output_path = "D:/LidarSet/labeled_trees_cec_v2.pcd";       // 输出: 标记后的PCD文件路径
    std::string preview_output_path = "D:/LidarSet/segmentation_preview_cec_v2.png"; // 输出: 分割预览图像路径
    std::string chm_debug_output_path = "D:/LidarSet/chm_debug_cec_v2.png";      // 输出: CHM调试图像路径

    ProcessingContext context; // 创建处理上下文实例
    context.chm_resolution = 0.2f; // CHM栅格单元大小（米）(可调参数)

    // 法线估计参数
    const int NE_K_SEARCH = 20; // K近邻数量 (可调参数)
    // const float NE_RADIUS_SEARCH = 0.3f; // 另一种选择: 半径搜索

    // 条件欧几里得聚类 (CEC) 参数
    const float CEC_CLUSTER_TOLERANCE = 0.35f; // 树干簇中点之间的最大距离（米）(可调参数)
    const int CEC_MIN_CLUSTER_SIZE = 25;      // 有效树干簇的最小点数 (可调参数)
    const int CEC_MAX_CLUSTER_SIZE = 1500;    // 有效树干簇的最大点数 (可调参数)

    // 树干簇过滤参数
    const float TRUNK_FILTER_MIN_HEIGHT = 1.5f;       // 树干簇的最小物理高度（米）(可调参数)
    const float TRUNK_FILTER_MIN_ELONGATION_RATIO = 2.0f; // 高度与最大宽度的最小比率 (可调参数)
    const float MIN_CLUSTER_WIDTH_FOR_RATIO = 1e-3f; // 用于避免除零的最小宽度

    // 从CHM中寻找树顶
    const float TREETOP_SEARCH_RADIUS_M = 2.5f; // 在树干基部周围CHM中的搜索半径（米）(可调参数)
    const float MIN_TREETOP_NORMALIZED_HEIGHT = 1.0f; // 树顶最小归一化高度
    const float MIN_DIST_SQ_BETWEEN_TREETOPS = 1.0f * 1.0f; // 树顶点之间最小距离的平方 (1米)

    // 像素所有权参数
    const float CROWN_RADIUS_FACTOR = 0.7f; // 影响半径 = 树高 * 因子 (可调参数)
    const float CROWN_MIN_RADIUS_PIXELS = 2.0f; // CHM像素中的最小影响半径
    const float MIN_CHM_HEIGHT_FOR_OWNERSHIP = 0.1f; // 像素具有所有权的最小CHM高度
    const float MAX_INFLUENCE_RADIUS_FACTOR_FOR_REFINEMENT = 1.0f; // 细化时最大影响半径因子 (树高的百分比)


    // --- 加载点云 ---
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZL>);
    std::cout << "加载点云: " << pcd_input_path << std::endl;
    if (pcl::io::loadPCDFile<pcl::PointXYZL>(pcd_input_path, *cloud) == -1) {
        PCL_ERROR("无法读取文件 %s\n", pcd_input_path.c_str());
        return (-1);
    }
    std::cout << "已加载 " << cloud->width * cloud->height << " 个数据点。" << std::endl;
    if (cloud->empty()) {
        PCL_ERROR("输入点云为空。\n");
        return -1;
    }

    // --- 计算全局点云范围和CHM维度 ---
    context.updateBounds(*cloud);
    context.calculateChmDimensions();

    if (context.min_x == std::numeric_limits<float>::max() || cloud->points.size() < 10) {
        std::cerr << "错误: 点云边界无效或点数过少。" << std::endl;
        return -1;
    }
    if (context.chm_width <= 0 || context.chm_height <= 0) {
        std::cerr << "错误: CHM维度无效 (" << context.chm_width << "x" << context.chm_height << ")。请检查分辨率或点云范围。" << std::endl;
        return -1;
    }
    std::cout << "CHM维度: " << context.chm_width << "x" << context.chm_height << " (分辨率: " << context.chm_resolution << "m)" << std::endl;

    // --- 创建冠层高度模型 (CHM) ---
    std::cout << "创建冠层高度模型 (CHM)..." << std::endl;
    cv::Mat topPointsMat = cv::Mat::zeros(context.chm_height, context.chm_width, CV_32F);
    for (const auto& pt_orig : cloud->points) {
        float normalized_z = pt_orig.z - context.min_z_orig; // 高于点云最低点的Z值
        int ix = static_cast<int>((pt_orig.x - context.min_x) / context.chm_resolution);
        int iy = static_cast<int>((pt_orig.y - context.min_y) / context.chm_resolution);

        if (ix >= 0 && ix < context.chm_width && iy >= 0 && iy < context.chm_height) {
            topPointsMat.at<float>(iy, ix) = std::max(topPointsMat.at<float>(iy, ix), normalized_z);
        }
    }
    std::cout << "CHM创建完成。" << std::endl;

    // --- 使用条件欧几里得聚类进行树干检测 ---
    std::cout << "开始使用CEC进行树干检测..." << std::endl;

    // 2. 估计法线
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZL>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZL>());
    pcl::NormalEstimationOMP<pcl::PointXYZL, pcl::Normal> ne; // 使用OMP版本
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud_downsampled);
    ne.setKSearch(NE_K_SEARCH);
    // ne.setRadiusSearch(NE_RADIUS_SEARCH); // 备选方案
    ne.compute(*normals);
    std::cout << "  法线估计完成。" << std::endl;

    // 3. 合并点和法线
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::concatenateFields(*cloud_downsampled, *normals, *cloud_with_normals);
    std::cout << "  点和法线已合并。" << std::endl;

    // 4. 条件欧几里得聚类
    pcl::ConditionalEuclideanClustering<pcl::PointXYZINormal> cec(true); // true表示强制执行条件
    cec.setInputCloud(cloud_with_normals);
    cec.setConditionFunction(&trunkCondition);
    cec.setClusterTolerance(CEC_CLUSTER_TOLERANCE);
    cec.setMinClusterSize(CEC_MIN_CLUSTER_SIZE);
    cec.setMaxClusterSize(CEC_MAX_CLUSTER_SIZE);

    std::vector<pcl::PointIndices> cluster_indices_cec;
    cec.segment(cluster_indices_cec);
    std::cout << "  CEC找到 " << cluster_indices_cec.size() << " 个初始簇。" << std::endl;

    // 5. 过滤树干簇并推导树顶
    std::vector<Eigen::Vector3f> treetops; // 存储 [世界坐标X, 世界坐标Y, 归一化CHM高度]
    for (const auto& indices : cluster_indices_cec) {
        if (indices.indices.empty()) continue;

        pcl::PointCloud<pcl::PointXYZL>::Ptr trunk_candidate_pts(new pcl::PointCloud<pcl::PointXYZL>);
        pcl::ExtractIndices<pcl::PointXYZL> extract; // 使用PointXYZL进行几何分析
        extract.setInputCloud(cloud_downsampled);    // 从用于法线估计的同一片点云中提取
        extract.setIndices(std::make_shared<pcl::PointIndices>(indices));
        extract.filter(*trunk_candidate_pts);

        if (trunk_candidate_pts->empty()) continue;

        // 根据物理属性过滤
        Eigen::Vector4f min_pt_cluster_eigen, max_pt_cluster_eigen; // PCL::getMinMax3D 使用 Eigen::Vector4f
        pcl::getMinMax3D(*trunk_candidate_pts, min_pt_cluster_eigen, max_pt_cluster_eigen);

        float cluster_height_m = max_pt_cluster_eigen[2] - min_pt_cluster_eigen[2]; // 物理高度
        if (cluster_height_m < TRUNK_FILTER_MIN_HEIGHT) continue;

        float cluster_width_x_m = max_pt_cluster_eigen[0] - min_pt_cluster_eigen[0];
        float cluster_width_y_m = max_pt_cluster_eigen[1] - min_pt_cluster_eigen[1];
        float cluster_max_width_m = std::max(cluster_width_x_m, cluster_width_y_m);
        if (cluster_max_width_m < MIN_CLUSTER_WIDTH_FOR_RATIO) cluster_max_width_m = MIN_CLUSTER_WIDTH_FOR_RATIO; // 避免除零
        if ((cluster_height_m / cluster_max_width_m) < TRUNK_FILTER_MIN_ELONGATION_RATIO) continue;

        // 计算树干候选簇的质心 (世界坐标)
        Eigen::Vector4f centroid_trunk_4f;
        pcl::compute3DCentroid(*trunk_candidate_pts, centroid_trunk_4f);
        Eigen::Vector3f centroid_trunk_3f(centroid_trunk_4f[0], centroid_trunk_4f[1], centroid_trunk_4f[2]);

        // 将树干质心 (世界坐标) 转换为CHM栅格坐标 (列, 行)
        int cx_grid = static_cast<int>((centroid_trunk_3f[0] - context.min_x) / context.chm_resolution);
        int cy_grid = static_cast<int>((centroid_trunk_3f[1] - context.min_y) / context.chm_resolution);

        if (cx_grid < 0 || cx_grid >= context.chm_width || cy_grid < 0 || cy_grid >= context.chm_height) continue;

        Eigen::Vector3f treetop_candidate;
        get_highest_point_from_chm(topPointsMat, Eigen::Vector2i(cx_grid, cy_grid), treetop_candidate, context, TREETOP_SEARCH_RADIUS_M);

        if (treetop_candidate[2] > MIN_TREETOP_NORMALIZED_HEIGHT) { // 确保树顶有一定高度 (归一化CHM高度 > 1m)
            bool too_close = false;
            for (const auto& existing_tt : treetops) {
                float dx = existing_tt[0] - treetop_candidate[0];
                float dy = existing_tt[1] - treetop_candidate[1];
                if (dx * dx + dy * dy < MIN_DIST_SQ_BETWEEN_TREETOPS) { // 如果与现有树顶距离小于1米
                    too_close = true;
                    break;
                }
            }
            if (!too_close) {
                treetops.push_back(treetop_candidate);
            }
        }
    }
    std::cout << "过滤后得到 " << treetops.size() << " 个树顶。" << std::endl;
    if (treetops.empty()) {
        std::cout << "未找到有效树顶。正在退出。" << std::endl;
        // 可以选择保存CHM或空的预览图像
        // imwrite(chm_debug_output_path, ...);
        // imwrite(preview_output_path, ...);
        return 0;
    }

    // --- CHM中的像素所有权分配 ---
    std::cout << "在CHM中分配像素所有权..." << std::endl;
    cv::Mat pixel_owner = cv::Mat(context.chm_height, context.chm_width, CV_32S, cv::Scalar(-1)); // CV_32S用于整数标签, 初始化为-1 (未分配)

    // 基于影响半径的初始分配
    for (int i = 0; i < treetops.size(); ++i) {
        float tree_height_norm = treetops[i][2]; // CHM中的归一化高度
        float influence_radius_m = tree_height_norm * CROWN_RADIUS_FACTOR;
        int influence_radius_pix = static_cast<int>(std::ceil(influence_radius_m / context.chm_resolution));
        influence_radius_pix = std::max(influence_radius_pix, static_cast<int>(CROWN_MIN_RADIUS_PIXELS));

        int cx_tree_pix = static_cast<int>((treetops[i][0] - context.min_x) / context.chm_resolution);
        int cy_tree_pix = static_cast<int>((treetops[i][1] - context.min_y) / context.chm_resolution);

        for (int r_offset = -influence_radius_pix; r_offset <= influence_radius_pix; ++r_offset) {
            for (int c_offset = -influence_radius_pix; c_offset <= influence_radius_pix; ++c_offset) {
                if (r_offset * r_offset + c_offset * c_offset > influence_radius_pix * influence_radius_pix) continue;

                int y_pix = cy_tree_pix + r_offset;
                int x_pix = cx_tree_pix + c_offset;

                if (x_pix >= 0 && x_pix < context.chm_width && y_pix >= 0 && y_pix < context.chm_height) {
                    if (topPointsMat.at<float>(y_pix, x_pix) > MIN_CHM_HEIGHT_FOR_OWNERSHIP) { // 仅分配给有一定高度的像素
                        // 简单分配：如果未分配或当前树“更好”（例如更近、更高 - 为简单起见此处未实现该逻辑）
                        if (pixel_owner.at<int>(y_pix, x_pix) == -1) {
                            pixel_owner.at<int>(y_pix, x_pix) = i; // 分配树的索引
                        }
                    }
                }
            }
        }
    }

    // 细化：将每个相关像素分配给其潜在影响范围内的最近树顶
    std::cout << "细化像素所有权..." << std::endl;
    for (int y_pix = 0; y_pix < context.chm_height; ++y_pix) {
        for (int x_pix = 0; x_pix < context.chm_width; ++x_pix) {
            if (topPointsMat.at<float>(y_pix, x_pix) <= MIN_CHM_HEIGHT_FOR_OWNERSHIP) continue; // 跳过地面/非常低的像素

            float best_d2 = std::numeric_limits<float>::max();
            int best_tree_idx = -1;

            float px_world = context.min_x + (x_pix + 0.5f) * context.chm_resolution;
            float py_world = context.min_y + (y_pix + 0.5f) * context.chm_resolution;

            for (int i = 0; i < treetops.size(); ++i) {
                float dx_world = px_world - treetops[i][0];
                float dy_world = py_world - treetops[i][1];
                float d2 = dx_world * dx_world + dy_world * dy_world;

                // 仅当像素位于此树的最大影响区域内时才考虑
                float max_influence_radius_m = treetops[i][2] * MAX_INFLUENCE_RADIUS_FACTOR_FOR_REFINEMENT;
                if (d2 < max_influence_radius_m * max_influence_radius_m) {
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best_tree_idx = i;
                    }
                }
            }
            pixel_owner.at<int>(y_pix, x_pix) = best_tree_idx;
        }
    }

    // --- 提取轮廓并标记点云 ---
    std::set<int> unique_tree_ids;
    for (int r = 0; r < context.chm_height; ++r) {
        for (int c = 0; c < context.chm_width; ++c) {
            int tid = pixel_owner.at<int>(r, c);
            if (tid >= 0) unique_tree_ids.insert(tid);
        }
    }
    if (unique_tree_ids.empty()) {
        std::cout << "像素所有权分配阶段后没有像素分配给任何树。正在退出。" << std::endl;
        return 0;
    }

    std::map<int, std::vector<std::vector<cv::Point2f>>> tree_contours_geo; // 树ID -> 轮廓列表 (每个轮廓是Point2f列表)
    std::cout << "提取树冠轮廓..." << std::endl;
    for (int tree_id : unique_tree_ids) {
        cv::Mat mask = cv::Mat::zeros(context.chm_height, context.chm_width, CV_8UC1);
        for (int r = 0; r < context.chm_height; ++r) {
            for (int c = 0; c < context.chm_width; ++c) {
                if (pixel_owner.at<int>(r, c) == tree_id)
                    mask.at<uchar>(r, c) = 255;
            }
        }
        std::vector<std::vector<cv::Point>> contours_px; // 使用OpenCV的点类型
        cv::findContours(mask, contours_px, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<std::vector<cv::Point2f>> contours_geo_current_tree;
        for (const auto& contour_px : contours_px) {
            std::vector<cv::Point2f> geo_single_contour;
            for (const auto& pt_px : contour_px) {
                geo_single_contour.emplace_back(
                    context.min_x + (pt_px.x + 0.5f) * context.chm_resolution,
                    context.min_y + (pt_px.y + 0.5f) * context.chm_resolution
                );
            }
            if (geo_single_contour.size() >= 3) { // 有效多边形至少需要3个点
                contours_geo_current_tree.push_back(geo_single_contour);
            }
        }
        if (!contours_geo_current_tree.empty()) {
            tree_contours_geo[tree_id] = contours_geo_current_tree;
        }
    }
    if (tree_contours_geo.empty()) {
        std::cout << "未提取到有效轮廓。正在退出。" << std::endl;
        return 0;
    }

    // --- 创建带标签的点云 ---
    std::cout << "标记原始点云..." << std::endl;
    pcl::PointCloud<pcl::PointXYZL>::Ptr labeled_cloud(new pcl::PointCloud<pcl::PointXYZL>);
    labeled_cloud->reserve(cloud->size()); // 预分配内存

    for (const auto& pt_orig : cloud->points) {
        pcl::PointXYZL pt_labeled = pt_orig;
        pt_labeled.label = 0; // 默认：未分配
        bool assigned = false; // 标记点是否已被分配

        for (const auto& kv_pair : tree_contours_geo) {
            int current_tree_id_zero_based = kv_pair.first;
            const auto& contours_for_this_tree = kv_pair.second;
            for (const auto& single_contour : contours_for_this_tree) {
                if (single_contour.size() < 3) continue;
                // cv::pointPolygonTest 需要 cv::Point2f
                if (cv::pointPolygonTest(single_contour, cv::Point2f(pt_labeled.x, pt_labeled.y), false) >= 0) {
                    pt_labeled.label = current_tree_id_zero_based + 1; // 标签从1开始
                    assigned = true;
                    break; // 点已分配给一个树，跳到下一个原始点
                }
            }
            if (assigned) {
                break; // 点已分配，跳出外层树ID循环
            }
        }
        labeled_cloud->push_back(pt_labeled);
    }
    labeled_cloud->width = labeled_cloud->size();
    labeled_cloud->height = 1;
    labeled_cloud->is_dense = true; // 假设所有点都是有效的

    // --- 保存结果 ---
    std::cout << "保存标记后的点云到: " << pcd_output_path << std::endl;
    pcl::io::savePCDFileASCII(pcd_output_path, *labeled_cloud);

    std::cout << "生成预览图像..." << std::endl;
    cv::Mat preview_image = cv::Mat::zeros(context.chm_height, context.chm_width, CV_8UC3);
    cv::RNG rng(12345); // 用于生成不同颜色
    for (int r = 0; r < context.chm_height; ++r) {
        for (int c = 0; c < context.chm_width; ++c) {
            int tid = pixel_owner.at<int>(r, c);
            if (tid >= 0) {
                // 为每个tree_id生成一个大致唯一的颜色
                cv::Scalar color = cv::Scalar(rng.uniform(30, 250), rng.uniform(30, 250), rng.uniform(30, 250));
                preview_image.at<cv::Vec3b>(r, c) = cv::Vec3b(color[0], color[1], color[2]);
            }
        }
    }
    // 在预览图上绘制树顶
    for (const auto& tt : treetops) {
        int cx_pix = static_cast<int>((tt[0] - context.min_x) / context.chm_resolution);
        int cy_pix = static_cast<int>((tt[1] - context.min_y) / context.chm_resolution);
        if (cx_pix >= 0 && cx_pix < context.chm_width && cy_pix >= 0 && cy_pix < context.chm_height) {
            cv::circle(preview_image, cv::Point(cx_pix, cy_pix), 3, cv::Scalar(0, 0, 255), -1); // 红色表示树顶
        }
    }
    cv::imwrite(preview_output_path, preview_image);
    std::cout << "预览图像已保存到: " << preview_output_path << std::endl;

    // 保存CHM用于调试
    cv::Mat chm_display;
    if (!topPointsMat.empty()) {
        double minVal, maxVal;
        cv::minMaxLoc(topPointsMat, &minVal, &maxVal); // 使用OpenCV的minMaxLoc
        if (maxVal > minVal) { // 避免CHM平坦时除以零
            topPointsMat.convertTo(chm_display, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
            cv::imwrite(chm_debug_output_path, chm_display);
            std::cout << "CHM调试图像已保存到: " << chm_debug_output_path << std::endl;
        }
        else if (!topPointsMat.empty()) { // CHM平坦但不为空
            topPointsMat.convertTo(chm_display, CV_8UC1); // 可能全黑或全白
            cv::imwrite(chm_debug_output_path, chm_display);
            std::cout << "CHM调试图像 (平坦) 已保存到: " << chm_debug_output_path << std::endl;
        }
    }

    std::cout << "处理完成。" << std::endl;
    return 0;
}