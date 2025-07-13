// 定义PCL_NO_PRECOMPILE宏，表示我们可能会使用自定义点类型，
// 并且不希望PCL库预编译常见的点类型组合，这在某些情况下可以减少编译时间或解决链接问题。
#define PCL_NO_PRECOMPILE
#include <iostream> // 用于标准输入输出流操作 (例如 cout, cerr)
#include <vector>   // 用于使用vector动态数组
#include <cmath>    // 用于数学函数 (例如 sqrt, ceil, pow)
#include <limits>   // 用于访问数值类型的极限值 (例如 numeric_limits<float>::max())
#include <algorithm> // 用于通用算法 (例如 max, sort)
#include <memory>   // 用于智能指针 (例如 shared_ptr)
#include <string>   // 用于使用string字符串类
#include <set>      // 用于使用set集合容器 (存储唯一且有序的元素)
#include <map>      // 用于使用map关联容器 (存储键值对)
#include <queue>
#include <random>


#include <pcl/io/pcd_io.h>      // PCL库中用于PCD文件输入输出的类和函数
#include <pcl/point_types.h>    // PCL库中定义的各种点类型 (例如 PointXYZ, PointXYZI)
#include <pcl/point_cloud.h>    // PCL库中PointCloud类的定义
#include <pcl/common/common.h>  // PCL通用模块，包含如getMinMax3D等函数
#include <pcl/common/centroid.h> // PCL通用模块，包含计算点云质心的函数 (compute3DCentroid)
#include <pcl/common/distances.h>
#include <pcl/filters/extract_indices.h> // PCL滤波器，用于根据点索引提取点云子集
#include <pcl/segmentation/extract_clusters.h>      // PCL分割模块，用于欧氏聚类提取
#include <pcl/segmentation/conditional_euclidean_clustering.h> // PCL分割模块，用于条件欧氏聚类
#include <pcl/sample_consensus/sac_model_circle.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/common/geometry.h>
#include <boost/math/special_functions/fpclassify.hpp>
#include <Eigen/Dense> // Eigen库的核心功能，主要用于矩阵和向量运算

#include <opencv2/opencv.hpp> // OpenCV库的主要头文件，包含了大部分OpenCV功能

// 命名空间
using namespace std;
using namespace pcl;
using namespace cv;

// 存储和传递整个处理流程中需要共享的上下文信息和参数
struct ProcessingContext {
    float min_x = numeric_limits<float>::max(); // 点云在X轴上的最小值，初始化为float最大值以便任何点都能更新它
    float max_x = numeric_limits<float>::lowest(); // 点云在X轴上的最大值，初始化为float最小值以便任何点都能更新它
    float min_y = numeric_limits<float>::max(); // 点云在Y轴上的最小值
    float max_y = numeric_limits<float>::lowest(); // 点云在Y轴上的最大值
    float min_z_orig = numeric_limits<float>::max(); // 点云在Z轴上的原始最小值 (用于计算归一化高度)
    float chm_resolution = 0.2f; // 冠层高度模型(CHM)的栅格分辨率 (单位：米/像素)

    int chm_width = 0;  // CHM的宽度 (像素数)
    int chm_height = 0; // CHM的高度 (像素数)

    // 更新点云的边界框信息 (min_x, max_x, min_y, max_y, min_z_orig)
    void updateBounds(const PointCloud<PointXYZI>& cloud) {
        PointXYZI min_pt, max_pt; // 用于存储点云的最小和最大点
        getMinMax3D(cloud, min_pt, max_pt); // PCL函数，计算点云在三个维度上的最小和最大坐标
        min_x = min_pt.x; // 更新X轴最小值
        max_x = max_pt.x; // 更新X轴最大值
        min_y = min_pt.y; // 更新Y轴最小值
        max_y = max_pt.y; // 更新Y轴最大值
        min_z_orig = min_pt.z; // 更新Z轴原始最小值
    }

    // 根据点云边界和CHM分辨率计算CHM的维度 (宽度和高度)
    void calculateChmDimensions() {
        // 确保边界和分辨率有效
        if (max_x > min_x && max_y > min_y && chm_resolution > 0) {
            // 宽度 = X方向范围 / 分辨率，向上取整以覆盖整个区域
            chm_width = static_cast<int>(ceil((max_x - min_x) / chm_resolution));
            // 高度 = Y方向范围 / 分辨率，向上取整
            chm_height = static_cast<int>(ceil((max_y - min_y) / chm_resolution));
        }
        else {
            // 如果边界或分辨率无效，则维度设为0
            chm_width = 0;
            chm_height = 0;
        }
    }
};

// 存储树干信息的结构体
struct TreeInfo {
    int id;
    Eigen::Vector3f position;
    float ground_z;
    float dbh;
    float height;
};

// --- 杆状物提取模块的参数 ---
const float POLE_Z_SLICE_HEIGHT = 5.2f; // Z轴分层进行水平聚类时，从点云最低点开始向上考虑的总高度 (单位：米)
const float POLE_Z_SLICE_RESOLUTION = 0.4f;  // 垂直分层时每个薄层的高度 (单位：米)
const float POLE_HORIZONTAL_CLUSTER_TOLERANCE = 0.3f;   // 水平欧氏聚类的距离容差 (单位：米)
const int POLE_HORIZONTAL_MIN_CLUSTER_SIZE = 10;    // 水平聚类中一个簇最少的点数
const int POLE_HORIZONTAL_MAX_CLUSTER_SIZE = 500;   // 水平聚类中一个簇最多的点数
const float POLE_MAX_HORIZONTAL_DIAMETER_THRESHOLD = 0.8f;  // 水平聚类后，对簇进行直径过滤的阈值 (单位：米)，大于此直径的簇不像树干横截面
const float POLE_VERTICAL_CEC_CLUSTER_TOLERANCE = 0.75f;  // 垂直条件欧氏聚类的距离容差 (单位：米)
const int POLE_VERTICAL_CEC_MIN_CLUSTER_SIZE = 30; // 垂直条件欧氏聚类中一个簇最少的点数 (调整此参数可能更重要)
const int POLE_VERTICAL_CEC_MAX_CLUSTER_SIZE = 2000;   // 垂直条件欧氏聚类中一个簇最多的点数
const float POLE_VERTICAL_CEC_HORIZONTAL_GROWTH_THRESHOLD = 0.25f; // 垂直条件欧氏聚类的自定义条件中，允许点在水平方向上增长的最大距离 (调整此参数可能更重要)

// --- 单木分割模块的参数 ---
// 这些参数用于后续从树干位置确定树顶，以及基于树顶进行冠层分割
const float MIN_TREETOP_NORMALIZED_HEIGHT = 0.5f;  // 树顶在归一化CHM上的最小高度，低于此高度的树顶候选被忽略 (单位：米)
const float MIN_DIST_SQ_BETWEEN_TREETOPS = 1.0f * 1.0f; // 两个树顶之间最小允许距离的平方(单位：米 ^ 2)，用于避免过密的树顶
const float CROWN_RADIUS_FACTOR = 0.65f;   // 树冠半径与树高的比例因子 (树冠半径 = CHM高度 * 此因子)
const float CROWN_MIN_RADIUS_PIXELS = 2.0f;   // 树冠影响区域的最小半径 (单位：像素)，确保即使很矮的树也有一定的覆盖范围
const float MIN_CHM_HEIGHT_FOR_OWNERSHIP = 0.1f; // CHM像元归属于某一棵树时，该像元自身的最小高度(单位：米)，过滤掉地表等过低区域

/**
 * @brief (杆状物提取) 垂直条件欧氏聚类的自定义条件函数
 * @param pointA 第一个点
 * @param pointB 第二个点
 * @param squared_distance 两点之间的欧氏距离的平方 (PCL的CEC会传入这个值，但这里我们不用)
 * @return 如果两点满足自定义的连接条件，则返回true，否则返回false
 * 这个函数定义了点如何才能被认为是“邻近”并属于同一个垂直簇的条件。
 * 这里主要限制了簇在水平方向上的增长，确保形成的簇是垂直细长的。
 */
bool Condition(const PointXYZ& pointA, const PointXYZ& pointB, float squared_distance) {
    float dx = pointA.x - pointB.x; // X方向的差值
    float dy = pointA.y - pointB.y; // Y方向的差值
    float horizontal_distance_sq = dx * dx + dy * dy; // 水平距离的平方
    // 如果两点之间的水平距离平方小于设定的阈值的平方，则认为它们满足条件
    // 这意味着只允许在水平方向上扩展POLE_VERTICAL_CEC_HORIZONTAL_GROWTH_THRESHOLD的距离
    return horizontal_distance_sq < (POLE_VERTICAL_CEC_HORIZONTAL_GROWTH_THRESHOLD * POLE_VERTICAL_CEC_HORIZONTAL_GROWTH_THRESHOLD);
}

/**
 * @brief 使用参考思路实现的杆状物/树干提取函数
 * @param input_cloud_main 原始输入的点云 (PointXYZI)
 * @param context_hint ProcessingContext，用于获取点云边界 (虽然函数内部也会计算)
 * @return 检测到的树干的质心位置 (世界坐标 X, Y, Z)
 */
vector<Eigen::Vector3f> extractTrunk(
    PointCloud<PointXYZI>::Ptr input_cloud_main,  // 输入的原始点云，使用智能指针
    const ProcessingContext& context_hint  // 传入处理上下文的常量引用
) {
    vector<Eigen::Vector3f> trunk_centroids;  // 用于存储最终提取到的树干质心
    // 创建一个新的点云对象，用于杆状物提取，点类型为PointXYZ_Pole(只关心XYZ)
    PointCloud<PointXYZ>::Ptr cloud_for_pole_extraction(new PointCloud<PointXYZ>);

    // 将输入的PointXYZI点云复制到PointXYZ_Pole点云中 (忽略intensity和自定义字段)
    copyPointCloud(*input_cloud_main, *cloud_for_pole_extraction);

    // 检查输入点云是否为空
    if (cloud_for_pole_extraction->empty()) {
        PCL_WARN("杆状物提取：输入点云为空。\n");
        return trunk_centroids;
    }

    PointXYZ min_pt_global, max_pt_global;
    getMinMax3D(*cloud_for_pole_extraction, min_pt_global, max_pt_global);
    cout << "杆状物提取：点云Z轴范围: [" << min_pt_global.z << ", " << min_pt_global.z + POLE_Z_SLICE_HEIGHT << "]" << endl;

    // 创建一个点云对象，用于存储通过水平聚类和直径过滤后得到的、用于后续垂直聚类的候选点
    PointCloud<PointXYZ>::Ptr candidate_points_for_vertical_clustering(new PointCloud<PointXYZ>);
    // 创建一个PCL的ExtractIndices对象，用于从点云中提取指定索引的点
    ExtractIndices<PointXYZ> extract_pole;

    cout << "杆状物提取：开始Z轴分层、水平聚类和直径过滤..." << endl;
    // 循环进行Z轴分层处理：从点云最低点开始，按POLE_Z_SLICE_RESOLUTION步长向上处理，直到达到POLE_Z_SLICE_HEIGHT设定的总高度
    for (float current_z_min = min_pt_global.z; current_z_min < min_pt_global.z + POLE_Z_SLICE_HEIGHT; current_z_min += POLE_Z_SLICE_RESOLUTION) {
        float current_z_max = current_z_min + POLE_Z_SLICE_RESOLUTION;
        PointCloud<PointXYZ>::Ptr slice_cloud(new PointCloud<PointXYZ>);
        PointIndices::Ptr slice_indices(new PointIndices);

        // 遍历整个用于杆状物提取的点云，找出在当前Z轴范围内的点
        for (int i = 0; i < cloud_for_pole_extraction->points.size(); ++i) {
            if (cloud_for_pole_extraction->points[i].z >= current_z_min && cloud_for_pole_extraction->points[i].z < current_z_max) {
                slice_indices->indices.push_back(i);
            }
        }

        // 如果当前薄层没有点，则跳过后续处理，继续下一个薄层
        if (slice_indices->indices.empty()) continue;

        // 设置ExtractIndices对象的输入点云和要提取的索引
        extract_pole.setInputCloud(cloud_for_pole_extraction);
        extract_pole.setIndices(slice_indices);
        extract_pole.filter(*slice_cloud);  // 执行提取，结果存储在slice_cloud中

        // 如果当前薄层提取出的点数少于水平聚类的最小簇大小要求，则认为不太可能形成有效簇，跳过
        if (slice_cloud->size() < static_cast<size_t>(POLE_HORIZONTAL_MIN_CLUSTER_SIZE)) continue;

        // 为当前薄片点云创建KdTree，用于加速近邻搜索
        search::KdTree<PointXYZ>::Ptr tree_horizontal(new search::KdTree<PointXYZ>);
        tree_horizontal->setInputCloud(slice_cloud);  // 设置KdTree的输入点云

        vector<PointIndices> horizontal_cluster_indices;  // 用于存储水平欧氏聚类结果的索引
        EuclideanClusterExtraction<PointXYZ> ec_horizontal;  // 创建欧氏聚类对象
        ec_horizontal.setClusterTolerance(POLE_HORIZONTAL_CLUSTER_TOLERANCE);
        ec_horizontal.setMinClusterSize(POLE_HORIZONTAL_MIN_CLUSTER_SIZE);
        ec_horizontal.setMaxClusterSize(POLE_HORIZONTAL_MAX_CLUSTER_SIZE);
        ec_horizontal.setSearchMethod(tree_horizontal);
        ec_horizontal.setInputCloud(slice_cloud);
        ec_horizontal.extract(horizontal_cluster_indices); // 执行聚类，结果是每个簇的点索引集合

        // 遍历通过水平欧氏聚类找到的所有簇
        for (const auto& h_indices : horizontal_cluster_indices) {

            PointCloud<PointXYZ>::Ptr horizontal_cluster(new PointCloud<PointXYZ>);
            PointIndices::Ptr current_h_cluster_indices(new PointIndices(h_indices));

            // 创建一个临时的ExtractIndices对象，用于提取当前水平簇的点
            ExtractIndices<PointXYZ> extract_h_cluster_temp;
            extract_h_cluster_temp.setInputCloud(slice_cloud);
            extract_h_cluster_temp.setIndices(current_h_cluster_indices);
            extract_h_cluster_temp.filter(*horizontal_cluster);

            // 如果提取出的水平簇为空，则跳过
            if (horizontal_cluster->points.empty()) continue;

            PointXYZ min_pt_h_cluster, max_pt_h_cluster;
            getMinMax3D(*horizontal_cluster, min_pt_h_cluster, max_pt_h_cluster);
            float width_x = max_pt_h_cluster.x - min_pt_h_cluster.x;
            float width_y = max_pt_h_cluster.y - min_pt_h_cluster.y;
            float diameter_approx_sq = width_x * width_x + width_y * width_y;

            // 如果近似直径小于设定的阈值，则认为这个水平簇可能是一个杆状物的横截面
            if (sqrt(diameter_approx_sq) < POLE_MAX_HORIZONTAL_DIAMETER_THRESHOLD) {
                // 将这个合格的水平簇中的所有点加入到用于垂直聚类的候选点云中
                *candidate_points_for_vertical_clustering += *horizontal_cluster;
            }
        }
    }
    cout << "杆状物提取：水平聚类和直径过滤完成。候选点数: " << candidate_points_for_vertical_clustering->size() << endl;

    // 检查用于垂直聚类的候选点数量是否足够
    if (candidate_points_for_vertical_clustering->empty() || candidate_points_for_vertical_clustering->size() < static_cast<size_t>(POLE_VERTICAL_CEC_MIN_CLUSTER_SIZE)) {
        PCL_WARN("杆状物提取：没有足够的候选点进行垂直聚类。\n");
        return trunk_centroids;
    }

    cout << "杆状物提取：开始垂直条件欧氏聚类..." << endl;
    vector<PointIndices> vertical_cluster_indices_cec;
    ConditionalEuclideanClustering<PointXYZ> cec(true);
    cec.setInputCloud(candidate_points_for_vertical_clustering);
    cec.setConditionFunction(&Condition);
    cec.setClusterTolerance(POLE_VERTICAL_CEC_CLUSTER_TOLERANCE);
    cec.setMinClusterSize(POLE_VERTICAL_CEC_MIN_CLUSTER_SIZE);
    cec.setMaxClusterSize(POLE_VERTICAL_CEC_MAX_CLUSTER_SIZE);
    cec.segment(vertical_cluster_indices_cec);
    cout << "杆状物提取：CEC找到 " << vertical_cluster_indices_cec.size() << " 个潜在垂直簇。" << endl;

    if (vertical_cluster_indices_cec.empty()) {
        PCL_WARN("杆状物提取：CEC未能找到任何垂直簇。\n");
        return trunk_centroids;
    }

    cout << "杆状物提取：使用CEC簇的质心作为树干候选..." << endl;
    for (const auto& v_indices : vertical_cluster_indices_cec) {
        PointCloud<PointXYZ>::Ptr cec_cluster(new PointCloud<PointXYZ>); // 存储CEC簇的点云
        PointIndices::Ptr current_v_cluster_indices(new PointIndices(v_indices));

        ExtractIndices<PointXYZ> extract_v_cluster_temp; // 创建局部的ExtractIndices对象
        extract_v_cluster_temp.setInputCloud(candidate_points_for_vertical_clustering);
        extract_v_cluster_temp.setIndices(current_v_cluster_indices);
        extract_v_cluster_temp.filter(*cec_cluster);  //提取当前CEC簇的点

        // 计算该CEC簇的质心
        Eigen::Vector4f centroid_4f;
        compute3DCentroid(*cec_cluster, centroid_4f);
        trunk_centroids.emplace_back(centroid_4f[0], centroid_4f[1], centroid_4f[2]);
        cout << "CEC簇质心: (" << centroid_4f[0] << ", " << centroid_4f[1] << ", " << centroid_4f[2] << ")" << endl;
    }

    return trunk_centroids;
}

/**
 * @brief 基于邻域最大点数的聚类算法（使用treetops作为种子点）
 * @param numPointsMat 输入的点数矩阵，每个像素值表示该位置的点云数量
 * @param treetops 过滤后的树顶坐标集合
 * @param context 处理上下文，包含坐标转换信息
 * @return 聚类标签矩阵，每个像素标记为所属聚类的ID
 */
Mat clusterByMaxNeighborWithTreetops(const Mat& numPointsMat,
    const vector<Eigen::Vector3f>& treetops,
    const ProcessingContext& context) {
    // 检查输入有效性
    if (numPointsMat.empty() || treetops.empty()) {
        return Mat();
    }

    int rows = numPointsMat.rows;
    int cols = numPointsMat.cols;

    // 创建输出标签矩阵，初始化为-1表示未标记
    Mat labels(rows, cols, CV_32SC1, Scalar(-1));

    // 定义优先队列元素结构
    struct PixelInfo {
        Point position;
        int count;
        int label;

        // 按点数从大到小排序
        bool operator<(const PixelInfo& other) const {
            return count < other.count;
        }
    };

    // 创建优先队列（最大堆）
    priority_queue<PixelInfo> pq;

    // 初始化：将树顶坐标作为种子点
    for (size_t i = 0; i < treetops.size(); ++i) {
        // 将3D坐标转换为图像坐标
        int x = static_cast<int>((treetops[i][0] - context.min_x) / context.chm_resolution);
        int y = static_cast<int>((treetops[i][1] - context.min_y) / context.chm_resolution);

        // 检查坐标是否在图像范围内
        if (x >= 0 && x < cols && y >= 0 && y < rows) {
            int count = numPointsMat.at<int>(y, x);

            if (count > 0) {
                PixelInfo seed;
                seed.position = Point(x, y);
                seed.count = count;
                seed.label = static_cast<int>(i) + 1;  // 使用索引作为标签

                pq.push(seed);
                labels.at<int>(y, x) = seed.label;
            }
        }
    }

    //const int dx[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    //const int dy[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
    const int dx[] = { -2,-1, 0, 1, 2,  // 第一行
                   -2,-1, 0, 1, 2,  // 第二行
                   -2,-1,    1, 2,  // 第三行（跳过中心点）
                   -2,-1, 0, 1, 2,  // 第四行
                   -2,-1, 0, 1, 2 }; // 第五行
    const int dy[] = { -2,-2,-2,-2,-2,  // 第一列
                       -1,-1,-1,-1,-1,  // 第二列
                        0, 0,    0, 0,  // 第三列（跳过中心点）
                        1, 1, 1, 1, 1,  // 第四列
                        2, 2, 2, 2, 2 }; // 第五列
    // 区域生长过程
    while (!pq.empty()) {
        PixelInfo current = pq.top();
        pq.pop();

        // 检查8邻域
        for (int i = 0; i < 24; ++i) {
            int nx = current.position.x + dx[i];
            int ny = current.position.y + dy[i];

            // 检查邻域像素是否在图像范围内且未被标记
            if (nx >= 0 && nx < cols && ny >= 0 && ny < rows &&
                labels.at<int>(ny, nx) == -1) {

                int neighborCount = numPointsMat.at<int>(ny, nx);

                // 如果邻域像素有点数，则标记并加入队列
                if (neighborCount > 0) {
                    PixelInfo neighbor;
                    neighbor.position = Point(nx, ny);
                    neighbor.count = neighborCount;
                    neighbor.label = current.label;

                    pq.push(neighbor);
                    labels.at<int>(ny, nx) = neighbor.label;
                }
            }
        }
    }
    return labels;
}

/**
 * @brief 计算三角形面积（海伦公式）
 * @return 面积值，如果三点共线返回0
 */
float calculateTriangleArea(const PointXYZ& p1,
    const PointXYZ& p2,
    const PointXYZ& p3)
{
    // 计算三角形边长
    float a = euclideanDistance(p2, p3);
    float b = euclideanDistance(p1, p3);
    float c = euclideanDistance(p1, p2);

    // 海伦公式
    float s = (a + b + c) / 2.0f;
    float area_sq = s * (s - a) * (s - b) * (s - c);

    // 处理负数情况（三点共线时）
    return area_sq > 0 ? sqrt(area_sq) : 0.0f;
}

/**
 * @brief 通过三点计算圆的半径
 * @return 半径值，如果三点共线返回NaN
 */
float calculateCircleRadius(const PointXYZ& p1,
    const PointXYZ& p2,
    const PointXYZ& p3)
{
    float area = calculateTriangleArea(p1, p2, p3);
    if (area < 1e-6f) {
        return numeric_limits<float>::quiet_NaN();
    }

    // 计算外接圆半径：R = a*b*c/(4*Area)
    float a = euclideanDistance(p2, p3);
    float b = euclideanDistance(p1, p3);
    float c = euclideanDistance(p1, p2);
    return (a * b * c) / (4 * area);
}

/**
 * @brief 基于三点拟合的稳健胸径计算
 * @param cloud 输入点云
 * @param trunk_center 树干中心
 * @param ground_z 地面高度
 * @param search_radius 搜索半径
 * @param num_iterations 随机采样次数
 * @return 胸径估计值
 */
float calculateDBH_ThreePointFit(const PointCloud<PointXYZI>::Ptr& cloud,
    const Eigen::Vector3f& trunk_center,
    float ground_z,
    float search_radius = 0.5f,
    int num_iterations = 100)
{
    // 1. 提取1.2m高度范围内的点
    vector<PointXYZ> candidates;
    const float height_min = ground_z + 1.1f;
    const float height_max = ground_z + 1.3f;

    for (const auto& pt : cloud->points) {
        float dist_sq = pow(pt.x - trunk_center[0], 2) + pow(pt.y - trunk_center[1], 2);
        if (pt.z >= height_min && pt.z <= height_max &&
            dist_sq <= search_radius * search_radius) {
            candidates.emplace_back(pt.x, pt.y, 0);
        }
    }

    if (candidates.size() < 3) return 0.0f;

    // 2. 多次随机采样三点组合
    vector<float> valid_radii;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, candidates.size()-1);

    for (int i = 0; i < num_iterations; ++i) {
        int idx1 = dis(gen);
        int idx2 = dis(gen);
        int idx3 = dis(gen);
        if (idx1 == idx2 || idx2 == idx3 || idx1 == idx3) continue;

        float r = calculateCircleRadius(candidates[idx1], 
                                      candidates[idx2],
                                      candidates[idx3]);
        if (isfinite(r) && r > 0.02f && r < 1.0f) {
            valid_radii.push_back(r);
        }
    }

    // 3. 统计有效半径
    if (valid_radii.empty()) return 0.0f;

    // 使用中位数作为最终估计（比平均值更抗噪声）
    sort(valid_radii.begin(), valid_radii.end());
    float median_radius = valid_radii[valid_radii.size() / 2];

    // 4. 直径补偿（针对部分圆弧的修正）
    float compensation_factor = 1.0f;
    if (valid_radii.size() < num_iterations / 2) {
        // 如果有效拟合比例低，说明可能是部分圆弧
        compensation_factor = 1.3f; // 经验补偿值
    }
    return 2 * median_radius * compensation_factor;
}


/**
 * @brief 计算单木树高和地面高度
 * @param cloud 原始点云
 * @param tree_id 树木ID
 * @param trunk_center 树干中心位置(XY)
 * @param context 处理上下文
 * @param ground_search_radius 地面点搜索半径(米)
 * @return pair(树高, 地面高度)
 */
pair<float, float> calculateTreeHeightAndGround(const PointCloud<PointXYZI>::Ptr& cloud,
    int tree_id,
    const Eigen::Vector3f& trunk_center,
    const ProcessingContext& context,
    float ground_search_radius = 0.5f) {
    float max_z = -numeric_limits<float>::max();
    float min_z = numeric_limits<float>::max();

    // 查找该树的最高点和邻近地面最低点
    for (const auto& pt : cloud->points) {
        float dist_sq = pow(pt.x - trunk_center[0], 2) + pow(pt.y - trunk_center[1], 2);
        if (dist_sq <= ground_search_radius * ground_search_radius) {
            if (pt.z > max_z) max_z = pt.z;
            if (pt.z < min_z) min_z = pt.z;
        }
    }

    return make_pair(max_z - min_z, min_z);
}

// 原始代码中的 loadXYZILabelTXT 函数
PointCloud<PointXYZI>::Ptr loadXYZILabelTXT(const string& filename) {
    cout << "正在读取点云..." << endl;
    PointCloud<PointXYZI>::Ptr cloud(new PointCloud<PointXYZI>);
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        return cloud;
    }
    string line;
    while (getline(infile, line)) {
        stringstream ss(line);
        PointXYZI pt;
        float label_val; // 变量名修改以避免与PointXYZL中的label混淆
        if (ss >> pt.x >> pt.y >> pt.z >> pt.intensity >> label_val) {
            if (label_val != 0) cloud->push_back(pt);
        }
    }
    infile.close();
    cout << "从TXT读取点数 (label != 0): " << cloud->size() << endl;
    return cloud;
}

int main() {
    ProcessingContext context;
    string path = "D:/Experiments_2025_spring/2025LiDAR/CourseWorkData/Origin/";

    string infpath = path + "Z9_training.txt"; // 请修改为你的路径
    // cout << "请输入用于单木分割的TXT点云文件路径: ";
    // cin >> infpath;
    PointCloud<PointXYZI>::Ptr cloud = loadXYZILabelTXT(infpath);

    //string infpath = path + "trees.pcd";
    //PointCloud<PointXYZI>::Ptr cloud(new PointCloud<PointXYZI>);
    //cout << "正在读取点云..." << endl;
    //if (io::loadPCDFile<PointXYZI>(infpath, *cloud) == -1)
    //{
    //    PCL_ERROR("读取点云失败\n");
    //    return (-1);
    //}
    

    if (cloud->empty()) {
        PCL_ERROR("加载的点云为空，程序退出。\n");
        return -1;
    }
    cout << "加载点云成功，点数: " << cloud->size() << endl;

    context.updateBounds(*cloud);
    context.calculateChmDimensions();

    if (context.min_x == numeric_limits<float>::max() || cloud->points.size() < 10) {
        cerr << "错误: 点云边界无效或点数过少。" << endl;
        return -1;
    }
    if (context.chm_width <= 0 || context.chm_height <= 0) {
        cerr << "错误: CHM维度无效 (" << context.chm_width << "x" << context.chm_height << ")。"
            << "请检查分辨率 (" << context.chm_resolution << ") 或点云范围。" << endl;
        return -1;
    }
    cout << "CHM维度: " << context.chm_width << "x" << context.chm_height
        << " (分辨率: " << context.chm_resolution << "m)" << endl;

    cout << "创建冠层高度模型 (CHM)..." << endl;
    Mat topPointsMat = Mat::zeros(context.chm_height, context.chm_width, CV_32F);
    Mat numPointsMat = Mat::zeros(context.chm_height, context.chm_width, CV_32SC1);
    for (const auto& pt_orig : cloud->points) {
        float normalized_z = pt_orig.z - context.min_z_orig;
        int ix = static_cast<int>((pt_orig.x - context.min_x) / context.chm_resolution);
        int iy = static_cast<int>((pt_orig.y - context.min_y) / context.chm_resolution);
        if (ix >= 0 && ix < context.chm_width && iy >= 0 && iy < context.chm_height) {
            topPointsMat.at<float>(iy, ix) = max(topPointsMat.at<float>(iy, ix), normalized_z);
        }
        numPointsMat.at<int>(iy, ix) += 1;
    }
    cout << "CHM创建完成。" << endl;

    cout << "开始采用欧式聚类方法检测并提取树干.." << endl;
    vector<Eigen::Vector3f> trunk_centroids_from_pole_method = extractTrunk(cloud, context);
    cout << "杆状物提取方法找到 " << trunk_centroids_from_pole_method.size() << " 个树干候选。" << endl;

    vector<Eigen::Vector3f> treetops_for_segmentation;
    // 过滤并转换树干候选
    if (!trunk_centroids_from_pole_method.empty()) {
        for (const auto& trunk_centroid : trunk_centroids_from_pole_method) {
            float trunk_x = trunk_centroid[0];
            float trunk_y = trunk_centroid[1];
            // 将树干位置转换为CHM网格坐标
            int cx_grid = static_cast<int>((trunk_x - context.min_x) / context.chm_resolution);
            int cy_grid = static_cast<int>((trunk_y - context.min_y) / context.chm_resolution);

            // 检查转换后的网格坐标是否在CHM范围内
            if (cx_grid >= 0 && cx_grid < context.chm_width && cy_grid >= 0 && cy_grid < context.chm_height) {
                float chm_height_at_trunk = topPointsMat.at<float>(cy_grid, cx_grid);
                if (chm_height_at_trunk > MIN_TREETOP_NORMALIZED_HEIGHT) {
                    Eigen::Vector3f current_treetop_candidate(trunk_x, trunk_y, chm_height_at_trunk);//直接用树干坐标作为树顶候选
                    bool too_close = false;
                    // 检查当前树顶候选是否与已有的树顶过近
                    for (const auto& existing_tt : treetops_for_segmentation) {
                        float dx = existing_tt[0] - current_treetop_candidate[0];
                        float dy = existing_tt[1] - current_treetop_candidate[1];
                        if (dx * dx + dy * dy < MIN_DIST_SQ_BETWEEN_TREETOPS) {
                            too_close = true;
                            break;
                        }
                    }
                    if (!too_close) {
                        treetops_for_segmentation.push_back(current_treetop_candidate);
                    }
                }
            }
        }
    }
    cout << "过滤并转换后得到 " << treetops_for_segmentation.size() << " 个树顶用于后续分割。" << endl;

    if (treetops_for_segmentation.empty()) {
        cout << "未找到有效树顶信息。正在退出。" << endl;
        return 0;
    }

    // 为每棵树计算统计信息
    vector<TreeInfo> tree_stats;
    cout << "开始计算单木参数..." << endl;
    for (size_t i = 0; i < treetops_for_segmentation.size(); ++i) {
        TreeInfo info;
        info.id = i + 1; // ID从1开始
        info.position = treetops_for_segmentation[i];

        // 先计算树高和地面高度
        auto height_ground = calculateTreeHeightAndGround(cloud, info.id, info.position, context);
        info.height = height_ground.first;
        info.ground_z = height_ground.second;

        // 基于该树的地面高度计算胸径
        info.dbh = calculateDBH_ThreePointFit(
            cloud,
            info.position,
            info.ground_z,
            0.5f,    // 搜索半径
            100      // 采样次数
        );

        //if (info.dbh < 0.01f || info.dbh > POLE_MAX_HORIZONTAL_DIAMETER_THRESHOLD) {
        //    treetops_for_segmentation.erase(treetops_for_segmentation.begin() + i);
        //    i--;
        //    continue;
        //}

        tree_stats.push_back(info);

        cout << "树 " << info.id << ": 位置(" << info.position[0] << "," << info.position[1]
            << "), 地面高度=" << info.ground_z << "m, 胸径=" << info.dbh
            << "m, 树高=" << info.height << "m" << endl;
    }

    cout << "开始进行像素归属和单木分割..." << endl;
    Mat pixel_owner = clusterByMaxNeighborWithTreetops(numPointsMat, treetops_for_segmentation, context);
    cout << "初步像素归属完成。" << endl;

    set<int> tree_ids_with_pixels;
    for (int y = 0; y < context.chm_height; ++y) {
        for (int x = 0; x < context.chm_width; ++x) {
            int tid = pixel_owner.at<int>(y, x);
            if (tid != -1) tree_ids_with_pixels.insert(tid);
        }
    }
    cout << "共有 " << tree_ids_with_pixels.size() << " 棵树分配到了像素。" << endl;

    map<int, vector<vector<Point2f>>> tree_contours_geo;
    cout << "正在提取树木轮廓..." << endl;
    for (int tree_id : tree_ids_with_pixels) {
        Mat mask = Mat::zeros(context.chm_height, context.chm_width, CV_8UC1);
        for (int y = 0; y < context.chm_height; ++y) {
            for (int x = 0; x < context.chm_width; ++x) {
                if (pixel_owner.at<int>(y, x) == tree_id)
                    mask.at<uchar>(y, x) = 255;
            }
        }
        vector<vector<Point>> contours_px;
        findContours(mask, contours_px, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        vector<vector<Point2f>> contours_geo_current_tree;
        for (const auto& contour : contours_px) {
            vector<Point2f> geo_pts;
            for (const auto& pt_px : contour) {
                float gx = context.min_x + (pt_px.x + 0.5f) * context.chm_resolution;
                float gy = context.min_y + (pt_px.y + 0.5f) * context.chm_resolution;
                geo_pts.emplace_back(gx, gy);
            }
            contours_geo_current_tree.push_back(geo_pts);
        }
        tree_contours_geo[tree_id] = contours_geo_current_tree;
    }

    PointCloud<PointXYZL>::Ptr labeled_cloud(new PointCloud<PointXYZL>);
    cout << "正在标记点云..." << endl;
    for (const auto& pt_orig : cloud->points) {
        PointXYZL pt_labeled;
        pt_labeled.x = pt_orig.x;
        pt_labeled.y = pt_orig.y;
        pt_labeled.z = pt_orig.z;
        pt_labeled.label = 0;

        int ix = static_cast<int>((pt_orig.x - context.min_x) / context.chm_resolution);
        int iy = static_cast<int>((pt_orig.y - context.min_y) / context.chm_resolution);

        if (ix >= 0 && ix < context.chm_width && iy >= 0 && iy < context.chm_height) {
            int owner_tree_id = pixel_owner.at<int>(iy, ix);
            if (owner_tree_id != -1) {
                bool point_in_polygon = false;
                if (tree_contours_geo.count(owner_tree_id)) {
                    const auto& contours_for_tree = tree_contours_geo[owner_tree_id];
                    for (const auto& contour_geo : contours_for_tree) {
                        if (!contour_geo.empty()) {
                            if (pointPolygonTest(contour_geo, Point2f(pt_orig.x, pt_orig.y), false) >= 0) {
                                point_in_polygon = true;
                                break;
                            }
                        }
                    }
                }
                if (point_in_polygon) {
                    pt_labeled.label = owner_tree_id;
                }
            }
        }
        labeled_cloud->push_back(pt_labeled);
    }
    cout << "点云标记完成。" << endl;



    // 输出结果到txt文件
    string result_path = path + "tree_parameters.txt";
    ofstream outfile(result_path);
    if (outfile.is_open()) {
        outfile << "ID X Y DBH Height\n";
        for (const auto& tree : tree_stats) {
            outfile << tree.id << " "
                << tree.position[0] << " " << tree.position[1] << " "
                << tree.dbh << " " << tree.height << "\n";
        }
        outfile.close();
        cout << "已保存单木参数到: " << result_path << endl;
    }
    else {
        cerr << "无法打开结果文件: " << result_path << endl;
    }

    cout << "正在保存带标签点云..." << endl;

    string savepath_pcd = path + "trees_divpro_test.pcd";
    ofstream test(savepath_pcd);
    if (!test) {
        cerr << "Cannot write to: " << savepath_pcd << endl;
        return -1;
    }
    test.close();
    io::savePCDFileASCII(savepath_pcd, *labeled_cloud);
    cout << "已保存带标签点云: " << savepath_pcd << endl;

    Mat preview_img = Mat::zeros(context.chm_height, context.chm_width, CV_8UC3);
    for (int y = 0; y < context.chm_height; ++y) {
        for (int x = 0; x < context.chm_width; ++x) {
            int tid = pixel_owner.at<int>(y, x);
            if (tid != -1) {
                unsigned char r_color = (tid * 50) % 255;
                unsigned char g_color = (tid * 90) % 255;
                unsigned char b_color = (tid * 120) % 255;
                preview_img.at<Vec3b>(y, x) = Vec3b(b_color, g_color, r_color);
            }
            else if (numPointsMat.at<int>(y, x) != 0)
                preview_img.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
        }
    }
    for (int i = 0; i < treetops_for_segmentation.size(); ++i) {
        int cx = static_cast<int>((treetops_for_segmentation[i][0] - context.min_x) / context.chm_resolution);
        int cy = static_cast<int>((treetops_for_segmentation[i][1] - context.min_y) / context.chm_resolution);
        if (cx >= 0 && cx < context.chm_width && cy >= 0 && cy < context.chm_height) {
            circle(preview_img, Point(cx, cy), 2.5, Scalar(255, 255, 255), -1);
        }
    }

    string preview_path = path + "trees_divpro_test.jpg";
    if (!preview_img.empty()) {
        imwrite(preview_path, preview_img);
        cout << "已保存分割预览图: " << preview_path << endl;
    }
    else {
        cout << "预览图为空，未保存。" << endl;
    }

    cout << "处理完成!" << endl;
    return 0;
}