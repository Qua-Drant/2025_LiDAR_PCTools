#include <iostream>         // 输入输出
#include <vector>           // 向量容器
#include <cmath>            // 数学函数（如 std::ceil, std::max）
#include <limits>           // FLT_MAX、FLT_MIN 等极值
#include <algorithm>        // std::max, std::min, std::sort 等
#include <Eigen/Dense>      // 用于点坐标表示、向量计算
#include <string>
#include <pcl/io/pcd_io.h>        // 加载和保存PCD文件
#include <pcl/point_types.h>      // PointXYZ, PointXYZI等定义
#include <pcl/point_cloud.h>      // pcl::PointCloud 容器类
#include <pcl/filters/passthrough.h>        // 高度裁剪
#include <pcl/filters/statistical_outlier_removal.h>  // 噪点过滤
#include <pcl/filters/voxel_grid.h>         // 下采样
#include <pcl/segmentation/extract_clusters.h>        // 欧式聚类
#include <pcl/surface/concave_hull.h>       // 构造树冠轮廓
#include <pcl/surface/convex_hull.h>        // 凸包
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace pcl;
using namespace cv;

// 读取原始TXT数据并提取树木点
PointCloud<PointXYZL>::Ptr loadXYZILabelTXT(const string& filename) {
    PointCloud<PointXYZL>::Ptr cloud(new PointCloud<PointXYZL>);
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "无法打开文件: " << filename << endl;
        return cloud;
    }

    string line;
    while (getline(infile, line)) {
        stringstream ss(line);
        PointXYZL pt;
        float intensity;
        ss >> pt.x >> pt.y >> pt.z >> intensity >> pt.label;
        if (pt.label != 0)  cloud->push_back(pt);
    }

    infile.close();
    cout << "读取点数: " << cloud->size() << endl;
    return cloud;
}

int main()
{
	// 读取TXT原始点云数据
    /*cout << "Please input the path of the Training TXT file: " << endl;
    string infpath;
    cin >> infpath;
    auto tree = loadXYZILabelTXT(infpath);*/

    // 读取PCD点云数据（已做二分类）
    cout << "请输入用于单木分割的PCD点云文件: " << endl;
    string infpath;
    cin >> infpath;

    PointCloud<PointXYZL>::Ptr cloud(new PointCloud<PointXYZL>);
    cout << "正在读取点云..." << endl;
    if (io::loadPCDFile<PointXYZL>(infpath, *cloud) == -1)
    {
        PCL_ERROR("读取点云失败\n");
        return (-1);
    }
    else cout << "读取点云成功！" << endl;


	/*------CHM（冠层高度模型）计算------*/
    const float res = 0.1f;     //栅格分辨率    
    float min_x = FLT_MAX;
    float max_x = FLT_MIN;
    float min_y = FLT_MAX;
    float max_y = FLT_MIN;
    float min_z = FLT_MAX;
	cout << "正在由点云计算 CHM（冠层高度模型）..." << endl;

    for (const auto& pt : cloud->points) {
        if (pt.x < min_x) min_x = pt.x;
        if (pt.x > max_x) max_x = pt.x;
        if (pt.y < min_y) min_y = pt.y;
        if (pt.y > max_y) max_y = pt.y;
        if (pt.z < min_z) min_z = pt.z;
    }

    int w = ceil((max_x - min_x) / res);
    int h = ceil((max_y - min_y) / res);

    // 初始化 CHM(冠层高度模型)，高度初始化为0
    Mat chm = Mat::zeros(h, w, CV_32F);
	// 将点云投影到 CHM 上
    for (auto& pt : *cloud) {
        int ix = (pt.x - min_x) / res;
        int iy = (pt.y - min_y) / res;
        pt.z = pt.z - min_z; //进行高度归一化
        float& cell = chm.at<float>(iy, ix);
        cell = max(cell, pt.z);
    }
	cout << "CHM 计算完成！" << endl;


    /*------单木估计与最大高度提取------*/
    cout << "正在进行单木位置估计与最大树高提取..." << endl;

    // 对 chm 进行 3x3 均值平滑
    Mat chm_smooth = chm.clone();
    blur(chm, chm_smooth, Size(3, 3));

	// 将树顶点信息存储到treetops, 存储树的中心点和最大高度
    vector<Eigen::Vector3f> treetops; // 存储 [x,y,h]
    int win = 4; // 搜索半径
    for (int y = win; y < h - win; ++y) {
        for (int x = win; x < w - win; ++x) {
            float v = chm_smooth.at<float>(y, x);
            bool local_max = (v > 0);
            for (int dy = -win; dy <= win && local_max; ++dy)
                for (int dx = -win; dx <= win; ++dx)
                    if (chm_smooth.at<float>(y+dy, x+dx) > v)
                    {
                        local_max = false; break;
                    }
            if (local_max) {
                float gx = min_x + (x + 0.5f) * res;
                float gy = min_y + (y + 0.5f) * res;
                treetops.emplace_back(gx, gy, chm.at<float>(y, x)); // 用原始 CHM 高度
            }
        }
    }
    // 将距离小于1m的treetops合成为一个treetops,位置取平均，高度取最大

    // 合并距离小于1m的树顶点(可注释)
    vector<Eigen::Vector3f> merged_treetops;
    vector<bool> merged(treetops.size(), false);
    const float merge_dist2 = 1.0f; 

    for (size_t i = 0; i < treetops.size(); ++i) {
        if (merged[i]) continue;
        vector<Eigen::Vector3f> group;
        group.push_back(treetops[i]);
        merged[i] = true;
        for (size_t j = i + 1; j < treetops.size(); ++j) {
            if (merged[j]) continue;
            float dx = treetops[i][0] - treetops[j][0];
            float dy = treetops[i][1] - treetops[j][1];
            float d2 = dx * dx + dy * dy;
            if (d2 < merge_dist2) {
                group.push_back(treetops[j]);
                merged[j] = true;
            }
        }
        // 计算合并后的中心和最大高度
        float sum_x = 0, sum_y = 0, max_h = 0;
        for (const auto& t : group) {
            sum_x += t[0];
            sum_y += t[1];
            if (t[2] > max_h) max_h = t[2];
        }
        float avg_x = sum_x / group.size();
        float avg_y = sum_y / group.size();
        merged_treetops.emplace_back(avg_x, avg_y, max_h);
    }
    treetops = merged_treetops;

    // 优化实现：采用树底的方式提取树木估计位置
    /*--------------BEGIN----------------*/










    /*----------------END----------------*/


    // 可视化树顶点
    /*for (const auto& t : treetops) {
        int cx = (t[0] - min_x) / res;
        int cy = (t[1] - min_y) / res;
        circle(chm_smooth, Point(cx, cy), 3, Scalar(255), -1);
    }
    imshow("CHM with Treetops", chm_smooth / (chm_smooth.empty() ? 1 : cv::norm(chm_smooth, NORM_INF)));
    waitKey(1);*/

    cout << "单木信息已保存！" << endl;

    /*------确认初步像素归属------*/
    cout << "正在生成圆形缓冲区，进行初步归属..." << endl;
  
	Mat pixel_owner = Mat::zeros(h, w, CV_32F); // 记录每个像素所属的树
    for (int i = 0; i < treetops.size(); ++i) {
        float r = treetops[i][2] * 0.6f; // 影响半径：60% 树高

        // 计算像素坐标
        int r_pix = ceil(r / res);
        int cx = (treetops[i][0] - min_x) / res;
        int cy = (treetops[i][1] - min_y) / res;

        // 在二维栅格上遍历
        for (int dy = -r_pix; dy <= r_pix; ++dy)
            for (int dx = -r_pix; dx <= r_pix; ++dx) {
                int x = cx + dx, y = cy + dy;
                if (x < 0 || y < 0 || x >= w || y >= h) continue; // 控制边界

                // 转到真实坐标
                float dxm = dx * res, dym = dy * res;
                if (dxm * dxm + dym * dym <= r * r)
                    pixel_owner.at<float>(y,x) = i; // 根据每棵树的影响半径，将像素点分配给最近的树，完成初步的像素归属。
            }
    }

/*  这段代码对每个像素点 (x, y) 进行遍历。
•	如果该像素未被任何树归属（pixel_owner[y][x] < 0），则跳过。
•	对于已归属的像素，计算其地理坐标 (px, py)。
•	遍历所有树，计算该像素到每棵树中心的欧氏距离平方 d2，找到最近的树（距离最小的树）。
•	将该像素最终归属到最近的树（pixel_owner[y][x] = best_i;）。
•	这样做的目的是解决像素在多个树影响范围重叠时的归属问题，确保每个像素只归属于最近的树。*/

/*  可以试试把这段调整到点云标记分割之后实现，把未标记的点云分到最近的树，解决一下栅格边界未被分割的问题  */
/*-------------------------------------------------------------------------------------------*/
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (pixel_owner.at<float>(y, x) == 0) continue;

            // 对有初步归属的像素，选距离最近的树顶
            float best_d2 = FLT_MAX; int best_i = -1;
            float px = min_x + (x + 0.5f) * res;
            float py = min_y + (y + 0.5f) * res;

            for (int i = 0; i < treetops.size(); ++i) {
                float dx = px - treetops[i][0];
                float dy = py - treetops[i][1];
                float d2 = dx * dx + dy * dy;
                if (d2 < best_d2) { best_d2 = d2; best_i = i; }
            }
            pixel_owner.at<float>(y, x) = best_i;
        }
    }
/*-------------------------------------------------------------------------------------------*/

    //存储所有被分配到像素的树的编号
    set<int> tree_ids; 
    cout << "正在分配分类编号..." << endl;
    // 遍历所有像素点，如果该像素归属于某棵树（tid >= 0），则将该树编号加入集合。
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int tid = pixel_owner.at<float>(y, x);
            if (tid >= 0) tree_ids.insert(tid); // 使用集合可以自动去重，最终得到所有实际分配到像素的树的编号列表。
        }
    }

    map<int, vector<vector<Point2f>>> tree_contours_geo;
	cout << "正在提取树木轮廓..." << endl;
    for (int tree_id : tree_ids) {
        // 创建二值 mask
        Mat mask = Mat::zeros(h, w, CV_8UC1);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                if (pixel_owner.at<float>(y, x) == tree_id)
                    mask.at<uchar>(y, x) = 255;
            }
        }

        // 提取轮廓（像素坐标）
        vector<vector<Point>> contours_px;
        findContours(mask, contours_px, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // 转换为真实坐标
        vector<vector<Point2f>> contours_geo;
        for (const auto& contour : contours_px) {
            vector<Point2f> geo;
            for (const auto& pt : contour) {
                float gx = min_x + (pt.x + 0.5f) * res;
                float gy = min_y + (pt.y + 0.5f) * res;
                geo.emplace_back(gx, gy);
            }
            contours_geo.push_back(geo);
        }
        tree_contours_geo[tree_id] = contours_geo;
    }

    // 创建二维栅格预览图
    Mat preview = Mat::zeros(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int tid = pixel_owner.at<float>(y, x);
            if (tid >= 0) {
                uchar color = (tid * 53) % 255;
                preview.at<Vec3b>(y, x) = Vec3b(color, color, 255 - color);
            }
        }

    // 创建带单木分类标签的点云
    PointCloud<PointXYZL>::Ptr labeled_cloud(new PointCloud<PointXYZL>);
	cout << "正在标记点云..." << endl;
    // 遍历原始点云
    for (const auto& pt : cloud->points) {
        PointXYZL pt_labeled;
        pt_labeled.x = pt.x;
        pt_labeled.y = pt.y;
        pt_labeled.z = pt.z;
        pt_labeled.label = 0; // 默认未分配

        // 判断该点是否属于某个树冠
        for (const auto& kv : tree_contours_geo) {
            int tree_id = kv.first;
            const auto& contours = kv.second;
            // 检查每个轮廓
            for (const auto& contour : contours) {
                // 使用OpenCV的pointPolygonTest判断点是否在多边形内
                vector<Point2f> poly = contour;
                Point2f pt2d(pt.x, pt.y);
                if (pointPolygonTest(poly, pt2d, false) >= 0) {
                    pt_labeled.label = tree_id + 1; // label为树编号+1，0为未分配
                    break;
                }
            }
            if (pt_labeled.label > 0) break;
        }
        labeled_cloud->push_back(pt_labeled);
    }

    // 保存带标签的点云
    cout << "分类后点云已生成！请输入结果的保存路径: ";
    string savepath;
    cin >> savepath;
    io::savePCDFileASCII(savepath, *labeled_cloud);
    cout << "已保存带标签点云: " << savepath << endl;

    cout << "请输入保存二值栅格预览结果的路径:";
    string reviewpath;
    cin >> reviewpath;
    imwrite(reviewpath, preview);

    // 保存标记好的树木点点云数据
    //cout << "Please input the path of the saved file: " << endl;
    //string savepath;
    //cin >> savepath;
    //io::savePCDFile(savepath, *tree);
}
