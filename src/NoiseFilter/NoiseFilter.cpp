// NoiseFilter.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。

#include <iostream>
#include <fstream>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_search.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>


using namespace std;
using namespace pcl;


void pclSorFilter(const PointCloud<PointXYZ>::Ptr& cloud,  //输入点云
    const int& k,       //KNN邻近搜索的K值
    const int& mult,    //标准差的倍数
    PointCloud<PointXYZ>::Ptr& inliers,     //非噪声点
    PointCloud<PointXYZ>::Ptr& outliers)    //噪声点
{
    //创建滤波器
    StatisticalOutlierRemoval<PointXYZ> sor(true);
    //设置参数
    sor.setInputCloud(cloud);
    sor.setMeanK(k);
    sor.setStddevMulThresh(mult);
    sor.filter(*inliers);

    //得到噪声点
    PointIndices outlier_indices;
    sor.getRemovedIndices(outlier_indices);
    copyPointCloud(*cloud, outlier_indices, *outliers);
}


void mySorFilter(const PointCloud<PointXYZ>::Ptr& cloud,  //输入点云
    const int& k,       //KNN邻近搜索的K值
    const int& mult,    //标准差的倍数
    PointCloud<PointXYZ>::Ptr& inliers,     //非噪声点
    PointCloud<PointXYZ>::Ptr& outliers)    //噪声点
{
    // 第一步：建立索引
    cout << "请选择建立索引的方式: 1.kd树索引 2.octree索引" << endl;
    int index_type;
    cin >> index_type;

    // 第二步：声明一个vector<float>类型变量，用于存放每个点对应的Kdist
    vector<float> Kdistances;

    /*---------------kd树索引--------------*/
    if (index_type == 1)
    {
        clock_t time0 = clock();

        KdTreeFLANN<PointXYZ> kdtree;
        kdtree.setInputCloud(cloud);

        clock_t time1 = clock();
        cout << "共耗时：" << (time1 - time0) / 1000.0 << "s。" << endl;

        // 第三步：循环遍历每个点，计算每个点的Kdist
        cout << "请选择搜索方式：1.KNN搜索 2.半径搜索" << endl;
        int search_type;
        cin >> search_type;

        /*-------------KNN搜索-------------*/
        if (search_type == 1) {
            for (int i = 0; i < cloud->size(); i++)
            {
                vector<int> nb_indices; //存储临近点的索引号
                vector<float> nb_distances; //存储临近点到查询点的距离平方

                PointXYZ p = cloud->points[i];
                kdtree.nearestKSearch(p, k, nb_indices, nb_distances);//KNN搜索
                // 计算Kdist
                float distance = 0;
                for (int j = 0; j < k; j++)  distance += sqrt(nb_distances[j]);
                float Kdist = distance / k;
                Kdistances.push_back(Kdist);
            }
        }

        /*-------------半径搜索-------------*/
        else if (search_type == 2) {
            for (int i = 0; i < cloud->size(); i++)
            {
                vector<int> nb_indices; //存储临近点的索引号
                vector<float> nb_distances; //存储临近点到查询点的距离平方

                cout << "请输入半径搜索的半径：";
                float radius;
                cin >> radius;

                PointXYZ p = cloud->points[i];
                kdtree.radiusSearch(p, radius, nb_indices, nb_distances);//半径搜索
                // 计算Kdist
                float distance = 0;
                for (int j = 0; j < nb_indices.size(); j++)  distance += sqrt(nb_distances[j]);
                float Kdist = distance / nb_indices.size();
                Kdistances.push_back(Kdist);
            }
        }
    }

    /*---------------octree索引------------*/
    else if (index_type == 2)
    {

        clock_t time0 = clock();

        octree::OctreePointCloudSearch<PointXYZ> octree(0.2);
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();

        clock_t time1 = clock();
        cout << "共耗时：" << (time1 - time0) / 1000.0 << "s。" << endl;

        // 第三步：循环遍历每个点，计算每个点的Kdist
        cout << "请选择搜索方式：1.KNN搜索 2.半径搜索" << endl;
        int search_type;
        cin >> search_type;

        /*-------------KNN搜索-------------*/
        if (search_type == 1) {
            for (int i = 0; i < cloud->size(); i++)
            {
                vector<int> nb_indices; //存储临近点的索引号
                vector<float> nb_distances; //存储临近点到查询点的距离平方

                PointXYZ p = cloud->points[i];
                octree.nearestKSearch(p, k, nb_indices, nb_distances);//KNN搜索
                // 计算Kdist
                float distance = 0;
                for (int j = 0; j < k; j++)  distance += sqrt(nb_distances[j]);
                float Kdist = distance / k;
                Kdistances.push_back(Kdist);
            }
        }

        /*-------------半径搜索--------------*/
        else if (search_type == 2) {
            for (int i = 0; i < cloud->size(); i++)
            {
                vector<int> nb_indices; //存储临近点的索引号
                vector<float> nb_distances; //存储临近点到查询点的距离平方

                cout << "请输入半径搜索的半径：";
                float radius;
                cin >> radius;

                PointXYZ p = cloud->points[i];
                octree.radiusSearch(p, radius, nb_indices, nb_distances);//半径搜索
                // 计算Kdist
                float distance = 0;
                for (int j = 0; j < nb_indices.size(); j++)  distance += sqrt(nb_distances[j]);
                float Kdist = distance / nb_indices.size();
                Kdistances.push_back(Kdist);
            }
        }
    }

    // 第四步：计算所有Kdists的平均meanKdist与标准差stddevKdist
    float meanKDist = accumulate(Kdistances.begin(), Kdistances.end(), 0.0) / Kdistances.size(); //计算平均Kdist
    float stddevKDist = sqrt(inner_product(Kdistances.begin(), Kdistances.end(), Kdistances.begin(), 0.0) / Kdistances.size() - meanKDist * meanKDist);

    // 第五步：设定阈值 threshold = meanKDist + mult * stddevKDist
    float threshold = meanKDist + mult * stddevKDist;

    // 判断每个点的Kdist是否大于阈值，将大于阈值的点作为噪声点，将小于阈值的点作为非噪声点
    for (int idx = 0; idx < cloud->size(); idx++)
    {
        if (Kdistances[idx] > threshold) outliers->push_back(cloud->points[idx]);
        else inliers->push_back(cloud->points[idx]);
    }

}

int main()
{
    // 1. 读取点云文件
    cout << "请输入要读取的PCD点云文件路径（路径中不要有空格）：" << endl;
    string infpath;
    cin >> infpath;
    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    if (io::loadPCDFile<PointXYZ>(infpath, *cloud) == -1)
    {
        PCL_ERROR("读取点云失败\n");
        return (-1);
    }
    cout << "读取点云成功！" << endl;


    // 2. 调用滤波函数
    cout << "请输入KNN邻近搜索的K值：";
    int k;
    cin >> k;
    cout << "请输入视为噪声的中误差倍数：";
    int mult;
    cin >> mult;
    cout << "正在估算噪声点..." << endl;

    PointCloud<PointXYZ>::Ptr inliers(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr outliers(new PointCloud<PointXYZ>);

    clock_t time0 = clock();
    // pclSorFilter(cloud, k, mult, inliers, outliers);
    mySorFilter(cloud, k, mult, inliers, outliers);
    clock_t time1 = clock();

    cout << "共耗时：" << (time1 - time0) / 1000.0 << "s。" << endl;

    // 3.保存去噪结果
    cout << "请输入要保存的PCD文件路径（路径中不要有空格）：" << endl;
    string outfpath;
    cin >> outfpath;
    io::savePCDFile(outfpath, *inliers);
    cout << "保存成功！" << endl;

    // 4. 可视化
    cout << "是否需要可视化？（y/n）" << endl;
    char vis;
    cin >> vis;
    if (vis == 'y' || vis == 'Y')
    {
        visualization::PCLVisualizer::Ptr viewer(new visualization::PCLVisualizer("Cloud Viewer1"));
        viewer->setBackgroundColor(0.5, 1.0, 1.0);

        visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_color2(inliers, 0, 0, 255); //蓝色显示非噪声点
        viewer->addPointCloud(inliers, cloud_color2, "PointCloud");

        visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_color(outliers, 255, 0, 0); //红色显示噪声点
        viewer->addPointCloud(outliers, cloud_color, "PointCloud2");

        viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2, "PointCloud"); //设置点大小
        viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2, "PointCloud2"); //设置点大小
        viewer->addCoordinateSystem();
        viewer->spin();
    }
}