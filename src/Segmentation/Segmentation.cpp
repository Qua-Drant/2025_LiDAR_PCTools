// Segmentation.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。

#include <iostream>
#include <string>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include "RegionGrowing.h"
#include "RANSACPlane.h"
#include <numeric>

using namespace std;
using namespace pcl;

int main()
{
	// 读取点云数据
	cout << "Please input the path of the PCD file: " << endl;
	string infpath;
	cin >> infpath;
	
	PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
	if (io::loadPCDFile<PointXYZ>(infpath, *cloud) == -1)
	{
		PCL_ERROR("failed to load PCD file.\n");
		return -1;
	}

	vector<vector<int>> clusters;
	vector<int> unclustered;

	cout << "Choose the segmentation method: RegionGrowing(rg) / RANSAC(rp)" << endl;
	string method;
	cin >> method;

	if (method == "rg")
	{
		// 设置参数
		Params par;
		par.k = 20;
		par.neighbour_num = 20;
		par.min_segment_size = 30;
		par.theta_threshold = 15.0 / 180 * M_PI;
		par.curvature_threshold = 0.7;
		// 实例化区域生长对象，并执行算法
		RegionGrowing rg(cloud, par);
		rg.doRegionGrowing(clusters, unclustered);
	}
	else if (method == "rp")
	{
		// 实例化RANSAC平面分割对象
		RANSACParams par;
		par.distance_threshold = 0.05;
		par.error_threshold = 0.01;
		par.max_iteration_num = 500;
		par.min_pts = 50;

		RANSACPlane rp(cloud, par);
		rp.segment(clusters, unclustered);
	}

	// 保存分割结果
	PointCloud<PointXYZL>::Ptr segmented_cloud(new PointCloud<PointXYZL>);
	copyPointCloud(*cloud, *segmented_cloud);
	for (int i = 0; i < clusters.size(); i++)
	{		
		for (int j = 0; j < clusters[i].size(); j++)
		{
			segmented_cloud->points[clusters[i][j]].label = i+1;
		}
	}
	cout << "Please input the path to save the ground points as PCD file: " << endl;
	string outpath;
	cin >> outpath;
	io::savePCDFile(outpath, *segmented_cloud);

	// 将分割结果转化为彩色点云
	PointCloud<PointXYZRGB>::Ptr colored_cloud(new PointCloud<PointXYZRGB>());
	copyPointCloud(*cloud, *colored_cloud);
	for (int i = 0; i < clusters.size(); i++)
	{
		// 随机生产RGB值
		int r = rand() % 255;
		int g = rand() % 255;
		int b = rand() % 255;
		for (int j = 0; j < clusters[i].size(); j++)
		{
			colored_cloud->points[clusters[i][j]].r = r;
			colored_cloud->points[clusters[i][j]].g = g;
			colored_cloud->points[clusters[i][j]].b = b;
		}
	}

	for (int i = 0; i < unclustered.size(); i++)
	{
		// 将未聚类的点设置为黑色
		colored_cloud->points[unclustered[i]].r = 0;
		colored_cloud->points[unclustered[i]].g = 0;
		colored_cloud->points[unclustered[i]].b = 0;
	}

	// 可视化分割结果
	cout << "Visualization? (y/n): " << endl;
	char choice;
	cin >> choice;
	if (choice == 'y')
	{
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Cloud Viewer1"));
		viewer->setBackgroundColor(0.5, 1.0, 1.0);
		viewer->addPointCloud(colored_cloud, "point cloud");
		viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2, "point cloud");
		viewer->addCoordinateSystem();
		viewer->spin();
	}
	return 1;
}

