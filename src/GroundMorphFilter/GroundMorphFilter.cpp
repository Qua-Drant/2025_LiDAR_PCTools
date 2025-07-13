// GroundMorphFilter.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
#include <iostream>
#include <string>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/common.h>
#include "morthFilter.h"

using namespace std;
using namespace pcl;


morphFilter::morphFilter(PointCloud<PointXYZ>::Ptr& pc, Params par)
{
	cloud = pc;
	params = par;
	grid_data = NULL;

}

morphFilter::~morphFilter()
{
	cloud->resize(0);
	delete[] grid_data;
}

int morphFilter::Rasterization()
{
	if (cloud == NULL)
		return 0;

	if (cloud->empty())
		return 0;

	// 获取点云的最大最小XYZ值
	getMinMax3D(*cloud, min, max);
	
	//根据最大最小XY值，确定栅格化的行列数
	rows = int((max.y - min.y) / params.resolution) + 1;//取整+1
	cols = int((max.x - min.x) / params.resolution) + 1;//取整+1
	cout << "The size of the raster:" << rows << " * " << cols << endl;

	// 初始化网格数据
	if (grid_data != NULL)
		delete[] grid_data;
	grid_data = new float[rows * cols];
	fill_n(grid_data, rows * cols, FLT_MAX);

	// 遍历点云中的每个点p,若点p的高程值小于它所落入的网格所记录的高程值
	// 则更新对应网格中的高程值
	for (const auto& p : cloud->points)
	{
		int r = (p.y - min.y) / params.resolution;
		int c = (p.x - min.x) / params.resolution;
		float v = grid_data[r * cols + c];
		grid_data[r * cols + c] = p.z < v ? p.z : v;
	}

	// 根据邻近网格中的高程值对空洞进行填补
	for(int r=0; r < rows; r++)
		for (int c = 0; c < cols; c++)
		{
			if (grid_data[r * cols + c] == FLT_MAX)
				grid_data[r * cols + c] = fillHole(r, c, grid_data);
		}

		cout << "Rasterization finished!" << endl;
		return 1;
}

float morphFilter::fillHole(const int& row, const int& col, float* grid)
{
	int level = 1;
	while (true)
	{
		// 采用螺旋搜索方式近似获取当前空网格的临近网格
		for (int r = row - level; r <= row + level ; r += 2 * level - 1)
			for (int c = col - level; c <= col + level; c += 2 * level - 1)
			{
				if (c >= 0 && c < cols && r >= 0 && r < rows)
					if (grid[r * cols + c] < FLT_MAX)
						return grid[r * cols + c];
			}
		level++;
	}
}

int morphFilter::open(int win_size, float* grid)
{
	// 腐蚀
	erosion(win_size, grid);
	// 膨胀
	dilation(win_size, grid);
	return 1;
}

void morphFilter::erosion(int win_size, float* grid)
{
	// 创建一个临时栅格，并将腐蚀运算前的栅格数据拷贝到临时栅格中
	float* grid_tmp = new float[rows * cols];
	memcpy(grid_tmp, grid, sizeof(float) * rows * cols);

	// 遍历栅格数据
	for (int row = 0; row < rows; row++)
		for (int col = 0; col < cols; col++)
		{
			float v = grid[row * cols + col];
			// 对当前网格进行窗口大小为win_size的腐蚀运算
			for (int r = row - win_size; r <= row + win_size; r++)
				for (int c = col - win_size; c <= col + win_size; c++)
				{
					if (r >= 0 && r < rows && c >= 0 && c < cols)
					{
						float v_tmp = grid_tmp[r * cols + c];
						if (v_tmp < v) grid[row * cols + col] = v_tmp;
					}
				}
		}

	// 释放临时栅格
	delete[] grid_tmp;
}

void morphFilter::dilation(int win_size, float* grid)
{
	// 创建一个临时栅格，并将膨胀运算前的栅格数据拷贝到临时栅格中
	float* grid_tmp = new float[rows * cols];
	memcpy(grid_tmp, grid, sizeof(float) * rows * cols);

	// 遍历栅格数据
	for (int row = 0; row < rows; row++)
		for (int col = 0; col < cols; col++)
		{
			float v = grid[row * cols + col];
			// 对当前网格进行窗口大小为win_size的膨胀运算
			for (int r = row - win_size; r <= row + win_size; r++)
				for (int c = col - win_size; c <= col + win_size; c++)
				{
					if (r >= 0 && r < rows && c >= 0 && c < cols)
					{
						float v_tmp = grid_tmp[r * cols + c];
						if (v_tmp > v) grid[row * cols + col] = v_tmp;
					}
				}
		}

	// 释放临时栅格
	delete[] grid_tmp;
}

int morphFilter::seperatePoints(vector<int>& ground_indices, vector<int>& nonground_indices)
{
	// 定义一个rows * cols大小的数组，用来存储对应高程栅格的坡度值
	// 并将其中的坡度值均初始化为1
	float* grid_slope = new float[rows * cols];
	fill_n(grid_slope, rows * cols, 1.0);

	// 计算栅格中每个网格的坡度值
	for (int row = 1; row < rows - 1; row++)
		for (int col = 1; col < cols - 1; col++)
		{
			float a = grid_data[(row - 1) * cols + (col - 1)];
			float b = grid_data[row * cols + (col - 1)];
			float c = grid_data[(row + 1) * cols + (col - 1)];
			float d = grid_data[(row - 1) * cols + col];
			float e = grid_data[row * cols + col];
			float f = grid_data[(row + 1) * cols + col];
			float g = grid_data[(row - 1) * cols + (col + 1)];
			float h = grid_data[row * cols + (col + 1)];
			float i = grid_data[(row + 1) * cols + (col + 1)];

			double dz_dx = (c + 2 * f + i - a - 2 * d - g) / (8 * params.resolution);
			double dz_dy = (g + 2 * h + i - a - 2 * b - c) / (8 * params.resolution);

			grid_slope[row * cols + col] = atan(sqrt(dz_dx * dz_dx + dz_dy * dz_dy));
		}

	// 遍历点云中的点p,对于点p判断它所落入的高程网格所记录的高程值之间的高差
	// 是否小于阈值？小于则记为地面点，否则为非地面点
	ground_indices.resize(0);
	nonground_indices.resize(0);
	for (int i = 0; i < cloud->size(); i++)
	{
		// 计算当前点所属网格
		int row = (cloud->points[i].y - min.y) / params.resolution;
		int col = (cloud->points[i].x - min.x) / params.resolution; 
		
		// 当前点与网格记录的高程之间的高程差
		float dz = abs(cloud->points[i].z - grid_data[row * cols + col]);

		// 阈值
		float elevationThreshold = params.elevationTh + grid_slope[row * cols + col] * params.scalingFactor;

		if (dz <= elevationThreshold) ground_indices.push_back(i);
		else nonground_indices.push_back(i);
	}

	// 释放坡度栅格
	delete[] grid_slope;

	return 1;
}

int morphFilter::doFiltering(vector<int>& ground_indices, vector<int>& nonground_indices)
{
	// 1. 对点云数据进行栅格化
	Rasterization();

	// 2. 迭代滤波，更新高程栅格数据
	// ① 首先定义一个临时栅格，用来存放开运算前的高程栅格数据
	float* grid_tmp = new float[rows * cols];
	// ② 从窗口半径为1个网格开始，对栅格进行开运算，并更新高程栅格数据，直到到达最大窗口尺寸
	for (int win_size = 1; win_size < params.windowMax; win_size++)
	{
		clock_t time0 = clock(); // 记录单次迭代开始的时间
		cout << "-------------------------------------" << endl;
		cout << "Curent size of the morphology window:" << win_size << endl;

		// 将当前高程栅格数据中的数据拷贝到临时栅格中
		memcpy(grid_tmp, grid_data, sizeof(float) * rows * cols);

		// 在grid_tmp中进行开运算
		open(win_size, grid_tmp);

		// 遍历网格，计算开运算前后的高差值，并与高差阈值作比较
		float ele_threshold = params.resolution * win_size * params.slopeTol; // 与窗口大小相关的高差阈值
		for (int row = 0; row < rows; row++)
			for (int col = 0; col < cols; col++)
			{
				float elevation_before = grid_data[row * cols + col];
				float elevation_after = grid_tmp[row * cols + col];
				// 若开运算前后的高差大于高差阈值，说明该网格为非地面点
				// 需要将grid_data中的数据更新为开运算后的高程，否则不改变高程
				grid_data[row * cols + col] = abs(elevation_after - elevation_before) > ele_threshold ?
					elevation_after : elevation_before;
			}

		clock_t time1 = clock(); // 记录单次迭代结束的时间
		cout << "Iteration" << win_size << "takes time:" << (time1 - time0) / 1000.0 << "s" << endl << endl;
	}

	// 释放临时栅格
	delete[] grid_tmp;

	// 3. 分离地面点与非地面点
	seperatePoints(ground_indices, nonground_indices);

	return 1;
}


int main()
{
	// 读取点云数据
	cout << "Please input the path of the PCD file:" << endl;
	string infpath;
	cin >> infpath;

	PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>());
	if (io::loadPCDFile<PointXYZ>(infpath, *cloud) == -1)
	{
		PCL_ERROR("Failed to load the PCD file!\n");
		return -1;
	}

	// 设置点云滤波参数（hard coding)， 也可改为用户输入
	Params par;
	par.resolution = 2.0; // 栅格化的分辨率
	par.windowMax = 15; // 形态学运算的最大窗口尺寸
	par.slopeTol = 0.3; // 坡度容差范围
	par.elevationTh = 0.5; // 高差阈值
	par.scalingFactor = 1.05; // 与坡度相关的尺度因子

	// 进行SMRF地面滤波
	vector<int> ground_indices, nonground_indices;
	morphFilter filter(cloud, par);
	filter.doFiltering(ground_indices, nonground_indices);

	//保存滤波结果
	PointCloud<PointXYZ>::Ptr ground_cloud(new PointCloud<PointXYZ>());
	PointCloud<PointXYZ>::Ptr nonground_cloud(new PointCloud<PointXYZ>());
	copyPointCloud(*cloud, ground_indices, *ground_cloud);
	copyPointCloud(*cloud, nonground_indices, *nonground_cloud);

	cout << "Please input the path to save the ground points as PCD files:" << endl;
	string outfpath1;
	cin >> outfpath1;
	io::savePCDFile(outfpath1, *ground_cloud);

	cout << "Please input the path to save the non-ground points as PCD files:" << endl;
	string outfpath2;
	cin >> outfpath2;
	io::savePCDFile(outfpath2, *nonground_cloud);

	// 可视化
	cout << "Visulization? y(yes) or n(no):" << endl;
	char tag;
	cin >> tag;
	if (tag == 'y')
	{
		boost::shared_ptr<visualization::PCLVisualizer> viewer(new visualization::PCLVisualizer("cloud Viewerl"));
		viewer->setBackgroundColor(0.5, 1.0, 1.0);
		visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_color1 (ground_cloud, 0, 0, 255);//蓝色显示地面点
		viewer->addPointCloud(ground_cloud, cloud_color1, "PointCloud1");
		visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_color2 (nonground_cloud,255,0,0);//红色显示非地面点
		viewer->addPointCloud(nonground_cloud, cloud_color2, "PointCloud2");
		viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2, "PointCloud1");
		viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2, "PointCloud2");
		viewer->addCoordinateSystem();
		viewer->spin();
	} 

	cout << "The filtering is finished!" << endl;
	return 1;
}