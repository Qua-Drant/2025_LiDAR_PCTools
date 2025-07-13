#pragma once
#include <vector>
#include <string>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

using namespace std;
using namespace pcl;

struct Params {
	float resolution;	  //栅格化的分辨率，即栅格中每个网格的大小
	int windowMax;	      //形态学运算的最大窗口尺寸(以网格个数为单位)、
	float slopeTol;       //坡度的容差范围，用正切值表示 slopeTol = tan(slope)
	float elevationTh;    //高度阈值
	float scalingFactor;  //与坡度相关的尺度因子，eth = elevationTh + slope * scalingFactor
};


class morphFilter
{
private:
	PointCloud<PointXYZ>::Ptr cloud; //输入点云
	Params params; //参数
	PointXYZ min, max;
	float* grid_data;
	int rows, cols;

public:
	// 构造函数
	morphFilter(PointCloud<PointXYZ>::Ptr& pc, Params par);
	// 析构函数
	~morphFilter();

	// 执行形态学滤波的API
	int doFiltering(vector<int>& ground_indices, vector<int>& nonground_indices);

private:
	// 对点云进行栅格化
	int Rasterization();
	// 填补栅格中的孔洞
	float fillHole(const int& row, const int& col, float* grid);
	// 开运算
	int open(int win_size, float* grid);
	// 腐蚀
	void erosion(int win_size, float* grid);
	// 膨胀
	void dilation(int win_size, float* grid);
	// 根据高程格网所记录的高程值和坡度信息，分离地面点和非地面点
	int seperatePoints(vector<int>& ground_indices, vector<int>& nonground_indices);

};

