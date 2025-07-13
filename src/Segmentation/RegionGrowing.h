#pragma once
#include <vector>
#include <string>
#include <pcl/point_types.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>

using namespace std;
using namespace pcl;


struct Params {
	int k;
	int neighbour_num;
	int min_segment_size;
	float theta_threshold;
	float curvature_threshold;
};


class RegionGrowing
{
private:
	PointCloud<PointXYZ>::Ptr cloud;
	Params para;

	search::Search<PointXYZ>::Ptr tree;
	PointCloud<Normal>::Ptr normals;

public:
	RegionGrowing(const PointCloud<PointXYZ>::Ptr& pc, const Params& pa);

	~RegionGrowing();

	// 执行区域生长算法的API
	int doRegionGrowing(vector<vector<int>>& clusters, vector<int>& unclustered);

private:
	vector<int> sortCurvatures();
};

