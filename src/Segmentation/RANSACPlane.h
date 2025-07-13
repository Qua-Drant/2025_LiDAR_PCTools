#pragma once
#include <vector>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/common/eigen.h>

using namespace std;
using namespace pcl;
using namespace Eigen;


struct RANSACParams {
	int min_pts;
	int max_iteration_num;
	float distance_threshold;
	float error_threshold;
};

class RANSACPlane
{

private:
	PointCloud<PointXYZ>::Ptr cloud;
	RANSACParams param;

public:
	RANSACPlane(const PointCloud<PointXYZ>::Ptr& pc, const RANSACParams& pa);
	~RANSACPlane();
	int segment(vector<vector<int>>& clusters, vector<int>& unclustered);

// private:
	bool computePlaneCoefficients(int* pt_indices, Vector4f& coefficient);
	void findUnclusteredPointsWithinDistance(const Vector4f& coefficient,
		const vector<int>& unclustered, vector<int>& inliers, float& distance);

};

