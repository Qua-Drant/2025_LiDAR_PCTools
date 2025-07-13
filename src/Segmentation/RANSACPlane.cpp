#include "RANSACPlane.h"

using namespace std;
using namespace pcl;
using namespace Eigen;

RANSACPlane::RANSACPlane(const PointCloud<PointXYZ>::Ptr& pc, const RANSACParams& pa)
{
	param = pa;
	cloud.reset(new PointCloud<PointXYZ>);
	copyPointCloud(*pc, *cloud);
}

RANSACPlane::~RANSACPlane()
{
	if (cloud)
		cloud->resize(0);
}


// 解算平面系数函数computePlaneCoefficients()
/*入参：pt_indices, cloud中3个点的索引号  出参：coefficient, 拟合平面的系数  返回：3点共面时返回true, 否则返回false */
bool RANSACPlane::computePlaneCoefficients(int* pt_indices, Vector4f& coefficient)
{
	if (pt_indices[0] == pt_indices[1] || pt_indices[1] == pt_indices[2] || pt_indices[2] == pt_indices[0])
		return false;
	// 获取3点坐标
	Array4fMap p0 = (*cloud)[pt_indices[0]].getArray4fMap();
	Array4fMap p1 = (*cloud)[pt_indices[1]].getArray4fMap();
	Array4fMap p2 = (*cloud)[pt_indices[2]].getArray4fMap();

	// 检查3点是否共线
	Array4f p1p0 = p1 - p0;
	Array4f p2p0 = p2 - p0;
	Array4f dy1dy2 = p1p0 / p2p0;
	if ((dy1dy2[0] == dy1dy2[1] && dy1dy2[2] == dy1dy2[1]))
		return false;

	// 计算3点平面的法向量
	coefficient[0] = (p1p0[1] * p2p0[2] - p1p0[2] * p2p0[1]);
	coefficient[1] = (p1p0[2] * p2p0[0] - p1p0[0] * p2p0[2]);
	coefficient[2] = (p1p0[0] * p2p0[1] - p1p0[1] * p2p0[0]);
	coefficient[3] = 0.0;

	// 归一化
	coefficient.normalize();

	// 计算d
	coefficient[3] = -1.0 * (coefficient.template head<4>().dot(p0.matrix()));
	return true;
}

// 查找距平面在容差范围内的点的函数findUnclusteredPointsWithinDistance()
/*入参：coefficient, 平面系数  unclustered, 未分类点索引集  出参：inliers, 距平面在容差范围内的点索引集  distance, 平面到点的距离*/ 
void RANSACPlane::findUnclusteredPointsWithinDistance(const Vector4f& coefficient,
	const vector<int>& unclustered, vector<int>& inliers, float& distance)
{
	inliers.resize(0);
	distance = 0;
	for (int i = 0; i < unclustered.size(); i++)
	{
		PointXYZ p = (*cloud)[unclustered[i]];
		Vector4f pt(p.x, p.y, p.z, 1.0f);
		float d = abs(coefficient.dot(pt));

		if (d < param.distance_threshold)
		{
			inliers.push_back(unclustered[i]);
			distance += d;
		}
	}
}

int RANSACPlane::segment(vector<vector<int>>& clusters, vector<int>& unclustered)
{
	// 1. 分割前准备
	// 获取输入点云cloud中点的个数pts_num
	int pts_num = cloud->size();
	vector<int> inliers;

	// 将未分类点索引集unclustered大小初始化为pts_num，元素按递增顺序排列
	unclustered.resize(pts_num);
	iota(unclustered.begin(), unclustered.end(), 0);

	// 2. 当未分类点索引集unclustered中的元素个数多于3个时，执行以下操作
	while (unclustered.size() > 3)
	{
		// 利用RANSAC从未归类的点中寻找一个可拟合平面的点集
		inliers.resize(0);
		float error = FLT_MAX;
		for (int itr_num = 0; itr_num < param.max_iteration_num; itr_num++)
		{
			// ① 从未归类的点中随机选取3个不重复、不共线的点
			int random_indices[3];
			random_indices[0] = ((double)rand() / RAND_MAX) * unclustered.size();
			random_indices[1] = ((double)rand() / RAND_MAX) * unclustered.size();
			random_indices[2] = ((double)rand() / RAND_MAX) * unclustered.size();

			Vector4f coefficient;
			// 用computePlaneCoefficients()函数判断这3个点是否共面
			if (computePlaneCoefficients(random_indices, coefficient))
			{
				// ② 寻找距离当前平面在容差范围内的点
				vector<int> inliers_temp;
				inliers_temp.resize(0);
				float distance = 0;
				this->findUnclusteredPointsWithinDistance(coefficient, unclustered, inliers_temp, distance);

				// ③ 判断是否比已记录结果更优，是则更新
				if (inliers_temp.size() > inliers.size())
				{
					inliers = inliers_temp;
					error = distance / inliers.size();
				}
			}
			else continue;
		}

		// 若经过最大次数迭代后，得到的inliers中点的个数大于规定的最小数，则说明是有效分割
		if (inliers.size() > param.min_pts)
		{
			clusters.push_back(inliers);
			cout << "clustered + 1" << endl;
			// 将inliers中的点从unclustered中删除
			for (int i = 0; i < inliers.size(); i++)
			{
				vector<int>::iterator itr = find(unclustered.begin(), unclustered.end(), inliers[i]);
				if (itr != unclustered.end())
					unclustered.erase(itr);
			}
		}
	}
	return 1;
}