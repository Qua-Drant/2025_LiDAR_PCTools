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


// ����ƽ��ϵ������computePlaneCoefficients()
/*��Σ�pt_indices, cloud��3�����������  ���Σ�coefficient, ���ƽ���ϵ��  ���أ�3�㹲��ʱ����true, ���򷵻�false */
bool RANSACPlane::computePlaneCoefficients(int* pt_indices, Vector4f& coefficient)
{
	if (pt_indices[0] == pt_indices[1] || pt_indices[1] == pt_indices[2] || pt_indices[2] == pt_indices[0])
		return false;
	// ��ȡ3������
	Array4fMap p0 = (*cloud)[pt_indices[0]].getArray4fMap();
	Array4fMap p1 = (*cloud)[pt_indices[1]].getArray4fMap();
	Array4fMap p2 = (*cloud)[pt_indices[2]].getArray4fMap();

	// ���3���Ƿ���
	Array4f p1p0 = p1 - p0;
	Array4f p2p0 = p2 - p0;
	Array4f dy1dy2 = p1p0 / p2p0;
	if ((dy1dy2[0] == dy1dy2[1] && dy1dy2[2] == dy1dy2[1]))
		return false;

	// ����3��ƽ��ķ�����
	coefficient[0] = (p1p0[1] * p2p0[2] - p1p0[2] * p2p0[1]);
	coefficient[1] = (p1p0[2] * p2p0[0] - p1p0[0] * p2p0[2]);
	coefficient[2] = (p1p0[0] * p2p0[1] - p1p0[1] * p2p0[0]);
	coefficient[3] = 0.0;

	// ��һ��
	coefficient.normalize();

	// ����d
	coefficient[3] = -1.0 * (coefficient.template head<4>().dot(p0.matrix()));
	return true;
}

// ���Ҿ�ƽ�����ݲΧ�ڵĵ�ĺ���findUnclusteredPointsWithinDistance()
/*��Σ�coefficient, ƽ��ϵ��  unclustered, δ�����������  ���Σ�inliers, ��ƽ�����ݲΧ�ڵĵ�������  distance, ƽ�浽��ľ���*/ 
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
	// 1. �ָ�ǰ׼��
	// ��ȡ�������cloud�е�ĸ���pts_num
	int pts_num = cloud->size();
	vector<int> inliers;

	// ��δ�����������unclustered��С��ʼ��Ϊpts_num��Ԫ�ذ�����˳������
	unclustered.resize(pts_num);
	iota(unclustered.begin(), unclustered.end(), 0);

	// 2. ��δ�����������unclustered�е�Ԫ�ظ�������3��ʱ��ִ�����²���
	while (unclustered.size() > 3)
	{
		// ����RANSAC��δ����ĵ���Ѱ��һ�������ƽ��ĵ㼯
		inliers.resize(0);
		float error = FLT_MAX;
		for (int itr_num = 0; itr_num < param.max_iteration_num; itr_num++)
		{
			// �� ��δ����ĵ������ѡȡ3�����ظ��������ߵĵ�
			int random_indices[3];
			random_indices[0] = ((double)rand() / RAND_MAX) * unclustered.size();
			random_indices[1] = ((double)rand() / RAND_MAX) * unclustered.size();
			random_indices[2] = ((double)rand() / RAND_MAX) * unclustered.size();

			Vector4f coefficient;
			// ��computePlaneCoefficients()�����ж���3�����Ƿ���
			if (computePlaneCoefficients(random_indices, coefficient))
			{
				// �� Ѱ�Ҿ��뵱ǰƽ�����ݲΧ�ڵĵ�
				vector<int> inliers_temp;
				inliers_temp.resize(0);
				float distance = 0;
				this->findUnclusteredPointsWithinDistance(coefficient, unclustered, inliers_temp, distance);

				// �� �ж��Ƿ���Ѽ�¼������ţ��������
				if (inliers_temp.size() > inliers.size())
				{
					inliers = inliers_temp;
					error = distance / inliers.size();
				}
			}
			else continue;
		}

		// �����������������󣬵õ���inliers�е�ĸ������ڹ涨����С������˵������Ч�ָ�
		if (inliers.size() > param.min_pts)
		{
			clusters.push_back(inliers);
			cout << "clustered + 1" << endl;
			// ��inliers�еĵ��unclustered��ɾ��
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