#include "RegionGrowing.h"
#include <iostream>
#include <string>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>


using namespace std;
using namespace pcl;


RegionGrowing::RegionGrowing(
	const PointCloud<PointXYZ>::Ptr& pc,
	const Params& pa)
{
	para = pa;
	cloud.reset(new PointCloud<PointXYZ>);
	copyPointCloud(*pc, *cloud);
	normals = NULL;
	tree = NULL;
}

int RegionGrowing::doRegionGrowing(vector<vector<int>>& clusters, vector<int>& unclustered)
{
	// Step 1: ����kdtree����ɷ����������ʵļ���
	cout << "Creating KD tree..." << endl;
	if (tree == NULL)
		tree.reset(new search::KdTree<PointXYZ>);
	tree->setInputCloud(cloud);

	cout << "Estimating normals and curvatures..." << endl;
	if (normals == NULL)
		normals.reset(new PointCloud<Normal>);
	NormalEstimation<PointXYZ, Normal> normal_estimator;
	normal_estimator.setInputCloud(cloud);
	normal_estimator.setSearchMethod(tree);
	normal_estimator.setKSearch(para.k);
	normal_estimator.compute(*normals);

	// Step 2: ����ǰ׼��
	cout << "Determing the neighbourhood of each point..." << endl;
	int pts_num = cloud->points.size();
	vector<vector<int>> point_neighbourhood;
	for (int i = 0; i < pts_num; i++)
	{
		vector<int> nb_indices;
		vector<float> nb_distances;
		tree->nearestKSearch(cloud->points[i], para.neighbour_num, nb_indices, nb_distances);
		point_neighbourhood.push_back(nb_indices);
	}

	vector<bool> point_flags;	// ��ǵ��Ƿ񱻾���
	point_flags.resize(pts_num, false);

	// ���������Ű����ʴ�С��������
	vector<int> sorted_curvature_indices = sortCurvatures();

	// Step 3: ������������
	cout << "Region growing..." << endl;
	clusters.resize(0);
	unclustered.resize(0);
	float cos_thera_threshold = cos(para.theta_threshold);
	do
	{
		clock_t time0 = clock();
		// һ��ѭ��Ϊһ����������
		vector<int> inliers;
		vector<int> seeds;
		vector<bool> point_visited_flags;
		point_visited_flags.resize(pts_num, 0);
		seeds.push_back(sorted_curvature_indices[0]);
		while (!seeds.empty())
		{
			int seed_idx = seeds[0];
			seeds.erase(seeds.begin());
			inliers.push_back(seed_idx);
			point_visited_flags[seed_idx] = 1;

			Eigen::Map<Eigen::Vector3f> seed_p_normal(static_cast<float*>((*normals)[seed_idx].normal));
			vector<int> seed_neighbours = point_neighbourhood[seed_idx];
			for (int i = 0; i < seed_neighbours.size(); i++)
			{
				int nb_p_idx = seed_neighbours[i];
				if (point_flags[nb_p_idx] == true || point_visited_flags[nb_p_idx] == true)
					continue;

				point_visited_flags[nb_p_idx] = 1;
				Eigen::Map<Eigen::Vector3f> nb_p_normal(static_cast<float*>((*normals)[nb_p_idx].normal));
				float dot_product = abs(nb_p_normal.dot(seed_p_normal));
				if (dot_product > cos_thera_threshold)
				{
					if ((*normals)[nb_p_idx].curvature < para.curvature_threshold)
						// ��������ֵ���������������Ϊ�µ����ӵ���뵽seeds��
						// �õ���ڴ�seeds��ȡ��ʱ���ൽ��ǰcluster��
						seeds.push_back(nb_p_idx);
					else inliers.push_back(nb_p_idx);	// ���������뵽��ǰcluster��
				}
			}
		}

		if (inliers.size() > para.min_segment_size)
		{
			clusters.push_back(inliers);
			clock_t time1 = clock();
			cout << " --> New Cluster: " << clusters.size() << " is generated, Taking time: " << (time1 - time0) / 1000.0 << "s. And "
				<< sorted_curvature_indices.size() << " points unclustered." << endl;
		}
		else unclustered.insert(unclustered.end(), inliers.begin(), inliers.end());

		// ���������ɵ�cluster�еĵ��Ӧ�ı�ǽ��и��£�������Щ�����������ɾ��
		for (int i = 0; i < inliers.size(); i++)
		{
			point_flags[inliers[i]] = true;
			vector<int>::iterator itr = find(sorted_curvature_indices.begin(), sorted_curvature_indices.end(), inliers[i]);
			if (itr != sorted_curvature_indices.end())
				sorted_curvature_indices.erase(itr);
		}
	} while (sorted_curvature_indices.size() > 0);

	cout << "Totally" << unclustered.size() << " points are unclustered." << endl;
	point_flags.resize(0);
	point_neighbourhood.resize(0);

	return 1;
}

vector<int> RegionGrowing::sortCurvatures()
{
	vector<float> v;
	for (int i = 0; i < normals->points.size(); i++)
		v.push_back((*normals)[i].curvature);
	vector<int> idx(v.size());
	iota(idx.begin(), idx.end(), 0);
	sort(idx.begin(), idx.end(),
		[&v](int i1, int i2) {return v[i1] < v[i2]; });
	return idx;
}

RegionGrowing::~RegionGrowing()
{
	if (cloud)
	{
		cloud->resize(0);
	}
	if (normals)
	{
		normals->resize(0);
	}
}