#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace pcl;
using namespace std;
using namespace Eigen;

// 基于最小二乘的法向量计算
void IsNormal(const PointCloud<PointXYZ>::Ptr& cloud, //输入点云
	const int& k_neighbors, //KNN搜索的邻近点个数
	PointCloud<Normal>::Ptr& normals) //输出的点云法向量
{
	// 1. 建立kdtree空间索引
	KdTreeFLANN<PointXYZ>::Ptr kdtree(new KdTreeFLANN<PointXYZ>);
	kdtree->setInputCloud(cloud);

	// 2. 将normals实例化，元素个数为0
	if (normals == NULL)normals = PointCloud<Normal>::Ptr(new PointCloud<Normal>);
	if (!normals -> empty()) normals -> resize(0);

	// 3. 循环遍历每一个点，根据它周围的k邻近点，用最小二乘计算法向量
	for (const auto& point : *cloud)
	{
		// 3.1 确定k邻域
		PointXYZ searchPoint = point;			//要查询的点
		vector<int> KNNIndices(k_neighbors);	//用来存放搜索到的K邻近点的索引值
		vector<float> KNNSquaredDistances(k_neighbors);		//用来存放K邻近点的对应到查询点的距离的平方

		// 3.2 对K邻域进行最小二乘计算，求法向量
		// 搜索返回值大于0时，表示搜索成功
		if (kdtree->nearestKSearch(searchPoint, k_neighbors, KNNIndices, KNNSquaredDistances) > 0)
		{
			MatrixXf A = MatrixXf::Random(k_neighbors, 3);
			MatrixXf b = MatrixXf::Random(k_neighbors, 1);
			MatrixXf X = MatrixXf::Random(3, 1);
			// A,b,X分别对应于公式 A * X = b中的矩阵
			// 下面对矩阵中的元素进行赋值
			for (int i = 0; i < k_neighbors; i++)
			{
				A(i, 0) = cloud->points[KNNIndices[i]].x;
				A(i, 1) = cloud->points[KNNIndices[i]].y;
				A(i, 2) = cloud->points[KNNIndices[i]].z;
				b(i, 0) = -1;
			}
			// 按公式 X = (A^T * A) ^ -1 * A^T * b求解X
			X = (A.transpose() * A).inverse() * A.transpose() * b;

			// 归一化
			double norm_xyz = sqrt(X(0) * X(0) + X(1) * X(1) + X(2) * X(2));
			double nx = X(0) / norm_xyz;
			double ny = X(1) / norm_xyz;
			double nz = X(2) / norm_xyz;
			normals->push_back(Normal(nx, ny, nz));
		}
		else normals->push_back(Normal());
	}
}

// 基于PCA的法向量计算
void pcaNormal(const PointCloud<PointXYZ>::Ptr& cloud, //输入点云
	const int& k_neighbors, //KNN搜索的邻近点个数
	PointCloud<Normal>::Ptr& normals)
{
	// 1. 建立kdtree空间索引
	KdTreeFLANN<PointXYZ>::Ptr kdtree(new KdTreeFLANN<PointXYZ>);
	kdtree->setInputCloud(cloud);

	// 2. 将normals实例化，元素个数为0
	if (normals == NULL)normals = PointCloud<Normal>::Ptr(new PointCloud<Normal>);
	if (!normals->empty()) normals->resize(0);

	// 3. 循环遍历每一个点，根据它周围的k邻近点，用PCA计算法向量
	for (const auto& point : *cloud)
	{
		// 3.1 确定k邻域
		PointXYZ searchPoint = point;			//要查询的点
		vector<int> KNNIndices(k_neighbors);	//用来存放搜索到的K邻近点的索引值
		vector<float> KNNSquaredDistances(k_neighbors);		//用来存放K邻近点的对应到查询点的距离的平方

		// 3.2 对K邻域进行PCA计算，求法向量
		if (kdtree->nearestKSearch(searchPoint, k_neighbors, KNNIndices, KNNSquaredDistances) > 0)
		{
			PointCloud<PointXYZ>::Ptr nb_cloud(new PointCloud<PointXYZ>);
			copyPointCloud(*cloud, KNNIndices, *nb_cloud);

			// 对邻域内的点进行PCA分解，计算法向量
			Vector4f centroid;	// 质心
			Matrix3f covariance; // 协方差矩阵
			compute3DCentroid(*nb_cloud, centroid); // 计算质心
			computeCovarianceMatrix(*nb_cloud, centroid, covariance);// 计算协方差矩阵
			
			Matrix3f eigenVectors;
			Vector3f eigenValues;
			SelfAdjointEigenSolver<Matrix3f> eigen_solver(covariance, ComputeEigenvectors);

			eigenVectors = eigen_solver.eigenvectors();// 特征向量，用3×3矩阵表示，每一列对应一个特征向量
			eigenValues = eigen_solver.eigenvalues();// 特征值，按从大到小排列

			//cout << "eigen vectors are: " << endl << eigenVectors << endl;
			//cout << "eigen values are: " << endl << eigenValues << endl;

			// 选择最小特征值对应的特征向量即为当前点的法向量
			double nx = eigenVectors(0, 0);
			double ny = eigenVectors(0, 1);
			double nz = eigenVectors(0, 2);
			normals->push_back(Normal(nx,ny,nz));
		}
		else normals->push_back(Normal());
	}
}