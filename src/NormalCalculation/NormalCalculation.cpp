// NormalCalculation.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。

#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <NormalCal.h>

using namespace std;
using namespace pcl;

// 计算点云几何特征
void covarianceFeatures(const PointCloud<PointXYZ>::Ptr& cloud,  //入参，输入点云
	const int& k_neighbours,	//入参，KNN搜索的邻近点个数
	vector<float>& curvature,	//出参，曲率
	vector<float>& linearity,	//出参，线性度
	vector<float>& planarity,	//出参，平面度
	vector<float>& scattering)	//出参，散乱性
{
	// 将结果写入TXT文件
	cout << "请输入需要保存的TXT文件路径（路径中不要有空格）：" << endl;
	string txt_file_path;
	cin >> txt_file_path;
	ofstream of(txt_file_path);
	if (!of.is_open())
	{
		cout << "文件打开失败！" << endl;
	}
	cout << "正在计算并保存TXT文件..." << endl;

	// 建立kdtree空间索引
	KdTreeFLANN<PointXYZ>::Ptr kdtree(new KdTreeFLANN<PointXYZ>);
	kdtree->setInputCloud(cloud);

	// 循环遍历每一个点，根据它周围的k邻近点，用PCA计算协方差特征
	for (const auto& point : *cloud)
	{
		// 确定k邻域
		PointXYZ searchPoint = point;			//要查询的点
		vector<int> KNNIndices(k_neighbours);	//用来存放搜索到的K邻近点的索引值
		vector<float> KNNSquaredDistances(k_neighbours);		//用来存放K邻近点的对应到查询点的距离的平方

		// 对K邻域进行PCA计算，求法向量
		if (kdtree->nearestKSearch(searchPoint, k_neighbours, KNNIndices, KNNSquaredDistances) > 0)
		{
			PointCloud<PointXYZ>::Ptr nb_cloud(new PointCloud<PointXYZ>);
			copyPointCloud(*cloud, KNNIndices, *nb_cloud);

			// 对邻域内的点进行PCA分解
			Vector4f centroid;	// 质心
			Matrix3f covariance; // 协方差矩阵
			compute3DCentroid(*nb_cloud, centroid); // 计算质心
			computeCovarianceMatrix(*nb_cloud, centroid, covariance);// 计算协方差矩阵

			Matrix3f eigenVectors;
			Vector3f eigenValues;
			SelfAdjointEigenSolver<Matrix3f> eigen_solver(covariance, ComputeEigenvectors);

			eigenVectors = eigen_solver.eigenvectors();// 特征向量，用3×3矩阵表示，每一列对应一个特征向量
			eigenValues = eigen_solver.eigenvalues();// 特征值，按从小到大排列
			Vector3f normalizedEigenValues = eigenValues / eigenValues.sum(); // 归一化特征值

			double l1 = normalizedEigenValues(2);
			double l2 = normalizedEigenValues(1);
			double l3 = normalizedEigenValues(0);
			curvature.push_back(l3/(l1+l2+l3)); // 曲率
			linearity.push_back((l1 - l2) / l3); // 线性度
			planarity.push_back((l2 - l3) / l1); // 平面度
			scattering.push_back(l3 / l1); // 散乱性
		}
		else
		{
			curvature.push_back(0);
			linearity.push_back(0);
			planarity.push_back(0);
			scattering.push_back(0);
		}

		of << point.x << "\t" << point.y << "\t" << point.z << "\t" << curvature.back() << "\t" <<
			linearity.back() << "\t" << planarity.back() << "\t" << scattering.back() << endl;
	}
	of.close();
	cout << "TXT文件保存成功！" << endl;
}


int main()
{
	// 1.读取点云文件
	cout << "请输入需要读取的PCD点云文件路径（路径中不要有空格）：" << endl;
	string infpath;
	cin >> infpath;
	PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
	if (io::loadPCDFile(infpath, *cloud) == -1)
	{
		PCL_ERROR("读取点云失败！\n");
		return -1;
	}
	cout << "读取点云成功！" << endl;

	// 2.求解法向量
	cout << "请输入KNN搜索的K值：";
	int k;
	cin >> k;
	PointCloud<Normal>::Ptr normals(new PointCloud<Normal>);

	cout << "正在计算法向量..." << endl;
	clock_t time0 = clock();
	IsNormal(cloud, k, normals);	//调用函数，实现利用最小二乘计算法向量
	pcaNormal(cloud, k, normals);	//调用函数，利用PCA计算法向量
	clock_t time1 = clock();

	cout << "共耗时：" << (time1 - time0) / 1000.0 << "s." << endl;

	// 3.计算几何特征
	//vector<float> curvature, linearity, planarity, scattering;
	//cout << "正在计算几何特征..." << endl;
	//clock_t time0 = clock();
	//covarianceFeatures(cloud, k, curvature, linearity, planarity, scattering);
	//clock_t time1 = clock();

	//cout << "共耗时：" << (time1 - time0) / 1000.0 << "s." << endl;

	// 4.可视化
	cout << "是否可视化点云？ (y/n)" << endl;
	char tag;
	cin >> tag;
	if (tag == 'y')
	{
		boost::shared_ptr<visualization::PCLVisualizer> viewer(new visualization::PCLVisualizer("Cloud Viewer"));
		viewer->setBackgroundColor(0.5, 1.0, 1.0);
		visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_color(cloud, 0, 125, 255); // 设置点云颜色
		viewer->addPointCloud(cloud, cloud_color, "PointCloud");
		viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2, "PointCloud"); // 设置点云大小
		viewer->addPointCloudNormals<PointXYZ, Normal>(cloud, normals, 20, 0.1, "normals"); // 添加法向量（20表示每20个点显示一个法线，0.1为法线长度）
		viewer->addText("normals", 180, 180);
		viewer->addCoordinateSystem();
		viewer->spin();
	}

	return 1;
}