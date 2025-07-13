// PCRegistration.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。


#include <iostream>
#include <string>
#include <numeric>
#include <Eigen/Eigenvalues>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/pfh.h>
#include <pcl/registration/correspondence_estimation.h>


using namespace std;
using namespace pcl;
using namespace Eigen;


int solveTransform(const PointCloud<PointXYZ>::Ptr& cloud_target,	// 待配准点云
	const PointCloud<PointXYZ>::Ptr& cloud_source,	// 参考点云
	MatrixXd& R, VectorXd& T)
{
	MatrixXd matrix_target = MatrixXd(3, cloud_target->size()); // 将Pointcloud用矩阵表示，方便后续计算
	MatrixXd matrix_source = MatrixXd(3, cloud_source->size());
	for (int i = 0; i < cloud_target->size(); i++)
		matrix_target.col(i) = Vector3d(cloud_target->points[i].x, cloud_target->points[i].y, cloud_target->points[i].z);
	for (int i = 0; i < cloud_source->size(); i++)
		matrix_source.col(i) = Vector3d(cloud_source->points[i].x, cloud_source->points[i].y, cloud_source->points[i].z);

	int nsize = matrix_target.cols(); // 点云大小

	// 求点云中心并移到中心点
	VectorXd meanT = matrix_target.rowwise().mean();
	VectorXd meanS = matrix_source.rowwise().mean();
	MatrixXd ReT = matrix_target.colwise() - meanT; // 计算均值
	MatrixXd ReS = matrix_source.colwise() - meanS;

	// 求解旋转矩阵	
	MatrixXd H = ReT * ReS.transpose(); // 计算协方差矩阵
	JacobiSVD<MatrixXd> svd(H, ComputeFullU | ComputeFullV); // 奇异值分解
	MatrixXd U = svd.matrixU();
	MatrixXd V = svd.matrixV();

	float det = (U * V.transpose()).determinant(); // 计算行列式
	Vector3d diagVec(1.0, 1.0, det); // 构造对角矩阵
	R = V * diagVec.asDiagonal() * U.transpose(); // 计算旋转矩阵

	T = meanS - R * meanT; // 计算平移向量

	return 1;
}


void estimateKeypoints(const PointCloud<PointXYZ>::Ptr& cloud,
	const float& curvature_threshold,
	const float& keypoint_resolution,
	const PointCloud<Normal>::Ptr& normals,
	PointCloud<PointXYZ>::Ptr& keypoints)
{
	// 根据曲率大小选取关键点
	for (int i = 0; i < cloud->size(); i++)
	{
		if (normals->points[i].curvature > curvature_threshold)
			keypoints->push_back(cloud->points[i]);
	}

	// 对初始的关键点进行降采样，以避免关键点过于密集
	// 用VoxelGrid对其进行降采样
	VoxelGrid<PointXYZ> voxel;
	voxel.setInputCloud(keypoints);
	voxel.setLeafSize(keypoint_resolution, keypoint_resolution, keypoint_resolution);
	voxel.filter(*keypoints); // 进行降采样
}


void computePFHFeatures(const PointCloud<PointXYZ>::Ptr& cloud,
	const PointCloud<Normal>::Ptr& normals,
	const PointCloud<PointXYZ>::Ptr& keypoints,
	const float& k,
	PointCloud<PFHSignature125>::Ptr& features)
{
	// 调用PCL的接口，计算关键点处的点特征直方图PFH
	features->resize(0);
	// 遍历keypoints，为keypoints中的每个点计算其点特征直方图
	search::KdTree<PointXYZ>::Ptr tree(new search::KdTree<PointXYZ>());
	tree->setInputCloud(cloud);
	for (int i = 0; i < keypoints->size(); i++)
	{
		PointXYZ search_p = (*keypoints)[i];
		vector<int> nb_indices;
		vector<float> nb_distances;
		tree->nearestKSearch(search_p, k, nb_indices, nb_distances); // 计算k近邻

		PFHEstimation<PointXYZ, Normal, PFHSignature125> pfh_estimation;
		VectorXf f(125);
		pfh_estimation.computePointPFHSignature(
			*cloud,
			*normals,
			nb_indices,
			5,
			f);

		PFHSignature125 f125;
		for (int j = 0; j < 125; j++)
			f125.histogram[j] = f(j); // 将计算得到的PFH特征值存入f125中
		features->push_back(f125); // 将PFH特征值存入features中
	}
}


void transformPointCloud(const PointCloud<PointXYZ>::Ptr& cloud_in,
	PointCloud<PointXYZ>::Ptr& cloud_out,
	const MatrixXd& R, const VectorXd& T)
{
	cloud_out = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
	for (int i = 0; i < cloud_in->size(); i++)
	{
		MatrixXd matrix_p(3, 1);
		matrix_p.col(0) = Vector3d(cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z);
		VectorXd p_transformed = R * matrix_p + T; // 计算变换后的点
		cloud_out->push_back(PointXYZ(p_transformed(0), p_transformed(1), p_transformed(2))); // 将变换后的点存入cloud_out中
	}
}


int countIdenticalPoints(
	const PointCloud<PointXYZ>::Ptr& cloud,
	const search::KdTree<PointXYZ>::Ptr& tree,
	const float& dis_tolerance)
{
	int count = 0;
	vector<int> nn_indices;
	vector<float> nn_distances;
	float tolerance = dis_tolerance * dis_tolerance; // 计算容差平方
	for (auto& p : cloud->points)
	{
		tree->nearestKSearch(p, 1, nn_indices, nn_distances); // 计算k近邻
		if (nn_distances[0] < tolerance) // 如果距离小于容差，则认为是相同点
			count++;
	}
	return count;
}


int coarseRegistration(const PointCloud<PointXYZ>::Ptr& cloud,
	const PointCloud<PointXYZ>::Ptr& cloud_reference,
	const float& k,
	const float& curvature_threshold,
	const float& keypoint_resolution,
	const int& max_iteration_num,
	const float& dis_err_tolerance,
	MatrixXd& R, VectorXd& T)
{
	// Step 1: 预处理-计算法向量
	cout << "Step 1: 预处理-计算法向量" << endl;
	PointCloud<Normal>::Ptr normals(new PointCloud<Normal>());
	PointCloud<Normal>::Ptr normals_reference(new PointCloud<Normal>());
	search::KdTree<PointXYZ>::Ptr tree(new search::KdTree<PointXYZ>());
	NormalEstimation<PointXYZ, Normal> ne;

	/*计算待配准点云法向量*/
	tree->setInputCloud(cloud);
	ne.setInputCloud(cloud);
	ne.setSearchMethod(tree);
	ne.setKSearch(k);
	ne.compute(*normals);

	/*计算参考点云法向量*/
	tree->setInputCloud(cloud_reference);
	ne.setInputCloud(cloud_reference);
	ne.setSearchMethod(tree);
	ne.setKSearch(k);
	ne.compute(*normals_reference);

	// Step 2: 计算待配准点云和参考点云中的关键点
	cout << "Step 2: 计算待配准点云和参考点云中的关键点" << endl;
	PointCloud<PointXYZ>::Ptr keypoints(new PointCloud<PointXYZ>());
	PointCloud<PointXYZ>::Ptr keypoints_reference(new PointCloud<PointXYZ>());
	estimateKeypoints(cloud, curvature_threshold, keypoint_resolution, normals, keypoints);
	estimateKeypoints(cloud_reference, curvature_threshold, keypoint_resolution, normals_reference, keypoints_reference);
	if (keypoints->size() < 3 || keypoints_reference->size() < 3)
	{
		cout << "ERROR：关键点个数太少，无法进行配准！" << endl;
		return 0;
	}

	// Step 3: 计算待配准点云和参考点云中的关键点特征
	cout << "Step 3: 计算待配准点云和参考点云中的关键点特征" << endl;
	PointCloud<PFHSignature125>::Ptr features(new PointCloud<PFHSignature125>());
	PointCloud<PFHSignature125>::Ptr features_reference(new PointCloud<PFHSignature125>());
	computePFHFeatures(cloud, normals, keypoints, k, features);
	computePFHFeatures(cloud_reference, normals_reference, keypoints_reference, k, features_reference);

	// Step 4: 估算关键点点对之间的对应关系
	cout << "Step 4: 估算关键点点对之间的对应关系" << endl;
	Correspondences all_correspondence;
	registration::CorrespondenceEstimation<PFHSignature125, PFHSignature125> est;
	est.setInputSource(features);
	est.setInputTarget(features_reference);
	est.determineReciprocalCorrespondences(all_correspondence);
	
	cout << "共有" << all_correspondence.size() << "对匹配点。" << endl;
	if (all_correspondence.size() < 3)
		return 0;

	// Step 5: 利用RANSAC排除误匹配，并求解最优R和T
	cout << "Step 5: 利用RANSAC排除误匹配，并求解最优R和T" << endl;
	PointCloud<PointXYZ>::Ptr resample_cloud(new PointCloud<PointXYZ>());
	VoxelGrid<PointXYZ> voxel;
	voxel.setInputCloud(cloud);
	voxel.setLeafSize(keypoint_resolution, keypoint_resolution, keypoint_resolution);
	voxel.filter(*resample_cloud); // 对待配准点云进行降采样

	PointCloud<PointXYZ>::Ptr tmp(new PointCloud<PointXYZ>());
	PointCloud<PointXYZ>::Ptr tmp_reference(new PointCloud<PointXYZ>());
	PointCloud<PointXYZ>::Ptr tmp_aligned(new PointCloud<PointXYZ>());

	int correspondence_size = all_correspondence.size();
	int max_count = 0;
	for (int i = 0; i < max_iteration_num; i++)
	{
		cout << "-----第" << i + 1 << "次迭代-----" << endl;
		// 从估算的对应关系中随机选取3对点
		int i1 = rand() % correspondence_size;
		int i2 = rand() % correspondence_size;
		int i3 = rand() % correspondence_size;
		if (i1 == i2 || i1 == i3 || i2 == i3)
			continue;	

		// 利用随机选取的3对点计算R和T
		tmp->resize(0);
		tmp_reference->resize(0);

		tmp->push_back(keypoints->points[all_correspondence[i1].index_query]);
		tmp->push_back(keypoints->points[all_correspondence[i2].index_query]);
		tmp->push_back(keypoints->points[all_correspondence[i3].index_query]);
		tmp_reference->push_back(keypoints_reference->points[all_correspondence[i1].index_match]);
		tmp_reference->push_back(keypoints_reference->points[all_correspondence[i2].index_match]);
		tmp_reference->push_back(keypoints_reference->points[all_correspondence[i3].index_match]);

		MatrixXd tmp_R;
		VectorXd tmp_T;
		solveTransform(tmp, tmp_reference, tmp_R, tmp_T); // 计算R和T

		//根据R和T对source进行变换，并统计其中与target满足距离小于dis_err_tolerance的点的个数
		transformPointCloud(resample_cloud, tmp_aligned, tmp_R, tmp_T); // 对待配准点云进行变换
		int count = countIdenticalPoints(tmp_aligned, tree, dis_err_tolerance); // 统计与参考点云中距离小于dis_err_tolerance的点的个数
		if (count > max_count)
		{
			max_count = count;
			R = tmp_R;
			T = tmp_T;
		}
		cout << "  " << count << "点对满足距离容差条件" << endl;

	}
	return 1;
}


int fineRegistrationICP(const PointCloud<PointXYZ>::Ptr& cloud, 
	const PointCloud<PointXYZ>::Ptr& cloud_reference, 
	MatrixXd& R, VectorXd& T)
{
	// 设置最大迭代次数
	int MaxIteration = 1000;

	// 初始化迭代次数
	int IterationNum = 1;

	// 初始化旋转矩阵R为单位阵
	R = MatrixXd::Identity(3, 3);
	//cout << R << endl;
	// 初始化平移向量T为零向量
	T = VectorXd::Zero(3);
	//cout << T << endl;
	// 初始化一个临时待配准点云 cloud_tmp , 将cloud拷贝到cloud_tmp
    PointCloud<PointXYZ>::Ptr cloud_tmp(new PointCloud<PointXYZ>(*cloud));

	while (IterationNum < MaxIteration)
	{
		// 1. 寻找临时待配准点云 cloud_tmp 在目标点云 cloud_reference 中的距离最近点，建立一一对应的匹配点 pts 和 pts_reference
		search::KdTree<PointXYZ>::Ptr tree(new search::KdTree<PointXYZ>());
		tree->setInputCloud(cloud_reference);

		PointCloud<PointXYZ>::Ptr pts(new PointCloud<PointXYZ>());
		PointCloud<PointXYZ>::Ptr pts_reference(new PointCloud<PointXYZ>());
		vector<float> distances;

		for (int i = 0; i < cloud_tmp->size(); i++)
		{
			PointXYZ search_p = (*cloud_tmp)[i];
			vector<int> nn_indices;
			vector<float> nn_distances;
			tree->nearestKSearch(search_p, 1, nn_indices, nn_distances); // 计算k近邻
			PointXYZ search_ref = (*cloud_reference)[nn_indices[0]];

			pts->push_back(search_p); // 将匹配点存入pts中
			pts_reference->push_back(search_ref); // 将匹配点存入pts_reference中
			distances.push_back(sqrt(nn_distances[0])); // 将距离存入distances中
		}

		//// 剔除错误点对
		//// 剔除距离大于3倍距离中值的点对
		vector<float> v = distances;
		sort(v.begin(), v.end());
		int site = v.size() / 2;
		float median_distance = v[site];
		for (int j = 0; j < distances.size(); j++)
		{
			if (distances[j] > 3 * median_distance)
			{
				pts->points.erase(pts->points.begin() + j);

				pts_reference->points.erase(pts_reference->points.begin() + j);
				distances.erase(distances.begin() + j);
				pts->width--;
				pts_reference->width--;
				j--;
			}
		}

		// 2. 根据 pts 和 pts_reference 解算增量旋转矩阵 dR 和 dT（直接调用已有函数 solveTransform())）
		MatrixXd dR = MatrixXd::Identity(3, 3);
		VectorXd dT = VectorXd::Zero(3);
		solveTransform(pts, pts_reference, dR, dT);
		// cout << dR << dT << endl;
		// 3. 将 dR 和 dT 累加到 R 和 T 上，即 R = dR * R , T = dR * T + dT
		R = dR * R;
		T = dR * T + dT;

		// 4. 更新临时待配准点云 cloud_tmp = R * cloud + T （直接调用已有函数 transformPointCloud()）
		PointCloud<PointXYZ>::Ptr cloud_tmp_new(new PointCloud<PointXYZ>);
		transformPointCloud(cloud_tmp, cloud_tmp_new, R, T);

		// 5. 统计 cloud_tmp 中的点到 cloud_reference 中的点的距离的平均值，
		// 若小于阈值则说明收敛，结束循环(break), 否则继续循环(IterationNum++))
		distances.resize(0); // 清空距离容器
		for (int i = 0; i < cloud_tmp_new->size(); i++)
		{
			PointXYZ search_p = (*cloud_tmp_new)[i];
			vector<int> nn_indices;
			vector<float> nn_distances;
			tree->nearestKSearch(search_p, 1, nn_indices, nn_distances); // 计算k近邻
			PointXYZ search_ref = (*cloud_reference)[nn_indices[0]];
			distances.push_back(sqrt(nn_distances[0])); // 将距离存入distances中
		}
		float mean_distance = accumulate(distances.begin(), distances.end(), 0.0) / distances.size();

		if (mean_distance < 0.1) break;
		else IterationNum++;
	}

	return 1;
}


int main()
{
	// Step1:读取点云文件
	cout << "请输入待配准的点云PCD文件路径(路径中不要有空格):" << endl;
	string infpath1;
	cin >> infpath1;
	cout << "请输入作为参考的点云PCD文件路径(路径中不要有空格):" << endl;
	string infpath2;
	cin >> infpath2;
	PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>());
	if (io::loadPCDFile<PointXYZ>(infpath1, *cloud) == -1)
	{
		PCL_ERROR("无法读取文件 %s \n", infpath1.c_str());
		return (-1);
	}
	PointCloud<PointXYZ>::Ptr cloud_reference(new PointCloud<PointXYZ>());
	if (io::loadPCDFile<PointXYZ>(infpath2, *cloud_reference) == -1)
	{
		PCL_ERROR("无法读取文件 %s \n", infpath2.c_str());
		return (-1);
	}


	// 根据输入判断进行粗配准还是进行精配准
	char flag;
	cout << "进行点云粗配准，请输入：C， 点云精配准，请输入：F" << endl;
	cin >> flag;

	// 定义空间变化矩阵R和T, cloud_reference  = R * cloud + T
	MatrixXd R;
	VectorXd T;
	int res = 0;
	if (flag == 'C')
		res = coarseRegistration(cloud, // 待配准点云
			cloud_reference,            // 参考点云
			10,                         // k邻域
			0.2,						// 关键点的曲率阈值
			0.3,						// 降采样分辨率
			20,						    // RANSAC最大迭代次数
			0.1,						// 容差距离
			R, T                        // 旋转矩阵和位移向量	
		);
	else if (flag == 'F')
		res = fineRegistrationICP(cloud, cloud_reference, R, T);
	if (res == 0) return 0;

	cout << "rotation is: " << endl << R << endl;
	cout << "translation is: " << endl << T << endl;

	// 对点云cloud进行空间变换
	PointCloud<PointXYZ>::Ptr cloud_aligned(new PointCloud<PointXYZ>());
	transformPointCloud(cloud, cloud_aligned, R, T); // 对待配准点云进行变换
	// 将变换后的点云保存到文件
	cout << "请输入变换后的点云文件保存路径(路径中不要有空格):" << endl;
	string outfpath;
	cin >> outfpath;
	io::savePCDFile(outfpath, *cloud_aligned);

	// 结果可视化
	cout << "是否可视化点云？(是：y，否：n)" << endl;
	char tag;
	cin >> tag;
	if (tag == 'y')
	{
		boost::shared_ptr<visualization::PCLVisualizer> viewer(new visualization::PCLVisualizer("Cloud Viewer1"));
		viewer->setBackgroundColor(0.5, 1.0, 1.0);
		visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_color1(cloud_reference, 255, 0, 0);
		viewer->addPointCloud(cloud_reference, cloud_color1, "PointCloud1");
		visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_color2(cloud, 0, 255, 0);
		viewer->addPointCloud(cloud, cloud_color2, "PointCloud2");
		visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_color3(cloud_aligned, 0, 0, 255);
		viewer->addPointCloud(cloud_aligned, cloud_color3, "PointCloud3");
		viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2, "PointCloud1");
		viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2, "PointCloud2");
		viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2, "PointCloud3");

		viewer->addCoordinateSystem();
		viewer->spin();

		return 1;
	}
}
