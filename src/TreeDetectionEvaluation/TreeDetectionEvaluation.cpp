// TreeDetectionEvaluation.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;
using namespace pcl;

#define DIS_THRESH 0.01

PointCloud<PointXYZL>::Ptr readLabeldPointCloudfromTXT(const string& fpath)
{
    cout << endl <<  "  --> 正在读取 【" << fpath << "】..." << endl;
    PointCloud<PointXYZL>::Ptr cloud(new PointCloud<PointXYZL>);
    ifstream infile(fpath);
    if (!infile.is_open())
    {
        cout << "      无法打开文件【" << fpath << "】" << endl;
        return NULL;
    }
    double x, y, z, label;

    while (infile >> x >> y >> z >> label)
    {
        PointXYZL p;
        p.x = x;
        p.y = y;
        p.z = z;
        p.label = int(label);
        cloud->points.push_back(p);
    }

    if (cloud->empty())
    {
        cout << "      点云读取结果为空！" << endl;
        return NULL;
    }

    return cloud;
}

int checkTXTFileName(const string& fpath)
{
    string extname = fpath.substr(fpath.find_last_of('.') + 1, 3);
    if (extname == "txt" || extname == "TXT")
        return 1;
    else
        return 0;
}

void PointbasedEvaluation(const PointCloud<PointXYZL>::Ptr& res_cloud, const PointCloud<PointXYZL>::Ptr& gt_cloud)
{
    vector <int> TP, FP, FN;
    KdTreeFLANN<PointXYZL>::Ptr kdtree(new KdTreeFLANN<PointXYZL>);
    kdtree->setInputCloud(res_cloud);

    cout << endl << "  -->正在进行基于点的精度计算..." << endl;
    int process = 0;
    int psize = gt_cloud->size();
    for (int i = 0; i < psize; i++)
    {
        PointXYZL gt_p = gt_cloud->points[i];
        vector<int> indices;
        vector<float> dises;
        kdtree->nearestKSearch(gt_p, 1,indices, dises);
        PointXYZL res_p = res_cloud->points[indices[0]];

        if (gt_p.label > 0) // gt里为true
        {
            if (dises[0] > DIS_THRESH) // 若距离大于DIS_THRESH表示在res里没有找到对应的点，则假设res里标记为false
                FN.push_back(i);
            else
            {
                if (res_p.label > 0)
                    TP.push_back(i);
                else
                    FN.push_back(i);
            }
        }
        else
        {
            if (dises[0] <= DIS_THRESH && res_p.label > 0)
                FP.push_back(i);
        }    

        int pro = (int((float)i / psize * 10));
        if ( pro > process)
        {
            process = pro;
            cout << "   " << process*10 << "%";
        }
    }
    cout << "   100%";

    float precision = (float)TP.size() / (float)(TP.size() + FP.size());
    float recall = (float)TP.size() / (float)(TP.size() + FN.size());
    float F_score = 2 * precision * recall / (precision + recall);
    cout.precision(4);
    cout << endl
        << "      ------------基于点的精度评价结果--------------------------" << endl
        << "         precision = " << precision * 100 << "%%" << endl
        << "         recall  = " << recall * 100 << "%%" << endl
        << "         F-score  = " << F_score * 100 << "%%" << endl
        << "      ---------------------------------------------------------" << endl;
    
    PointCloud<PointXYZ>::Ptr TP_cloud, FP_cloud, FN_cloud;
    TP_cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
    FP_cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
    FN_cloud = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>());
    copyPointCloud(*gt_cloud, TP, *TP_cloud);
    copyPointCloud(*gt_cloud, FP, *FP_cloud);
    copyPointCloud(*gt_cloud, FN, *FN_cloud);

	// 保存
	cout << endl << "  --> 请输入精度评定结果的保存路径（文件夹）:";
	string dir;
	cin >> dir;

    cout << endl << "  --> 正在保存精度评定结果..." << endl;
	string s = dir.substr(dir.length() - 1, 1);
	if (s != "\\" && s != "/")
		dir += "\\";
	string savepath;
	savepath = dir + "point_based_true_positive.pcd";
    io::savePCDFile(savepath, *TP_cloud);
    savepath = dir + "point_based_false_positive.pcd";
    io::savePCDFile(savepath, *FP_cloud);
    savepath = dir + "point_based_false_negative.pcd";
    io::savePCDFile(savepath, *FN_cloud);
    cout << endl << "  --> 完成！" << endl;


//     // 可视化
// 	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Point-based evaluation result:"));
// 	viewer->setBackgroundColor(1.0, 1.0, 1.0);
// 	pcl::visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_color1(TP_cloud, 255, 255, 0);//黄色表示TP
// 	viewer->addPointCloud(TP_cloud, cloud_color1, "PointCloud1");
// 	pcl::visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_color2(FP_cloud, 255, 0, 0);//红色表示FP
// 	viewer->addPointCloud(FP_cloud, cloud_color2, "PointCloud2");
// 	pcl::visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_color3(FN_cloud, 0, 0, 255);//蓝色表示FN
// 	viewer->addPointCloud(FN_cloud, cloud_color3, "PointCloud3");
// 	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "PointCloud1");//ÉèÖÃµã´óÐ¡
// 	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "PointCloud2");//ÉèÖÃµã´óÐ¡   
// 	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "PointCloud3");//ÉèÖÃµã´óÐ¡ 
// 
// 	viewer->addCoordinateSystem();
// 	viewer->spin();
}

struct Tree 
{
    Tree()
    {
        pts = NULL;
        min_p = PointXYZ(FLT_MAX, FLT_MAX, FLT_MAX);
        max_p = PointXYZ(FLT_MIN, FLT_MIN, FLT_MIN);
    };
    ~Tree()
    {};
    PointCloud<PointXYZL>::Ptr pts;
    PointXYZ min_p;
    PointXYZ max_p;
    PointXYZ centroid;
};

vector<Tree> getTrees(const PointCloud<PointXYZL>::Ptr& cloud)
{
    map<int, vector<int>> trees;
    for (int i = 0; i < cloud->size(); i++)
    {
        int label = cloud->points[i].label;
        if (label == 0)
           continue;

        trees[label].push_back(i);
    }

    vector<Tree> Ts;
    for (map<int, vector<int>>::iterator itr = trees.begin(); itr != trees.end(); itr++)
    {
        PointCloud<PointXYZL>::Ptr pts(new  PointCloud<PointXYZL>());
        PointCloud<PointXYZ>::Ptr pts_xyz(new  PointCloud<PointXYZ>());
        copyPointCloud(*cloud, itr->second, *pts);
        copyPointCloud(*cloud, itr->second, *pts_xyz);
        if (pts->size() < 20)
            continue;
        PointXYZ minp, maxp;
        getMinMax3D(*pts_xyz, minp, maxp);
        Eigen::Vector4f centroid;
        compute3DCentroid(*pts_xyz, centroid);
        
        Tree a_tree;
        a_tree.pts = pts;
        a_tree.min_p = minp;
        a_tree.max_p = maxp;
        a_tree.centroid = PointXYZ(centroid[0], centroid[1], centroid[2]);

        Ts.push_back(a_tree);
    }
    return Ts;
}

int areSameTrees(const Tree& ref_tree, const Tree& test_tree)
{
    if (ref_tree.min_p.x >= test_tree.max_p.x || ref_tree.max_p.x <= test_tree.min_p.x ||
        ref_tree.min_p.y >= test_tree.max_p.y || ref_tree.max_p.y <= test_tree.min_p.y ||
        ref_tree.min_p.z >= test_tree.max_p.z || ref_tree.max_p.z <= test_tree.min_p.z)
        return 0; // 返回0说明两颗树完全不相交

    KdTreeFLANN<PointXYZL>::Ptr kdtree(new KdTreeFLANN<PointXYZL>);
    kdtree->setInputCloud(ref_tree.pts);
    
    int num = 0; 

    vector<int> nb_indices;
    vector<float> nb_distances;
    for (int i = 0; i < test_tree.pts->size(); i++)
    {
        PointXYZL search_p = test_tree.pts->points[i];
        kdtree->nearestKSearch(search_p, 1, nb_indices, nb_distances);
        if (nb_distances[0] < DIS_THRESH)
            num++;
    }

    float overlap1 = (float)(num) / ref_tree.pts->size();
    float overlap2 = (float)(num) / test_tree.pts->size();

    if (overlap1 >= 0.5 && overlap2 >= 0.5)
        return 1;  // 返回1说明认为两颗树是同一颗树

    else
        return 0;

    /*
    if (overlap1 >= 0.5 && overlap2 < 0.5)
        return -1; // 返回-1， 说明结果树即test_tree为false positive

    if (overlap1 < 0.5 && overlap2 >= 0.5)
        return -2; // 返回-2， 说明gt树为false negative

    if (overlap1 < 0.5 && overlap2 < 0.5)
        return 0;// 返回0说明两颗树完全不相交*/


}

void ObjectbasedEvaluation(const PointCloud<PointXYZL>::Ptr& res_cloud, const PointCloud<PointXYZL>::Ptr& gt_cloud)
{
    vector<Tree> res_trees = getTrees(res_cloud);
    vector<Tree> gt_trees = getTrees(gt_cloud);

    PointCloud<PointXYZ>::Ptr res_tree_centroids(new PointCloud<PointXYZ>());
    PointCloud<PointXYZ>::Ptr gt_tree_centroids(new PointCloud<PointXYZ>());

    for (int i = 0; i < res_trees.size(); i++)
        res_tree_centroids->push_back(res_trees[i].centroid);
    for (int i = 0; i < gt_trees.size(); i++)
        gt_tree_centroids->push_back(gt_trees[i].centroid);


    int TP = 0, FN = 0, FP = 0;
	PointCloud<PointXYZL>::Ptr TP_cloud, FP_cloud, FN_cloud;
	TP_cloud = PointCloud<PointXYZL>::Ptr(new PointCloud<PointXYZL>());
	FP_cloud = PointCloud<PointXYZL>::Ptr(new PointCloud<PointXYZL>());
	FN_cloud = PointCloud<PointXYZL>::Ptr(new PointCloud<PointXYZL>());

    KdTreeFLANN<PointXYZ>::Ptr kdtree(new KdTreeFLANN<PointXYZ>);
    kdtree->setInputCloud(res_tree_centroids);

    cout << endl << "  -->正在进行基于对象的精度计算..." << endl;
	int process = 0;
	int tsize = gt_trees.size();
    vector<bool> res_flags(res_trees.size(), false);
    for (int i = 0; i < gt_trees.size(); i++)
    {
        int same_trees = 0;

        vector<int> indices;
        vector<float> distances;

        kdtree->nearestKSearch(gt_tree_centroids->points[i],2 /*res_tree_centroids->size()*/, indices, distances);

        for(int j = 0; j < indices.size(); j++)
        {
            if (!res_flags[indices[j]])
            {
                if (areSameTrees(gt_trees[i], res_trees[indices[0]]) == 1)
                {
                    same_trees = 1;
                }
                res_flags[indices[0]] = true;
                break;
            }
 
        }


        /*
        for (int j = 0; j < res_trees.size(); j++)
        {
           if (res_flags[j])
               continue;
  
            if (areSameTrees(gt_trees[i], kdtree, res_trees[j]) == 1)
            {
                same_trees = 1;
                res_flags[j] = true;
                //break;
            }
        }*/

        if (same_trees == 1)
        {
            TP++;
            TP_cloud->insert(TP_cloud->end(), gt_trees[i].pts->begin(), gt_trees[i].pts->end());
        }
        
        // 若遍历完之后仍没有找到相同的树，说明gt_tree为false negative
        if (same_trees == 0)
        {
            FN++;
            FN_cloud->insert(FN_cloud->end(), gt_trees[i].pts->begin(), gt_trees[i].pts->end());
        }

		int pro = (int((float)i / tsize * 10));
		if (pro > process)
		{
			process = pro;
			cout << "  " << process * 10 << "%";
		}
    }

    // res_flags中没有被标记的，则为 false positive
    for (int i = 0; i < res_flags.size(); i++)
    {
        if (res_flags[i] == true)
            continue;

        FP++;
        FP_cloud->insert(FP_cloud->end(), res_trees[i].pts->begin(), res_trees[i].pts->end());
    }
    cout << "  100%";
   
    float precision = (float)TP / (float)(TP + FP);
    float recall = (float)TP / (float)(TP + FN);
    float F_score = 2 * precision * recall / (precision + recall);
    cout.precision(4);
	cout << endl
		<< "      ------------基于对象的精度评价结果------------------------" << endl
        << "         No. of TP = " << TP << endl
        << "         No. of FP = " << FP << endl
        << "         No. of FN = " << FN << endl
		<< "         precision = " << precision * 100 << "%" << endl
		<< "         recall  = " << recall * 100 << "%" << endl
		<< "         F-score  = " << F_score * 100 << "%" << endl
		<< "      ---------------------------------------------------------" << endl;
    

	// 保存
	cout << endl << "  --> 请输入精度评定结果的保存路径（文件夹）:";
	string dir;
	cin >> dir;

    cout << endl << "  --> 正在保存精度评定结果..." << endl;
	string s = dir.substr(dir.length() - 1, 1);
	if (s != "\\" && s != "/")
		dir += "\\";
	string savepath;
	savepath = dir + "object_based_true_positive.pcd";
    if (TP_cloud->empty())
        cout << " TP为0!" << endl;
    else
	    io::savePCDFile(savepath, *TP_cloud);
	savepath = dir + "object_based_false_positive.pcd";
    if (FP_cloud->empty())
        cout << "FP为0！" << endl;
    else
	    io::savePCDFile(savepath, *FP_cloud);
	savepath = dir + "object_based_false_negative.pcd";
    if (FN_cloud->empty())
        cout << " FN为0！" << endl;
    else
	    io::savePCDFile(savepath, *FN_cloud);
    cout << endl << "  --> 完成！" << endl;

//     // 可视化
//     boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Object-based evaluation result:"));
//     viewer->setBackgroundColor(1.0, 1.0, 1.0);
//     pcl::visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_color1(TP_cloud, 255, 255, 0);//黄色表示TP
//     viewer->addPointCloud(TP_cloud, cloud_color1, "PointCloud1");
//     pcl::visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_color2(FP_cloud, 255, 0, 0);//红色表示FP
//     viewer->addPointCloud(FP_cloud, cloud_color2, "PointCloud2");
//     pcl::visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_color3(FN_cloud, 0, 0, 255);//蓝色表示FN
//     viewer->addPointCloud(FN_cloud, cloud_color3, "PointCloud3");
//     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "PointCloud1");//ÉèÖÃµã´óÐ¡
//     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "PointCloud2");//ÉèÖÃµã´óÐ¡   
//     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "PointCloud3");//ÉèÖÃµã´óÐ¡ 
// 
//     viewer->addCoordinateSystem();
//     viewer->spin();
}

int main()
{
    cout << " -------------请按照指示操作--------------" << endl;
    cout << "  请输入树木提取结果文件（txt)：";
    string res_file;
    cin >> res_file;
    if (checkTXTFileName(res_file) != 1)
    {
        cout << "  提示：输入的不是.txt文件！" << endl;
        return 0;
    }

    cout << endl <<  "  请输入树木提取真值文件（txt)：";
    string gt_file;
    cin >> gt_file;
    if (checkTXTFileName(gt_file) != 1)
    {
        cout << "  提示：输入的不是.txt文件！" << endl;
        return 0;
    }

    PointCloud<PointXYZL>::Ptr res_cloud = readLabeldPointCloudfromTXT(res_file);
    if (res_cloud == NULL)
        return 0;

    PointCloud<PointXYZL>::Ptr gt_cloud = readLabeldPointCloudfromTXT(gt_file);
    if (gt_cloud == NULL)
        return 0;

//      if (res_cloud->size() != gt_cloud->size())
//      {
//          cout << "请确保结果与真值中的点数及点坐标一致！" << endl;
//          return 0;
//      }
   
    char option;
    cout << endl << "  是否进行基于点的精度评定？是（y），否（n）" << endl << "   ";
    cin >> option;
    if (option == 'y')
    {
        PointbasedEvaluation(res_cloud, gt_cloud);
    }

    cout << endl << "  是否进行基于对象的精度评定？是（y），否（n）" << endl << "   ";
    cin >> option;
    if (option == 'y')
    {
        ObjectbasedEvaluation(res_cloud, gt_cloud);
    }

    return 1;
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
