// IOandVisualization.cpp - Sample code for PCL IO and visualization

#include <iostream>
#include <conio.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include <las/lasreader.hpp>
#include <las/laswriter.hpp>
#include <las/laszip_decompress_selective_v3.hpp>


using namespace std;
using namespace pcl;

// 将带有RBG的点云文件LAS另写为带有强度的PCD文件
int las2pcd_XYZI(const string& las_file_path, const string& pcd_file_path)
{
    // 读取LAS文件
    cout << "Loading LAS file..." << endl;
    LASreadOpener las_opener;
    las_opener.set_file_name(las_file_path.c_str());
    las_opener.set_decompress_selective(true);

    if (!las_opener.active())
    {
        cout << "Error: no input specified." << endl;
        return 0;
    }

    // 声明一个cloud,用来存放读取的点云x,y,z,intensity信息
    PointCloud<PointXYZI> ::Ptr cloud(new PointCloud<PointXYZI>);

    while (las_opener.active())
    {
        LASreader* las_reader = las_opener.open();
        if (las_reader == 0)
        {
            cout << "Error: Failed to load the LAS file." << endl;
            return 0;
        }

        LASheader* header = &(las_reader->header);

        size_t count = las_reader->npoints;
        double offset_x = las_reader->header.x_offset;
        double offset_y = las_reader->header.y_offset;
        double offset_z = las_reader->header.z_offset;

        if (!cloud)
            cloud = PointCloud<PointXYZI>::Ptr(new PointCloud<PointXYZI>);
        cloud->width = 0;
        cloud->height = 1;
        cloud->resize(0);
        cloud->is_dense = false;

        while (las_reader->read_point())
        {
            PointXYZI p;
            p.x = las_reader->point.get_x() - offset_x;
            p.y = las_reader->point.get_y() - offset_y;
            p.z = las_reader->point.get_z() - offset_z;
            p.intensity = las_reader->point.get_intensity();
            cloud->push_back(p);
        }

        las_reader->close();
        delete las_reader;
    }

    cout << "Loading LAS file finished!" << endl;

    // 写入PCD文件
    if (io::savePCDFileBinary<PointXYZI>(pcd_file_path, *cloud) == -1)
    {
        PCL_ERROR("Error saving PCD file.\n");
        return -1;
    }
    cout << "Saving PCD file finished!" << endl;

    return 1;
}

// 将带有RGB的点云文件PCD另写为txt文件
int pcd2txt_RGB(const string& pcd_file_path, const string& txt_file_path)
{
    PointCloud<PointXYZRGB> ::Ptr cloud(new PointCloud<PointXYZRGB>);

    if (io::loadPCDFile(pcd_file_path, *cloud) == -1)
    {
        PCL_ERROR("Error loading PCD file.\n");
        return -1;
    }
    cout << "Successed!" << endl;

    ofstream of(txt_file_path);
    if (!of.is_open())
    {
        cout << "Error opening txt file." << endl;
        return -1;
    }

    //遍历点云中的点
    for (int i = 0; i < cloud->size(); i++)
    {
        PointXYZRGB point = cloud->points[i];
        // 在txt文件中写入点的坐标和RGB值
        of << point.x << "\t" << point.y << "\t" << point.z << "\t" << (int)point.r << "\t" << (int)point.g << "\t" << (int)point.b << endl;
    }
    of.close();
    cout << "Saving successed!" << endl;

    return 1;

}

// 将带有强度的点云文件PCD另写为txt文件
int pcd2txt_intensity(const string& pcd_file_path, const string& txt_file_path)
{
    // 第一步：将PCD文件读入到一个PointClound对象中
    PointCloud<PointXYZI> ::Ptr cloud(new PointCloud<PointXYZI>);

    if (pcl::io::loadPCDFile(pcd_file_path, *cloud) == -1)
    {
        PCL_ERROR("Error loading PCD file.\n");
        return -1;
    }
    cout << "Successed!" << endl;

    // 第二步：声明一个写txt文件的对象
    ofstream of(txt_file_path);
    if (!of.is_open())
    {
        cout << "Error opening txt file." << endl;
        return -1;
    }

    // 遍历cloud中的每一个点，依次写出每个点的x,y,z、intensity信息
    for (int i = 0; i < cloud->size(); i++)
    {
        PointXYZI point = cloud->points[i];
        of << point.x << "\t" << point.y << "\t" << point.z << "\t" << point.intensity << endl;
    }
    of.close();
    cout << "Saving successed!" << endl;
     
    return 1;
}


// 主函数
int main()
{

    /*-----------------------进行文件预处理-----------------------*/
    // 介绍程序功能作用及处理流程
    cout << "This program can convert LAS file to PCD file(Intensity) and PCD file(RGB/Intensity) to TXT file." << endl;
    cout << "1. Input LAS file path and output PCD file path." << endl;
    cout << "2. Convert PCD file to TXT file." << endl;
    cout << "3. Choose convert type: RGB or Intensity." << endl;
    cout << "4. Visualization." << endl;
    cout << "Press any key to continue." << endl;
    int key = _getch(); // 等待用户输入任意字符，以便继续运行程序

    /*-----------------------LAS文件转PCD文件---------------------*/
    // 输入LAS文件路径
    bool flag = true;
    cout << "Input LAS filepath(Do not include space in the path): " << endl;
    string las_file_path;
    cin >> las_file_path;

    // 检查路径是否合法
    while (flag) {
        if (las_file_path.empty() || las_file_path.find(".las") == string::npos)
        {
            cout << "Invalid LAS file path. Please input again." << endl;
            // 重新输入LAS文件路径
            cin >> las_file_path;
        }
        else flag = false;
    }

    flag = true;
    cout << "Output PCD filepath(Do not include space in the path): " << endl;
    string output_file_path;
    cin >> output_file_path;

    // 检查路径是否合法
    while (flag) {
        if (output_file_path.empty() || output_file_path.find(".pcd") == string::npos)
        {
            cout << "Invalid Output PCD file path. Please input again." << endl;
            // 重新输入输出PCD文件路径
            cin >> output_file_path;
        }
        else flag = false;
    }
    clock_t time0 = clock(); //记录转换开始时间
    las2pcd_XYZI(las_file_path, output_file_path);
    clock_t time1 = clock(); //记录转换结束时间
    cout << "Take time " << (time1 - time0) / 1000.0 << "s" << endl;
    cout << "Finished!" << endl;

    /*-----------------------PCD文件转TXT文件------------------------*/

    // 读取点云文件路径
    string pcd_file_path = output_file_path;

    // 输入TXT文件路径
    flag = true;
    cout << "Input TXT filepath(Do not include space in the path): " << endl;
    string txt_file_path;
    cin >> txt_file_path;

    // 检查路径是否合法
    while (flag) {
        if (txt_file_path.empty() || txt_file_path.find(".txt") == string::npos)
        {
            cout << "Invalid txt file path. Please input again." << endl;
            // 重新输入输出TXT文件路径
            cin >> txt_file_path;
        }
        else flag = false;
    }

    // 选择转换类型(RGB或强度)
    cout << "Convert Type? RGB(1) or Intensity(2)" << endl;
    int type;
    cin >> type;

    // 转化类型为RGB
    if (type == 1)
    {
        clock_t time0 = clock(); //记录转换开始时间

        int res = pcd2txt_RGB(pcd_file_path, txt_file_path);

        clock_t time1 = clock(); //记录转换结束时间

        cout << "Take time " << (time1 - time0) / 1000.0 << "s" << endl;

        cout << "Visualization? y(yes), n(no)" << endl;
        char tag;
        cin >> tag;

        if (tag == 'y' || tag == 'Y')
        {
            // 重新读入点云
            PointCloud<PointXYZRGB> ::Ptr cloud(new PointCloud<PointXYZRGB>);
            if (pcl::io::loadPCDFile(pcd_file_path, *cloud) == -1)
            {
                PCL_ERROR("Error loading PCD file.\n");
                return -1;
            }

            // 声明一个可视化窗口对象，该窗口的名字为“Cloud Viewer1”
            pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Cloud Viewer1"));
            // 设置窗口的背景色，参数为归一化的RGB值
            viewer->setBackgroundColor(0.5, 1.0, 1.0);
            // 设置显示的点云对象，以及该点云对象的名字
            viewer->addPointCloud(cloud, "Color point cloud");
            // 设置渲染参数，如点的大小
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "Color point cloud");
            // 在窗口中添加坐标系信息
            viewer->addCoordinateSystem();
            viewer->spin();
        }
        cout << "Press e to exit." << endl;
        char e;  cin >> e;
        if (e == 'e') return 1;
    }

    // 转化类型为强度
    if (type == 2)
    {
        clock_t time0 = clock(); //记录转换开始时间

        int res = pcd2txt_intensity(pcd_file_path, txt_file_path);

        clock_t time1 = clock(); //记录转换结束时间

        cout << "Take time " << (time1 - time0) / 1000.0 << "s" << endl;

        cout << "Visualization? y(yes), n(no)" << endl;
        char tag;
        cin >> tag;

        if (tag == 'y' || tag == 'Y')
        {
            // 重新读入点云
            PointCloud<PointXYZI> ::Ptr cloud(new PointCloud<PointXYZI>);
            if (pcl::io::loadPCDFile(pcd_file_path, *cloud) == -1)
            {
                PCL_ERROR("Error loading PCD file.\n");
                return -1;
            }

            // 遍历所有点，记录最大和最小的强度值
            float max_intensity = -1e9;
            float min_intensity = 1e9;
            for (int i = 0; i < cloud->size(); i++)
            {
                PointXYZI point = cloud->points[i];
                if (point.intensity > max_intensity)
                {
                    max_intensity = point.intensity;
                }
                if (point.intensity < min_intensity)
                {
                    min_intensity = point.intensity;
                }
            }
            cout << "Max intensity: " << max_intensity << endl;
            cout << "Min intensity: " << min_intensity << endl;

            // 声明一个PointCloud<PointXYZRGB>对象，用于显示强度值
            PointCloud<PointXYZRGB> ::Ptr cloud_rgb(new PointCloud<PointXYZRGB>);
            // 复制 cloud 中的xyz值到 cloud_rgb 中，使用pcl的copy()函数
            copyPointCloud(*cloud, *cloud_rgb);
            // 遍历 cloud_rgb 中的点，根据强度值设置点的颜色
            for (int i = 0; i < cloud_rgb->size(); i++)
            {
                PointXYZRGB point = cloud_rgb->points[i];
                PointXYZI I = cloud->points[i];
                float intensity = I.intensity;
                float r = 255 * (intensity - min_intensity) / (max_intensity - min_intensity);
                float g = 255 * abs((2 * intensity - max_intensity - min_intensity) / (max_intensity - min_intensity));
                float b = 255 * (1 - (intensity - min_intensity) / (max_intensity - min_intensity));
                point.r = r;
                point.g = g;
                point.b = b;
                cloud_rgb->points[i] = point;
            }

            // 声明一个可视化窗口对象，该窗口的名字为“Cloud Viewer2”
            pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Cloud Viewer2"));
            // 设置窗口的背景色，参数为归一化的RGB值
            viewer->setBackgroundColor(0.5, 1.0, 1.0);
            // 设置显示的点云对象，以及该点云对象的名字
            viewer->addPointCloud(cloud_rgb, "Intensity point cloud");
            // 设置渲染参数，如点的大小
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "Intensity point cloud");
            // 在窗口中添加坐标系信息
            viewer->addCoordinateSystem();
            viewer->spin();
        }
        cout << "Press e to exit." << endl;
        char e;  cin >> e;
        if (e == 'e') return 1;
    }

    cout << "Invalid convert type." << endl;
    return -1;
}
