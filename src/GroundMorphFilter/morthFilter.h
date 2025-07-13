#pragma once
#include <vector>
#include <string>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

using namespace std;
using namespace pcl;

struct Params {
	float resolution;	  //դ�񻯵ķֱ��ʣ���դ����ÿ������Ĵ�С
	int windowMax;	      //��̬ѧ�������󴰿ڳߴ�(���������Ϊ��λ)��
	float slopeTol;       //�¶ȵ��ݲΧ��������ֵ��ʾ slopeTol = tan(slope)
	float elevationTh;    //�߶���ֵ
	float scalingFactor;  //���¶���صĳ߶����ӣ�eth = elevationTh + slope * scalingFactor
};


class morphFilter
{
private:
	PointCloud<PointXYZ>::Ptr cloud; //�������
	Params params; //����
	PointXYZ min, max;
	float* grid_data;
	int rows, cols;

public:
	// ���캯��
	morphFilter(PointCloud<PointXYZ>::Ptr& pc, Params par);
	// ��������
	~morphFilter();

	// ִ����̬ѧ�˲���API
	int doFiltering(vector<int>& ground_indices, vector<int>& nonground_indices);

private:
	// �Ե��ƽ���դ��
	int Rasterization();
	// �դ���еĿ׶�
	float fillHole(const int& row, const int& col, float* grid);
	// ������
	int open(int win_size, float* grid);
	// ��ʴ
	void erosion(int win_size, float* grid);
	// ����
	void dilation(int win_size, float* grid);
	// ���ݸ̸߳�������¼�ĸ߳�ֵ���¶���Ϣ����������ͷǵ����
	int seperatePoints(vector<int>& ground_indices, vector<int>& nonground_indices);

};

