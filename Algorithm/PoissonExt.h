#pragma once
#include "../Header.h"
#include "Pyramid.h"

class CPoissonExt: public QThread
{
	Q_OBJECT
public:
	CPoissonExt(Pyramid &pyramid);
	void run();
	~CPoissonExt(void);
	int prepare(int side, cv::Mat &extends,cv::Mat &vector);
	void poissonExtend(cv::Mat &dst, int size);
	template<class T_in, class T_out>
	inline T_out BilineaGetColor_clamp(cv::Mat& img, float px,float py);//clamp for outside of the boundary
	void cudaSolver(float* A, int* rowindex, int* columns,int N,int nz,float*Bx, float*X);
public:
signals:
	void sigFinished();

public:
	int cols,rows,times,ex;
	int *type,*index;
	cv::Mat _image1,_image2;
	std::vector<cv::Mat> &_extends1,&_extends2;
	std::vector<cv::Mat> &_vector;
	float _runtime;	
	bool _gpu_flag;
};

