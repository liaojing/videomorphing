#pragma once
#include "../Header.h"
#include "Pyramid.h"

class CQuadraticPath: public QThread
{
	Q_OBJECT

public:
	CQuadraticPath(Pyramid &pyramid);
	~CQuadraticPath(void);
	void cudaSolver(float* A, int* rowindex, int* columns,int N,int nz,float*Bx, float*X);
	void run();
	void optimize();

public:
signals:
	void sigFinished();

public:
	float _runtime;
	std::vector<cv::Mat>& _qpath;
	std::vector<cv::Mat>& _vector;
	int times,rows,cols;
	
};
