#pragma once

#include "..//Header.h"
#include "Pyramid.h"
#include "morph.h"

class CMatchingThread : public QThread
{
	Q_OBJECT

public:
signals:
	void sigUpdate();
	void sigFinished();

	public slots:
	void update_result();

public:
	CMatchingThread(Parameters& parameters,Pyramid &pyramids);
	~CMatchingThread();
	void run();
	template <class T>
	T BiLinear(cv::Mat& img, float2 p);
	void Resize(Mat& src,Mat& dst);
public:	
	float percentage;
	float run_time;
	QTimer *_timer;
	bool runflag;

private:
	Pyramid& _pyramids;
	Parameters& _parameters;
	Morph gpu_morph;

};


