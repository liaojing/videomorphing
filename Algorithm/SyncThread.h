#pragma once
#include "..//Header.h"
#include "Pyramid.h"


class CSyncThread: public QThread
{
	Q_OBJECT

public:
signals:
	void sigUpdate();
	void sigFinished();

	public slots:
		void update_result();
public:
	void optimize_level(int el);
	void genMatrix(int *row_ptr, int *col_ind, float *val, int N, int  nz, int* pagestride,float *rhs_x,float *rhs_y,float *rhs_z,int el);
	void run();
	void upsample_level(int el,int pel);
	void load_identity(int el);
public:
	CSyncThread(Parameters& parameters,Pyramid &pyramids);
	~CSyncThread(void);

public:
	bool runflag;
	float percentage;
	float run_time;

private:
	Pyramid& _pyramids;
	Parameters& _parameters;	
	QTimer *_timer;	
	std::vector<float*> d_x,d_y,d_z;
	int _total_l,_current_l;
	float _total_iter,_current_iter,_max_iter;
};

