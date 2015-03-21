#include "MatchingThread.h"
#include <helper_cuda.h>       // helper for CUDA error checking

CMatchingThread::CMatchingThread(Parameters& parameters,Pyramid &pyramids):_pyramids(pyramids),_parameters(parameters),gpu_morph(parameters,pyramids,runflag)
{
	runflag=true;
	
	percentage=0.0f;	
	_timer=NULL;
 	_timer=new QTimer(this);
    connect(_timer,SIGNAL(timeout()), this, SLOT(update_result()) );
	_timer->start(1000);
}

CMatchingThread::~CMatchingThread()
{
	if(_timer)
		delete _timer;
}


void CMatchingThread::update_result()
{
	
	omp_set_num_threads(10);
	
	int el=gpu_morph._current_l;
	if(el<1) el=1;
	int factor=_pyramids[0].factor_d/_pyramids[el].factor_d;
	float ratio_x=(float)_pyramids[0].width/(float)_pyramids[el].width;
	float ratio_y=(float)_pyramids[0].height/(float)_pyramids[el].height;
	if (_pyramids[el].v.size()>0)
	{
		//spatialy
		for(int i=0;i<_pyramids[el].depth;i++)
		{
			Mat temp=Mat(_pyramids[el].height,_pyramids[el].width,CV_32FC2);
			checkCudaErrors(cudaMemcpy2D(temp.data,temp.step, 
				_pyramids[el].v+i*_pyramids[el].pagestride,_pyramids[el].rowstride*sizeof(float2),
				_pyramids[el].width*sizeof(float2),_pyramids[el].height, cudaMemcpyDeviceToHost));

			if(ratio_x!=1||ratio_y!=1)
			{
				#pragma omp parallel for		
				for (int y=0; y<_pyramids[el].height; y++)
					for (int x=0; x<_pyramids[el].width; x++)
					{	
						
						Vec2f val=temp.at<Vec2f>(y,x);
						temp.at<Vec2f>(y,x)=Vec2f(val[0]*ratio_x,val[1]*ratio_y);
					}
			}

			if(temp.cols!=_pyramids[0].width||temp.rows!=_pyramids[0].height)
				Resize(temp,_pyramids._vector[MIN(i*factor,_pyramids[0].depth-1)]);
			else
				_pyramids._vector[MIN(i*factor,_pyramids[0].depth-1)]=temp.clone();
			
		}

		//temp
		if(factor>1)
		{
		   #pragma omp parallel for
			for(int i=0;i<_pyramids[el].depth-1;i++)
			{
				for (int k=1;k<factor;k++)
				{
					if (i*factor+k>=_pyramids[0].depth-1)
						continue;
					int beg=i*factor;
					int end=MIN((i+1)*factor,_pyramids[0].depth-1);
					float fa=(float)k/(float)(end-beg);
					_pyramids._vector[i*factor+k]=
						_pyramids._vector[beg]*(1-fa)+_pyramids._vector[end]*fa;					
				}
			}			
		}
		omp_set_num_threads(12);
		percentage=gpu_morph._current_iter/gpu_morph._total_iter*100;
		emit sigUpdate();
	
	}
}

void CMatchingThread::Resize(Mat& src,Mat& dst)
{
	#pragma omp parallel for	
	for (int y=0; y<dst.rows; y++)
		for (int x=0; x<dst.cols; x++)
		{
		
			float fy=(y+0.5)/dst.rows*src.rows-0.5;
			float fx=(x+0.5)/dst.cols*src.cols-0.5;

			dst.at<Vec2f>(y,x)=BiLinear<Vec2f>(src,make_float2(fx,fy));
		}

}

template <class T>
inline T CMatchingThread::BiLinear(cv::Mat& img, float2 p)
{
	int cols=img.cols;
	int rows=img.rows;
	int x[2],y[2];
	T value[2][2];

	x[0]=floor(p.x);
	y[0]=floor(p.y);

	x[1]=ceil(p.x);
	y[1]=ceil(p.y);	

	float u=p.x-x[0];
	float v=p.y-y[0];	

	for(int i=0;i<2;i++)
		for(int j=0;j<2;j++)
		{
			int temp_x,temp_y;
			temp_x=x[i];
			temp_y=y[j];
			
			temp_x=MAX(0,temp_x);
			temp_x=MIN(cols-1,temp_x);
			temp_y=MAX(0,temp_y);
			temp_y=MIN(rows-1,temp_y);			

			value[i][j]=img.at<T>(temp_y,temp_x);
		}
		
		return
			value[0][0]*(1-u)*(1-v)+value[0][1]*(1-u)*v+value[1][0]*u*(1-v)+value[1][1]*u*v;			

}

void CMatchingThread::run()
{
	//time
	clock_t start, finish;
	start = clock();
	gpu_morph.calculate_halfway_parametrization();
	finish = clock();
	run_time = (float)(finish - start) / CLOCKS_PER_SEC;
	
	_timer->stop();
	update_result();
	emit sigFinished();
}