#include <util/dimage.h>
#include <util/image_ops.h>
#include "imgio.h"
#include "pyramid.h"
#include <Resample/scale.h>

#define Max_stage1 4000000
#define Max_stage2 14000000
#define USE_IMAGEMAGICK 1
Pyramid::Pyramid()
{
	_video0=_video1=_forw0=_forw1=NULL;
}
Pyramid::~Pyramid()
{
	clear();
}

void Pyramid::clear()
{
	for(size_t i=0; i<m_data.size(); ++i)
		delete m_data[i];
	m_data.clear();
	_vector.clear();
	_qpath.clear();
	_extends1.clear();
	_extends2.clear();
	if(_video0)
	{
		cudaFreeArray(_video0);
		_video0=NULL;
	}
	if(_video1)
	{
		cudaFreeArray(_video1);
		_video1=NULL;
	}
	if(_forw0)
	{
		cudaFreeArray(_forw0);
		_forw0=NULL;
	}
	if(_forw1)
	{
		cudaFreeArray(_forw1);
		_forw1=NULL;
	}

}

template <class T> 
T log2(const T &v)/*{{{*/
{
	using std::log;
	return log(v)/log(T(2));
}

void Pyramid::build(std::vector<cv::Mat> &video0,std::vector<cv::Mat> &video1,std::vector<cv::Mat> &f0, std::vector<cv::Mat> &f1,int start_res)
{
	clear();
	
	int w,h,d;
	w=video0[0].cols;
	h=video0[0].rows;
	d=video0.size();
		
	for (int i=0;i<d;i++)
	{
		cv::Mat vector=cv::Mat::zeros(h,w,CV_32FC4);
		_vector.push_back(vector);
		
		cv::Mat image1=cv::Mat(h,w,CV_8UC4);
		cv::Mat image2=cv::Mat(h,w,CV_8UC4);
		int from_to[] = { 0,0,1,1,2,2,3,3 };
		cv::Mat mask=cv::Mat::zeros(h,w,CV_8UC1);
		cv::Mat src1[2]={video0[i],mask};
		cv::Mat src2[2]={video1[i],mask};
		cv::mixChannels(src1, 2, &image1, 1, from_to, 4 );
		cv::mixChannels(src2, 2, &image2, 1, from_to, 4 );
		_extends1.push_back(image1);
		_extends2.push_back(image2);

	}
	
	float2* mat3d_f0=new float2[f0[0].step/sizeof(float2)*h*d];
	float2* mat3d_f1=new float2[f1[0].step/sizeof(float2)*h*d];
	float4* mat3d_v0=new float4[_extends1[0].step/sizeof(uchar4)*h*d];
	float4* mat3d_v1=new float4[_extends2[0].step/sizeof(uchar4)*h*d];


	for (int i=0;i<d;i++)
	{
		memcpy(mat3d_f0+i*h*f0[i].step/sizeof(float2),f0[i].data,f0[i].step*h);
		memcpy(mat3d_f1+i*h*f1[i].step/sizeof(float2),f1[i].data,f1[i].step*h);
		cv::Mat temp;
		_extends1[i].convertTo(temp, CV_32FC4);
		memcpy(mat3d_v0+i*h*temp.step/sizeof(float4),temp.data,temp.step*h);
		_extends2[i].convertTo(temp, CV_32FC4);
		memcpy(mat3d_v1+i*h*temp.step/sizeof(float4),temp.data,temp.step*h);
	}

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();
	cudaExtent extent;
	extent.width = w; // Note, for cudaArrays the width field is the width in elements, not bytes
	extent.height = h;
	extent.depth = d;
	cudaMalloc3DArray(&_forw0,&desc,extent,cudaArrayLayered);
	cudaMemcpy3DParms params = {0}; // Initialize to 0
	params.srcPtr =  make_cudaPitchedPtr( (void*)mat3d_f0, f0[0].step, w, h );
	params.dstArray = _forw0;
	params.kind = cudaMemcpyHostToDevice;
	params.extent = extent; // This is the extent used to allocate the cudaArray 'array'
	cudaMemcpy3D(&params);

	cudaMalloc3DArray(&_forw1,&desc,extent,cudaArrayLayered);
	params.srcPtr =  make_cudaPitchedPtr( (void*)mat3d_f1, f1[0].step, w, h );
	params.dstArray = _forw1;
	cudaMemcpy3D(&params);
		
	desc = cudaCreateChannelDesc<float4>();
	cudaMalloc3DArray(&_video0,&desc,extent,cudaArrayLayered);
	params.srcPtr =  make_cudaPitchedPtr( (void*)mat3d_v0, _extends1[0].step*sizeof(float), w, h );
	params.dstArray = _video0;
	params.kind = cudaMemcpyHostToDevice;
	params.extent = extent; // This is the extent used to allocate the cudaArray 'array'
	cudaMemcpy3D(&params);

	cudaMalloc3DArray(&_video1,&desc,extent,cudaArrayLayered);
	params.srcPtr =  make_cudaPitchedPtr( (void*)mat3d_v1, _extends2[0].step*sizeof(float), w, h );
	params.dstArray = _video1;
	params.kind = cudaMemcpyHostToDevice;
	params.extent = extent; // This is the extent used to allocate the cudaArray 'array'
	cudaMemcpy3D(&params);

	delete[] mat3d_f0;
	delete[] mat3d_f1;
	delete[] mat3d_v0;
	delete[] mat3d_v1;


	append_new(w,h,d);//level0

	//runable level
	float decres_fa=(float)(w*h*d)/(float)(Max_stage1);
	decres_fa=MAX(sqrt(decres_fa),1);
	w/=decres_fa;
	h/=decres_fa;
	
	//scale to pyramids
	int el_t,el_x,el_y;
	el_t=1;
	el_y=log2((float)h)-log2((float)start_res)+1;
	el_x=log2((float)w)-log2((float)start_res)+1;
	int maxl=MAX(MAX(el_x,el_y),el_t);
	for(int el=0;el<maxl;el++)
	{	
		PyramidLevel &lvl = append_new(w,h,d);
		if(maxl-el<=el_x) w=ceil(w/2.0f); 
		if(maxl-el<=el_y) h=ceil(h/2.0f); 
		if(maxl-el<=el_t) d=ceil(d/2.0f); 
	}

	
}

void Pyramid::build(std::vector<cv::Mat> &video0,std::vector<cv::Mat> &video1, std::vector<cv::Mat> &f0, std::vector<cv::Mat> &f1, std::vector<cv::Mat> &b0, std::vector<cv::Mat> &b1,int start_res)
{
	clear();

	int w,h,d,prev_d,prev_w,prev_h;
	w=video0[0].cols;
	h=video0[0].rows;
	d=video0.size();

	prev_w=w;
	prev_h=h;
	prev_d=d;
	
	for (int i=0;i<d;i++)
	{
		cv::Mat vector=cv::Mat::zeros(h,w,CV_32FC2);
		_vector.push_back(vector);
		cv::Mat qpath=cv::Mat::zeros(h,w,CV_32FC2);
		_qpath.push_back(qpath);
		
		cv::Mat image1=cv::Mat(h,w,CV_8UC4);
		cv::Mat image2=cv::Mat(h,w,CV_8UC4);
		int from_to[] = { 0,0,1,1,2,2,3,3 };
		cv::Mat mask=cv::Mat::zeros(h,w,CV_8UC1);
		cv::Mat src1[2]={video0[i],mask};
		cv::Mat src2[2]={video1[i],mask};
		cv::mixChannels(src1, 2, &image1, 1, from_to, 4 );
		cv::mixChannels(src2, 2, &image2, 1, from_to, 4 );
		int ex=max(w,h)*0.1;
		cv::Mat extends1=cv::Mat(h+ex*2,w+ex*2,CV_8UC4,cv::Scalar(255,255,255,255));
		cv::Mat extends2=cv::Mat(h+ex*2,w+ex*2,CV_8UC4,cv::Scalar(255,255,255,255));
		image1.copyTo(extends1(cv::Rect(ex, ex, w, h)));
		image2.copyTo(extends2(cv::Rect(ex, ex, w, h)));
		_extends1.push_back(extends1);
		_extends2.push_back(extends2);				
	}
						
	// use cardinal bspline3 prefilter for downsampling
	kernel::base *pre = new kernel::generalized(
		new kernel::discrete::delta,
		new kernel::discrete::sampled(new kernel::generating::bspline3),
		new kernel::generating::bspline3);
	// no additional discrete processing
	kernel::discrete::base *delta = new kernel::discrete::delta;
	// use mirror extension
	extension::base *ext = new extension::mirror;
	image::rgba<float> *rgba0=new image::rgba<float>[d];
	image::rgba<float> *rgba1=new image::rgba<float>[d];
	cv::Mat *forw0=new cv::Mat[d];
	cv::Mat *forw1=new cv::Mat[d];
	cv::Mat *back0=new cv::Mat[d];
	cv::Mat *back1=new cv::Mat[d];
	
	//scale to pyramids
	append_new(w,h,d);//level0

	//runable level
	float decres_fa=(float)(w*h*d)/(float)(Max_stage2);
	decres_fa=MAX(sqrt(decres_fa),1);
	w/=decres_fa;
	h/=decres_fa;	
	


	int el_t,el_x,el_y;
	el_t=log2((float)d)-log2((float)start_res)+1;
	el_y=log2((float)h)-log2((float)start_res)+1;
	el_x=log2((float)w)-log2((float)start_res)+1;
	el_x=el_y=MAX(el_x,el_y);
	int maxl=MAX(el_x,el_t);

	int factor_t=1;	 
	for(int el=0;el<maxl;el++)
	{	
		PyramidLevel &lvl = append_new(w,h,d);
		
		if (el==0)
		{
			for (int t=0;t<d;t++)
			{
						
				cudaArray *img0,*img1;
				lvl.img0.push_back(img0);
				lvl.img1.push_back(img1);
		
				cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float>();
				cudaMallocArray(&lvl.img0[t], &ccd, w, h);
				cudaMallocArray(&lvl.img1[t], &ccd, w, h);

				cudaArray *forward0,*forward1,*backward0,*backward1;
				lvl.f0.push_back(forward0);
				lvl.f1.push_back(forward1);	
				lvl.b0.push_back(backward0);
				lvl.b1.push_back(backward1);		

				ccd = cudaCreateChannelDesc<float2>();
				cudaMallocArray(&lvl.f0[t], &ccd, w, h);
				cudaMallocArray(&lvl.f1[t], &ccd, w, h);
				cudaMallocArray(&lvl.b0[t], &ccd, w, h);
				cudaMallocArray(&lvl.b1[t], &ccd, w, h);
				
				cv::Mat image0,	image1;					
				video0[t].convertTo(image0, CV_32FC3);
				video1[t].convertTo(image1, CV_32FC3);
				image::load(&rgba0[t],(float*)image0.data,prev_w,prev_h);		
				image::load(&rgba1[t],(float*)image1.data,prev_w,prev_h);	
				scale(h, w,pre, delta, delta, ext, &rgba0[t], &rgba0[t]);
				scale(h,w, pre, delta, delta, ext, &rgba1[t], &rgba1[t]);	
								
				float *dataf=new float[w*h];
				image::store_gray(dataf,rgba0[t]);
				cudaMemcpy2DToArray(lvl.img0[t], 0, 0, dataf,w*sizeof(float), w*sizeof(float), h,cudaMemcpyHostToDevice);
				image::store_gray(dataf,rgba1[t]);
				cudaMemcpy2DToArray(lvl.img1[t], 0, 0, dataf,w*sizeof(float), w*sizeof(float), h,cudaMemcpyHostToDevice);
				delete[] dataf;

 
				image::rgba<float> temp;
				image::load(&temp,(float*)f0[t].data,prev_w,prev_h,f0[t].step/sizeof(float2),-50,50);
				scale(h, w,pre, delta, delta, ext, &temp, &temp);
				forw0[t]=cv::Mat(h,w,CV_32FC2);
				image::store((float*)forw0[t].data,temp,forw0[t].step/sizeof(float2),-50,50);
				image::load(&temp,(float*)f1[t].data,prev_w,prev_h,f1[t].step/sizeof(float2),-50,50);
				scale(h, w,pre, delta, delta, ext, &temp, &temp);
				forw1[t]=cv::Mat(h,w,CV_32FC2);
				image::store((float*)forw1[t].data,temp,forw1[t].step/sizeof(float2),-50,50);
				image::load(&temp,(float*)b0[t].data,prev_w,prev_h,b0[t].step/sizeof(float2),-50,50);
				scale(h, w,pre, delta, delta, ext, &temp, &temp);
				back0[t]=cv::Mat(h,w,CV_32FC2);
				image::store((float*)back0[t].data,temp,back0[t].step/sizeof(float2),-50,50);
				image::load(&temp,(float*)b1[t].data,prev_w,prev_h,b1[t].step/sizeof(float2),-50,50);
				scale(h, w,pre, delta, delta, ext, &temp, &temp);
				back1[t]=cv::Mat(h,w,CV_32FC2);
				image::store((float*)back1[t].data,temp,back1[t].step/sizeof(float2),-50,50);
				

			
				float ratiox=(float)w/(float)prev_w;
				float ratioy=(float)h/(float)prev_h;

				if(ratiox<1||ratioy<1)
				{
					#pragma omp parallel for
					for(int y=0;y<h;y++)
					for(int x=0;x<w;x++)
					{
						forw0[t].at<cv::Vec2f>(y,x)[0]*=ratiox;
						forw0[t].at<cv::Vec2f>(y,x)[1]*=ratioy;
						forw1[t].at<cv::Vec2f>(y,x)[0]*=ratiox;
						forw1[t].at<cv::Vec2f>(y,x)[1]*=ratioy;			
						back0[t].at<cv::Vec2f>(y,x)[0]*=ratiox;
						back0[t].at<cv::Vec2f>(y,x)[1]*=ratioy;
						back1[t].at<cv::Vec2f>(y,x)[0]*=ratiox;
						back1[t].at<cv::Vec2f>(y,x)[1]*=ratioy;			
					}
				}					

				cudaMemcpy2DToArray(lvl.f0[t], 0, 0, forw0[t].data,forw0[t].step, w*sizeof(float2), h,cudaMemcpyHostToDevice);
				cudaMemcpy2DToArray(lvl.f1[t], 0, 0, forw1[t].data,forw1[t].step, w*sizeof(float2), h,cudaMemcpyHostToDevice);
				cudaMemcpy2DToArray(lvl.b0[t], 0, 0, back0[t].data,back0[t].step, w*sizeof(float2), h,cudaMemcpyHostToDevice);
				cudaMemcpy2DToArray(lvl.b1[t], 0, 0, back1[t].data,back1[t].step, w*sizeof(float2), h,cudaMemcpyHostToDevice);
			}			
		}
		else if(el<maxl-1)
		{	
			float ratiox=(float)w/(float)prev_w;
			float ratioy=(float)h/(float)prev_h;

			for (int t=0;t<d;t++)	
			{
				cudaArray *img0,*img1;
				lvl.img0.push_back(img0);
				lvl.img1.push_back(img1);
		
				cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float>();
				cudaMallocArray(&lvl.img0[t], &ccd, w, h);
				cudaMallocArray(&lvl.img1[t], &ccd, w, h);

				cudaArray *forward0,*forward1,*backward0,*backward1;
				lvl.f0.push_back(forward0);
				lvl.f1.push_back(forward1);	
				lvl.b0.push_back(backward0);
				lvl.b1.push_back(backward1);	

				ccd = cudaCreateChannelDesc<float2>();
				cudaMallocArray(&lvl.f0[t], &ccd, w, h);
				cudaMallocArray(&lvl.f1[t], &ccd, w, h);	
				cudaMallocArray(&lvl.b0[t], &ccd, w, h);
				cudaMallocArray(&lvl.b1[t], &ccd, w, h);	
				scale(h, w, pre, delta, delta, ext, &rgba0[min(t*factor_t,prev_d-1)], &rgba0[t]);
				scale(h, w, pre, delta, delta, ext, &rgba1[min(t*factor_t,prev_d-1)], &rgba1[t]);	
				

				float *dataf=new float[w*h];
				image::store_gray(dataf,rgba0[t]);
				cudaMemcpy2DToArray(lvl.img0[t], 0, 0, dataf,w*sizeof(float), w*sizeof(float), h,cudaMemcpyHostToDevice);
				image::store_gray(dataf,rgba1[t]);
				cudaMemcpy2DToArray(lvl.img1[t], 0, 0, dataf,w*sizeof(float), w*sizeof(float), h,cudaMemcpyHostToDevice);
				delete[] dataf;			
			}

			for(int t=0;t<prev_d;t++)
				{
					image::rgba<float> temp;
					image::load(&temp,(float*)forw0[t].data,prev_w,prev_h,forw0[t].step/sizeof(float2),-50,50);
					scale(h, w,pre, delta, delta, ext, &temp, &temp);
					forw0[t]=cv::Mat(h,w,CV_32FC2);
					image::store((float*)forw0[t].data,temp,forw0[t].step/sizeof(float2),-50,50);
					image::load(&temp,(float*)forw1[t].data,prev_w,prev_h,forw1[t].step/sizeof(float2),-50,50);
					scale(h, w,pre, delta, delta, ext, &temp, &temp);
					forw1[t]=cv::Mat(h,w,CV_32FC2);
					image::store((float*)forw1[t].data,temp,forw1[t].step/sizeof(float2),-50,50);
					image::load(&temp,(float*)back0[t].data,prev_w,prev_h,back0[t].step/sizeof(float2),-50,50);
					scale(h, w,pre, delta, delta, ext, &temp, &temp);
					back0[t]=cv::Mat(h,w,CV_32FC2);
					image::store((float*)back0[t].data,temp,back0[t].step/sizeof(float2),-50,50);
					image::load(&temp,(float*)back1[t].data,prev_w,prev_h,back1[t].step/sizeof(float2),-50,50);
					scale(h, w,pre, delta, delta, ext, &temp, &temp);
					back1[t]=cv::Mat(h,w,CV_32FC2);
					image::store((float*)back1[t].data,temp,back1[t].step/sizeof(float2),-50,50);

					
					if(ratiox<1||ratioy<1)
					{
						#pragma omp parallel for
						for(int y=0;y<h;y++)
						for(int x=0;x<w;x++)
						{
							forw0[t].at<cv::Vec2f>(y,x)[0]*=ratiox;
							forw0[t].at<cv::Vec2f>(y,x)[1]*=ratioy;
							forw1[t].at<cv::Vec2f>(y,x)[0]*=ratiox;
							forw1[t].at<cv::Vec2f>(y,x)[1]*=ratioy;	
							back0[t].at<cv::Vec2f>(y,x)[0]*=ratiox;
							back0[t].at<cv::Vec2f>(y,x)[1]*=ratioy;
							back1[t].at<cv::Vec2f>(y,x)[0]*=ratiox;
							back1[t].at<cv::Vec2f>(y,x)[1]*=ratioy;	
						}
					}
				}

			if(factor_t>1)
			{
				for (int t=0;t<d;t++)	
				{	
					if(t*factor_t>prev_d-1)
						continue;
				#pragma omp parallel for
				for(int y=0;y<h;y++)
					for(int x=0;x<w;x++)
					{							
						if(t*factor_t+1<prev_d)
						{	
							cv::Vec2f v=forw0[t*factor_t].at<cv::Vec2f>(y,x);
							float2 p=make_float2(x,y)+make_float2(v[0],v[1]);
							forw0[t*factor_t].at<cv::Vec2f>(y,x)+=BiLinear<cv::Vec2f>(forw0[t*factor_t+1],p);
							
							v=forw1[t*factor_t].at<cv::Vec2f>(y,x);
							p=make_float2(x,y)+make_float2(v[0],v[1]);
							forw1[t*factor_t].at<cv::Vec2f>(y,x)+=BiLinear<cv::Vec2f>(forw1[t*factor_t+1],p);							
						}	

						
						if(t>0)
						{	
							cv::Vec2f v=back0[t*factor_t].at<cv::Vec2f>(y,x);
							float2 p=make_float2(x,y)+make_float2(v[0],v[1]);
							back0[t*factor_t].at<cv::Vec2f>(y,x)+=BiLinear<cv::Vec2f>(back0[t*factor_t-1],p);
							
							v=back1[t*factor_t].at<cv::Vec2f>(y,x);
							p=make_float2(x,y)+make_float2(v[0],v[1]);
							back1[t*factor_t].at<cv::Vec2f>(y,x)+=BiLinear<cv::Vec2f>(back1[t*factor_t-1],p);							
						}	
							
					}	
			
			}
		}	
		
		for(int t=0;t<d;t++)
		{
			if(factor_t>1)
			{
				forw0[t]=forw0[min(t*factor_t,prev_d-1)].clone();
				forw1[t]=forw1[min(t*factor_t,prev_d-1)].clone();
				back0[t]=back0[min(t*factor_t,prev_d-1)].clone();
				back1[t]=back1[min(t*factor_t,prev_d-1)].clone();
			}			


			cudaMemcpy2DToArray(lvl.f0[t], 0, 0, forw0[t].data,forw0[t].step, w*sizeof(float2), h,cudaMemcpyHostToDevice);
			cudaMemcpy2DToArray(lvl.f1[t], 0, 0, forw1[t].data,forw1[t].step, w*sizeof(float2), h,cudaMemcpyHostToDevice);
			cudaMemcpy2DToArray(lvl.b0[t], 0, 0, back0[t].data,back0[t].step, w*sizeof(float2), h,cudaMemcpyHostToDevice);				
			cudaMemcpy2DToArray(lvl.b1[t], 0, 0, back1[t].data,back1[t].step, w*sizeof(float2), h,cudaMemcpyHostToDevice);	
		}			
				
		
	}		
		prev_w=w;
		prev_h=h;
		prev_d=d;
		if(maxl-el<=el_x) w=ceil(w/2.0f); 
		if(maxl-el<=el_y) h=ceil(h/2.0f); 
		if(maxl-el<=el_t) d=ceil((d+1)/2.0f),factor_t=2; else factor_t=1;
	}

	for(int i=m_data.size()-2;i>=0;i--)
	{
		if(m_data[i+1]->depth!=m_data[i]->depth)
			m_data[i]->factor_d=m_data[i+1]->factor_d*2;
		else
			m_data[i]->factor_d=m_data[i+1]->factor_d;
	}

	delete[] rgba0;
	delete[] rgba1;
	delete[] forw0;
	delete[] forw1;	
	delete[] back0;
	delete[] back1;	
}


template <class T>
inline T Pyramid::BiLinear(cv::Mat& img, float2 p)
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

PyramidLevel &Pyramid::append_new(int w, int h, int d)
{
	m_data.push_back(new PyramidLevel(w,h,d));
	return *m_data.back();
}

PyramidLevel::PyramidLevel(int w, int h, int d)
	: width(w), height(h),depth(d)
{
	// align on 128 byte boundary
	rowstride = (w + 31)/32 * 32;
	pagestride = rowstride*h;
	inv_wh = 1.0f/(w*h);
	impmask_rowstride = (w+4)/5+2;
	impmask_pagestride= impmask_rowstride*((h+4)/5+2);
	
	factor_d=1.0;

}
PyramidLevel::~PyramidLevel()
{
	for(int i=0;i<img0.size();i++)
		if(img0[i])
		cudaFreeArray(img0[i]);
	for(int i=0;i<img1.size();i++)
		if(img1[i])
		cudaFreeArray(img1[i]);
	for(int i=0;i<f0.size();i++)
		if(f0[i])
		cudaFreeArray(f0[i]);
	for(int i=0;i<f1.size();i++)
		if(f1[i])
		cudaFreeArray(f1[i]);
	for(int i=0;i<b0.size();i++)
		if(b0[i])
		cudaFreeArray(b0[i]);
	for(int i=0;i<b1.size();i++)
		if(b1[i])
		cudaFreeArray(b1[i]);
	
	img0.clear();
	img1.clear();
	f0.clear();
	f1.clear();
	b0.clear();
	b1.clear();

	ssim.cross.clear();
	ssim.luma.clear();
	ssim.mean.clear();
	ssim.var.clear();
	ssim.value.clear();
	ssim.counter.clear();

	tps.axy.clear();
	tps.b.clear();

	ui.axy.clear();
	ui.b.clear();

	temp.ref.clear();
	temp.mask.clear();

	v.clear();
		
	improving_mask.clear();
}

KernPyramidLevel::KernPyramidLevel(PyramidLevel &lvl)
{
	ssim.cross = &lvl.ssim.cross;
	ssim.var = &lvl.ssim.var;
	ssim.mean = &lvl.ssim.mean;
	ssim.luma = &lvl.ssim.luma;
	ssim.value = &lvl.ssim.value;
	ssim.counter = &lvl.ssim.counter;

	tps.axy = &lvl.tps.axy;
	tps.b = &lvl.tps.b;

	ui.axy = &lvl.ui.axy;
	ui.b = &lvl.ui.b;

	temp.ref = &lvl.temp.ref;
	temp.mask = &lvl.temp.mask;

	v = &lvl.v;

	improving_mask = &lvl.improving_mask;

	rowstride = lvl.rowstride;
	pagestride = lvl.pagestride;
	impmask_rowstride = lvl.impmask_rowstride;
	impmask_pagestride = lvl.impmask_pagestride;

	pixdim = make_int2(lvl.width, lvl.height);
	inv_wh = lvl.inv_wh;
	factor_d=lvl.factor_d;
}



template <class T>
__global__ void internal_vector_to_image(rod::dimage_ptr<T> res,
							 const T *v, KernPyramidLevel lvl,
							 T mult)
{
	int tx = threadIdx.x, ty = threadIdx.y,
		bx = blockIdx.x, by = blockIdx.y;

	int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

	if(!res.is_inside(pos))
		return;

	res += res.offset_at(pos);
	v += mem_index(lvl, pos);

	*res = *v * mult;
}


template <class T>
void internal_vector_to_image(rod::dimage<T> &dest,
							  const T* orig,
							  const PyramidLevel &lvl,
							  T mult)
{
	dim3 bdim(32,8),
		 gdim((lvl.width+bdim.x-1)/bdim.x,
			  (lvl.height+bdim.y-1)/bdim.y);

	// const cast is harmless
	KernPyramidLevel klvl(const_cast<PyramidLevel&>(lvl));

	dest.resize(lvl.width, lvl.height);

	internal_vector_to_image<T><<<gdim, bdim>>>(&dest, orig, klvl, mult);
}

template
void internal_vector_to_image(rod::dimage<float> &dest,
							  const float* orig,
							  const PyramidLevel &lvl, float mult);

template
void internal_vector_to_image(rod::dimage<float2> &dest,
							  const float2* orig,
							  const PyramidLevel &lvl, float2 mult);


template <class T>
__global__ void image_to_internal_vector(T *v,
										 rod::dimage_ptr<const T> in,
										 KernPyramidLevel lvl,
										 T mult)
{
	int tx = threadIdx.x, ty = threadIdx.y,
		bx = blockIdx.x, by = blockIdx.y;

	int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

	if(!in.is_inside(pos))
		return;

	in += in.offset_at(pos);
	v += mem_index(lvl, pos);

	*v = *in * mult;
}

template <class T>
void image_to_internal_vector(T* dest,
							  const rod::dimage<T> &orig,
							  const PyramidLevel &lvl,
							  T mult)
{
	assert(lvl.width == orig.width());
	assert(lvl.height == orig.height());

	// const cast is harmless
	KernPyramidLevel klvl(const_cast<PyramidLevel&>(lvl));

	//dest.resize(lvl.width*lvl.height);

	dim3 bdim(32,8),
		 gdim((lvl.width+bdim.x-1)/bdim.x,
			  (lvl.height+bdim.y-1)/bdim.y);

	image_to_internal_vector<T><<<gdim, bdim>>>(dest, &orig, klvl, mult);
}

template
void image_to_internal_vector(float *dest,
							  const rod::dimage<float> &orig,
							  const PyramidLevel &lvl, float mult);

template
void image_to_internal_vector(float2 *dest,
							  const rod::dimage<float2> &orig,
							  const PyramidLevel &lvl, float2 mult);

