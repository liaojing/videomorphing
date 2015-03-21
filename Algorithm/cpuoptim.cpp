#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
//#include <util/linalg.h>
// #include <util/dmath.h>
// #include <util/cuda_traits.h>
// #include "parameters.h"
// #include "pyramid.h"
#include "morph.h"

#define next2power(x) (int)ceil(log((float)x)/log(2.0f))


void Morph::cpu_optimize_level(PyramidLevel &lvl,PyramidLevel &lv0) const
{
	int w=lvl.width;
	int h=lvl.height;
	int d=lvl.depth;
	int num=w*h;	
	float2 *v=new float2[lvl.pagestride*d];

	for(int z=0;z<d;z++)
	{
		cv::Mat   A   =  cv::Mat::zeros(num,num,CV_32FC1);
		cv::Mat   Bx  =  cv::Mat::zeros(num,1,CV_32FC1);
		cv::Mat   By  =  cv::Mat::zeros(num,1,CV_32FC1);
		cv::Mat   X  =   cv::Mat::zeros(num,1,CV_32FC1);
		cv::Mat   Y  =   cv::Mat::zeros(num,1,CV_32FC1);

		//set matrixs for tps
		#pragma omp parallel for
		for(int y=0;y<h;y++)
		for(int x=0;x<w;x++)
		{
			int i=y*w+x;
			//dxx
			if(x>1)
				A.at<float>(i,i-2)+=1.0f*m_params.w_tps*2.0f,	A.at<float>(i,i-1)+=-2.0f*m_params.w_tps*2.0f,	A.at<float>(i,i)+=1.0f*m_params.w_tps*2.0f;
			if(x>0&&x<w-1)
				A.at<float>(i,i-1)+=-2.0f*m_params.w_tps*2.0f, A.at<float>(i,i)+=4.0f*m_params.w_tps*2.0f,	A.at<float>(i,i+1)+=-2.0f*m_params.w_tps*2.0f;
			if(x<w-2)
				A.at<float>(i,i)+=1.0f*m_params.w_tps*2.0f,	A.at<float>(i,i+1)+=-2.0f*m_params.w_tps*2.0f,	A.at<float>(i,i+2)+=1.0f*m_params.w_tps*2.0f;
			//dyy
			if(y>1)
				A.at<float>(i,i-2*w)+=1.0f*m_params.w_tps*2.0f, A.at<float>(i,i-w)+=-2.0f*m_params.w_tps*2.0f,	A.at<float>(i,i)+=1.0f*m_params.w_tps*2.0f;
			if(y>0&&y<h-1)
				A.at<float>(i,i-w)+=-2.0f*m_params.w_tps*2.0f,  A.at<float>(i,i)+=4.0f*m_params.w_tps*2.0f,		A.at<float>(i,i+w)+=-2.0f*m_params.w_tps*2.0f;
			if(y<h-2)
				A.at<float>(i,i)+=1.0f*m_params.w_tps*2.0f,	 A.at<float>(i,i+w)+=-2.0f*m_params.w_tps*2.0f,	A.at<float>(i,i+2*w)+=1.0f*m_params.w_tps*2.0f;

			//dxy
			if(x>0&&y>0)
				A.at<float>(i,i-w-1)+=2.0f*m_params.w_tps*2.0f,	A.at<float>(i,i-w)+=-2.0f*m_params.w_tps*2.0f,	A.at<float>(i,i-1)+=-2.0f*m_params.w_tps*2.0f,	A.at<float>(i,i)+=2.0f*m_params.w_tps*2.0f;
			if(x<w-1&&y>0)
				A.at<float>(i,i-w)+=-2.0f*m_params.w_tps*2.0f,	A.at<float>(i,i-w+1)+=2.0f*m_params.w_tps*2.0f,	A.at<float>(i,i)+=2.0f*m_params.w_tps*2.0f,	A.at<float>(i,i+1)+=-2.0f*m_params.w_tps*2.0f;
			if(x>0&&y<h-1)
				A.at<float>(i,i-1)+=-2.0f*m_params.w_tps*2.0f,	A.at<float>(i,i)+=2.0f*m_params.w_tps*2.0f,	A.at<float>(i,i+w-1)+=2.0f*m_params.w_tps*2.0f,	A.at<float>(i,i+w)+=-2.0f*m_params.w_tps*2.0f;
			if(x<w-1&&y<h-1)
				A.at<float>(i,i)+=2.0f*m_params.w_tps*2.0f,	A.at<float>(i,i+1)+=-2.0f*m_params.w_tps*2.0f,	A.at<float>(i,i+w)+=-2.0f*m_params.w_tps*2.0f,	A.at<float>(i,i+w+1)+=2.0f*m_params.w_tps*2.0f;

		}

		//set matrix for ui
		float factor=pow(2.0f,next2power(d)-next2power(lv0.depth));
		for(int k=0;k<m_params.cnt.size();k++)
		for(int l=0;l<m_params.cnt[k].size();l++)
		{
			int2 li=m_params.cnt[k][l].li;
			int2 ri=m_params.cnt[k][l].ri;
			if(m_params.lp[li.x][li.y].p.z*factor!=z)
				continue;

			float x0=(m_params.lp[li.x][li.y].p.x+0.5)/lv0.width*w-0.5f;
			float y0=(m_params.lp[li.x][li.y].p.y+0.5)/lv0.height*h-0.5f;				
			float x1=(m_params.rp[ri.x][ri.y].p.x+0.5)/lv0.width*w-0.5f;
			float y1=(m_params.rp[ri.x][ri.y].p.y+0.5)/lv0.height*h-0.5f;
			float weight=MIN(m_params.lp[li.x][li.y].weight,m_params.rp[ri.x][ri.y].weight);

			float con_x=(x0+x1)/2.0f;
			float con_y=(y0+y1)/2.0f;			
			float vx=(x1-x0)/2.0f;
			float vy=(y1-y0)/2.0f;

			for(int y=floor(con_y);y<=ceil(con_y);y++)
				for(int x=floor(con_x);x<=ceil(con_x);x++)
				{
					if(x >=0 && x < w && y >= 0 && y < h)
					{
						float bilinear_w=(1.0-fabs(y-con_y))*(1.0-fabs(x-con_x))*weight;
						int i=y*w+x;
						A.at<float>(i,i)+=bilinear_w*m_params.w_ui*lvl.inv_wh*2.0f;
						Bx.at<float>(i,0)+=bilinear_w*vx*m_params.w_ui*lvl.inv_wh*2.0f;
						By.at<float>(i,0)+=bilinear_w*vy*m_params.w_ui*lvl.inv_wh*2.0f;
					}
				}

		}

		//set boundary condition
		int x,y,index;
		switch(m_params.bcond)
		{
		case BCOND_NONE:
			break;

		case BCOND_CORNER://corner

			x=0,y=0;
			index=y*w+x;
			A.at<float>(index,index)+=m_params.w_ui*lvl.inv_wh;

			x=0,y=h-1;
			index=y*w+x;
			A.at<float>(index,index)+=m_params.w_ui*lvl.inv_wh;

			x=w-1,y=h-1;
			index=y*w+x;
			A.at<float>(index,index)+=m_params.w_ui*lvl.inv_wh;

			x=w-1,y=0;
			index=y*w+x;
			A.at<float>(index,index)+=m_params.w_ui*lvl.inv_wh;			

			break;


		case BCOND_BORDER:
			for(int t=0;t<d;t++)
			{
				for (x=0;x<w;x++)
				{
					y=0;
					index=y*w+x;
					A.at<float>(index,index)+=m_params.w_ui*lvl.inv_wh;

					y=h-1;
					index=y*w+x;
					A.at<float>(index,index)+=m_params.w_ui*lvl.inv_wh;
				}

				for(y=1;y<h-1;y++)
				{
					x=0;
					index=y*w+x;
					A.at<float>(index,index)+=m_params.w_ui*lvl.inv_wh;

					x=w-1;
					index=y*w+x;
					A.at<float>(index,index)+=m_params.w_ui*lvl.inv_wh;

				}
			}
			break;
		}


		cv::Mat A_inv=A.inv();
		if(countNonZero(A_inv)<1)
			A_inv=A.inv(cv::DECOMP_SVD);

		X=A_inv*Bx;
		Y=A_inv*By;		

		//load to vx,vy				
		#pragma omp parallel for
		for(int y=0; y<h; ++y)
		{
			for(int x=0; x<w; ++x)
			{
				int i = mem_index(lvl,make_int2(x,y))+z*lvl.pagestride;
				assert(i < lvl.pagestride*d);

				v[i].x = X.at<float>(y*w+x,0);
				v[i].y = Y.at<float>(y*w+x,0);
			}
		}
					
	}	

	lvl.v.copy_from_host(v, lvl.pagestride*d);
	delete[] v;
}