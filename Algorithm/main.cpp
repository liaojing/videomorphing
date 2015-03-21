#include <iostream>
#include <util/timer.h>
#include <util/image_ops.h>
#include <cstdio>
#include "imgio.h"
#include "pyramid.h"
#include "morph.h"
#include "param_io.h"
#include "parameters.h"
#if CUDA_SM < 20
#   include <util/cuPrintf.cuh>
#endif
#include <opencv2/core/wimage.hpp>

struct callback_data
{
    callback_data(const rod::dimage<float3> &_dimg0,
                  const rod::dimage<float3> &_dimg1)
        : dimg0(_dimg0)
        , dimg1(_dimg1)
    {
    }

    const rod::dimage<float3> &dimg0, &dimg1;
};

bool callback(const std::string &text, int cur, int max,
              const rod::dimage<float2> *halfway,
              const rod::dimage<float> *ssim_error,
              void *data)
{
    static int idx=0;
    if(halfway)
    {
        callback_data *cbdata = reinterpret_cast<callback_data *>(data);


        rod::dimage<float3> result;
        render_halfway_image(result, *halfway, cbdata->dimg0, cbdata->dimg1);

        char buffer[100];
        sprintf(buffer,"hw_%03d.png",idx++);

        std::cout << cur << "/" << max << ": " << buffer << std::endl;

        save(buffer,result);
    }

      //  std::cout << "has halfway" << std::endl;

    //std::cout << cur << "/" << max << ": " << text << std::endl;
    return true;
}




void OpticalFlow(std::vector<cv::Mat>& v1, std::vector<cv::Mat>& v2, float2 *f1,float2 *f2,float2 *b1, float2 *b2)
{
	std::vector<cv::Mat> gray1,gray2;	

	int rows=v1[0].rows;
	int cols=v1[0].cols;
	int times=v1.size();
	
	for(int i=0;i<times;i++)
	{
		cv::Mat Img1,Img2;	
		cvtColor(v1[i], Img1, CV_RGB2GRAY);
		cvtColor(v2[i], Img2, CV_RGB2GRAY);

		gray1.push_back(Img1);
		gray2.push_back(Img2);		
	}

#pragma omp parallel for
	for(int t=0;t<times;t++)
	{		
		cv::Mat flow1=cv::Mat(rows,cols,CV_32FC2);
		cv::Mat flow2=cv::Mat(rows,cols,CV_32FC2);

		
		if (t<10)
		{
			flow1=cv::Mat(rows,cols,CV_32FC2,cv::Scalar(0,-30));
			flow2=cv::Mat(rows,cols,CV_32FC2,cv::Scalar(0,-15));
		}
		else if(t<20)
		{
			flow1=cv::Mat(rows,cols,CV_32FC2,cv::Scalar(0,15));
			flow2=cv::Mat(rows,cols,CV_32FC2,cv::Scalar(0,-15));
		}
		else
		{
			flow1=cv::Mat(rows,cols,CV_32FC2,cv::Scalar(0,15));
			flow2=cv::Mat(rows,cols,CV_32FC2,cv::Scalar(0,30));
		}

		memcpy(f1+t*cols*rows,flow1.data,cols*rows*sizeof(float2));
		memcpy(f2+t*cols*rows,flow2.data,cols*rows*sizeof(float2));
	}

#pragma omp parallel for
	for(int t=times-1;t>=0;t--)
	{	
		cv::Mat flow1=cv::Mat(rows,cols,CV_32FC2);
		cv::Mat flow2=cv::Mat(rows,cols,CV_32FC2);	
					 		
		if (t<11)
		{
			flow1=cv::Mat(rows,cols,CV_32FC2,cv::Scalar(0,30));
			flow2=cv::Mat(rows,cols,CV_32FC2,cv::Scalar(0,15));
		}
		else if(t<21)
		{
			flow1=cv::Mat(rows,cols,CV_32FC2,cv::Scalar(0,-15));
			flow2=cv::Mat(rows,cols,CV_32FC2,cv::Scalar(0,15));
		}
		else
		{
			flow1=cv::Mat(rows,cols,CV_32FC2,cv::Scalar(0,-15));
			flow2=cv::Mat(rows,cols,CV_32FC2,cv::Scalar(0,-30));
		}	
				

		memcpy(b1+t*cols*rows,flow1.data,cols*rows*sizeof(float2));
		memcpy(b2+t*cols*rows,flow2.data,cols*rows*sizeof(float2));
	}   

	gray1.clear();
	gray2.clear();	
	
}

template <class T>
void  Mat2Dimg(cv::Mat &temp,rod::dimage<T> &dimg)
{
	IplImage temp2 = temp;

	cv::WImageView<unsigned char> img = &temp2;

	dimg.resize(img.Width(), img.Height());

	switch(img.Channels())
	{
	case 1:
		{
			rod::dimage<uchar1> temp;
			temp.copy_from_host(img(0,0), img.Width(),
				img.Height(), img.WidthStep());
			convert(&dimg, &temp, false);
		}
		break;
	case 2:
		{
			rod::dimage<uchar2> temp;
			temp.copy_from_host((uchar2 *)img(0,0), img.Width(),
				img.Height(), img.WidthStep()/sizeof(uchar2));
			convert(&dimg, &temp, false);
		}
		break;
	case 3:
		{
			cv::Mat temp;
			cvtColor((cv::Mat)img.Ipl(), temp, cv::COLOR_RGB2RGBA);
			IplImage temp2 = temp;

			cv::WImageViewC<unsigned char,4> temp3(&temp2);

			rod::dimage<uchar4> temp4;
			temp4.copy_from_host((uchar4 *)temp3(0,0), temp3.Width(),
				temp3.Height(), temp3.WidthStep()/sizeof(uchar4));
			convert(&dimg, &temp4, false);
		}
		break;
	case 4:
		{
			rod::dimage<uchar4> temp;
			temp.copy_from_host((uchar4 *)img(0,0), img.Width(),
				img.Height(), img.WidthStep()/sizeof(uchar4));
			convert(&dimg, &temp, false);
		}
		break;
	default:
		throw std::runtime_error("Invalid number of channels");
	}
}


int main(int argc, char *argv[])
{
    try
    {
#if CUDA_SM < 20
        cudaPrintfInit();
#endif

//         if(argc <= 1 || argc >= 4)
//             throw std::runtime_error("Bad arguments");

        // default values defined in default ctor
        
		std::vector<cv::Mat> video0,video1;
		for(int i=0;i<MAX_FRAME;i++)
		{
			char str[]="c:\\users\\jliaoaa\\desktop\\ball\\video1\\frame000.png";
			int a=i/100;
			int b=i%100/10;
			int c=i%10;
			str[35]='1';
			str[42]='0'+a;
			str[43]='0'+b;
			str[44]='0'+c;
			cv::Mat image=cv::imread(str);
			cvtColor(image,image,CV_BGR2RGB);
			video0.push_back(image);

			str[35]='2';
			str[42]='0'+a;
			str[43]='0'+b;
			str[44]='0'+c;
			image=cv::imread(str);
			cvtColor(image,image,CV_BGR2RGB);
			video1.push_back(image);
		}		
				

		int rows=video1[0].rows;
		int cols=video1[0].cols;
		int times=video1.size();
		float2 *f0,*f1,*b0,*b1;
		f0=new float2[times*rows*cols];
		f1=new float2[times*rows*cols];
		b0=new float2[times*rows*cols];
		b1=new float2[times*rows*cols];
		OpticalFlow(video0,video1,f0,f1,b0,b1);

		Parameters params;
		params.w_ui=100.0f;
		params.w_ssim=1.0f;
		params.w_tps=0.0005f;
		params.w_temp=0.5f;
		params.ssim_clamp=0.0f;
		params.bcond=BCOND_NONE;
		params.eps=0.01;
		params.start_res=16;
		params.max_iter_drop_factor=2;
		params.max_iter=2000;

		std::vector<Conp> llist;
		std::vector<Conp> rlist;
		std::vector<Connect> clist;
		for(int i=0;i<MAX_FRAME;i++)
		{
			Conp lp;
			Conp rp;
			Connect c;
			if(i==0)
 			{
				lp.p.x=263;
				lp.p.y=404;
				lp.p.z=i;
				lp.p.w=1;
				lp.weight=1.0f;
				llist.push_back(lp);

				rp.p.x=262;
				rp.p.y=424;
				rp.p.z=i;
				rp.p.w=1;
				rp.weight=1.0f;
				rlist.push_back(rp);

			}
 			else
 			{
				lp=llist[llist.size()-1];
				lp.p.x+=f0[lp.p.z*rows*cols+lp.p.y*cols+lp.p.x].x;
				lp.p.y+=f0[lp.p.z*rows*cols+lp.p.y*cols+lp.p.x].y;
				lp.p.z=i;
				lp.p.w=1;
				lp.weight=1.0f;
				llist.push_back(lp);

				rp=rlist[rlist.size()-1];
				rp.p.x+=f1[rp.p.z*rows*cols+rp.p.y*cols+rp.p.x].x;
				rp.p.y+=f1[rp.p.z*rows*cols+rp.p.y*cols+rp.p.x].y;
				rp.p.z=i;
				rp.p.w=1;
				rp.weight=1.0f;
				rlist.push_back(rp);
 			}
			
			c.li.x=0;
			c.li.y=i;
			c.ri.x=0;
			c.ri.y=i;
			clist.push_back(c);	
		}	

		params.lp.push_back(llist);
		params.rp.push_back(rlist);
		params.cnt.push_back(clist);


 		Pyramid pyramid;
 		pyramid.build(video0, video1,f0,f1,b0,b1, params.start_res, params.verbose);

		
        Morph morph(params,pyramid);

#if 0
        callback_data cbdata(morph.image0(), morph.image1());
        morph.set_callback(&callback, &cbdata);
#endif

        std::vector<rod::dimage<float2>> halfway;
		for(int i=0;i<MAX_FRAME;i++)
		{
			halfway.push_back(rod::dimage<float2>());
		}
		
		if(!morph.calculate_halfway_parametrization(halfway))
		{
			
			throw std::runtime_error("Aborted");
		}       

        /* to convert from halfway (a dimage<float2>) to std::vector<float2>,
           do:

           std::vector<float2> host; // will have size halfway.width()*halfway.height()
           halfway.copy_to_host(host);
        */

		for(int i=0;i<MAX_FRAME;i++)
		{

			 rod::dimage<float3> result;
			 rod::dimage<float3> dimg0,dimg1;
			 
			Mat2Dimg(video0[i],dimg0);
			Mat2Dimg(video1[i],dimg1);
			
			 render_halfway_image(result, halfway[i], dimg0, dimg1);
			
			char filename[128];
			sprintf(filename,"result%d.png",i);
			save(filename,result);
		}

	
#if CUDA_SM < 20
        cudaPrintfDisplay(stdout, true);
        cudaPrintfEnd();
#endif
    }
    catch(std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}


