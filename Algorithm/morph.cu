#include <util/dimage.h>
#include <util/image_ops.h>
#include <util/linalg.h>
#include <util/symbol.h>
#include <util/timer.h>
#include <util/dmath.h>
#include <ctime>
#include <assert.h>
#include "stencils.h"
#include "imgio.h"
#include "morph.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>

#if CUDA_SM < 20
#   include <util/cuPrintf.cu>
#else
extern "C"
{
extern _CRTIMP __host__ __device__ __device_builtin__ int     __cdecl printf(const char*, ...);
}
#endif

__constant__ rod::Matrix<fmat5,5,5> c_tps_data;
__constant__ rod::Matrix<imat3,5,5> c_improvmask;
__constant__ imat3 c_improvmask_offset;
__constant__ rod::Matrix<imat5,5,5> c_iomask;

texture<float, 2, cudaReadModeElementType> tex_img0, tex_img1;
texture<float2, 2, cudaReadModeElementType> tex_f0, tex_f1,tex_v;

extern void initialize_temp(PyramidLevel &lvl,int i,int dir);


__device__ int isignbit(int i)/*{{{*/
{
    return (unsigned)i >> 31;
}/*}}}*/
__device__ int2 calc_border(int2 p, int2 dim)/*{{{*/
{
    int2 B;

    // I'm proud of the following lines, and they're faster too

#if 1
    int s = isignbit(p.x-2);
    int aux = p.x - (dim.x-2);
    B.x = p.x*s + (!s)*(2 + (!isignbit(aux))*(1+aux));

    s = isignbit(p.y-2);
    aux = p.y - (dim.y-2);
    B.y = p.y*s + (!s)*(2 + (!isignbit(aux))*(1+aux));
#endif


#if 0
    if(p.y==0)
        B.y = 0;
    else if(p.y==1)
        B.y = 1;
    else if(p.y==dim.y-2)
        B.y = 3;
    else if(p.y==dim.y-1)
        B.y = 4;
    else
        B.y = 2;

    if(p.x==0)
        B.x = 0;
    else if(p.x==1)
        B.x = 1;
    else if(p.x==dim.x-2)
        B.x = 3;
    else if(p.x==dim.x-1)
        B.x = 4;
    else
        B.x = 2;
#endif

    return B;
}/*}}}*/

// Auxiliary functions --------------------------------------------------------

__device__ float ssim(float2 mean, float2 var, float cross, /*{{{*/
                      float counter, float ssim_clamp)
{
	if(counter <= 1)
        return 0;

    const float c2 = pow2(255 * 0.03); // 58.5225

    mean /= counter;

    var = (var-counter*mean*mean)/counter;
    var.x = max(0.0f, var.x);
    var.y = max(0.0f, var.y);


    cross = (cross - counter*mean.x*mean.y)/counter;
    

    const float c3 = 29.26125f;
    float2 sqrtvar = sqrt(var);

    float c = (2*sqrtvar.x*sqrtvar.y + c2) / (var.x + var.y + c2),
          s = (abs(cross) + c3)/(sqrtvar.x*sqrtvar.y + c3);

    float value = c*s;
    

    //float value = (2*cross + c2)/(var.x+var.y+c2);

    return max(min(1.0f,value),ssim_clamp);
    //return saturate(1.0f-c*s);

      
}/*}}}*/

// Level processing -----------------------------------------------------------

Morph::Morph(Parameters &params, Pyramid &pyramid,bool &run_flag)
    : m_cb(run_flag)    
    , m_params(params)
	, m_pyramid(pyramid)
{
	
   _total_l=pyramid.size()-1;
   _current_l=_total_l;
   _total_iter=_current_iter=0;
   _max_iter=params.max_iter;
   int iter_num=params.max_iter;
	for (int el=_total_l-1;el>=0;el--)
	{
		if(el>0)
		{
			_total_iter+=iter_num*pyramid[el].width*pyramid[el].height*pyramid[el].depth;
			iter_num/=params.max_iter_drop_factor;
		}	
	}
}

Morph::~Morph()
{
	
}



bool Morph::calculate_halfway_parametrization()
{
    cpu_optimize_level(m_pyramid[_total_l],m_pyramid[0]);   
   
   for(_current_l=_total_l-1;_current_l>0;_current_l--)
	{
		if(m_cb)
		{
			int el=_current_l;
			upsample(m_pyramid[el], m_pyramid[el+1]);			       
 			initialize_level(m_pyramid[el],m_pyramid[0]);
			optimize_level(m_pyramid[el]);
			clear_level(m_pyramid[el]);		
			_max_iter/=m_params.max_iter_drop_factor;	
		}	
	}   
	
    return true;
}

const int INIT_BW = 32,
          INIT_BH = 4,
          INIT_NB = 4;
__global__
__launch_bounds__(INIT_BW*INIT_BH, INIT_NB)
__global__ void kernel_initialize_level(KernPyramidLevel lvl,int page,
                                        float ssim_clamp)
{
    int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

    if(!lvl.contains(pos))
        return;

    int2 B = calc_border(pos,lvl.pixdim);

    int counter=0;
    float2 mean = {0,0}, 
           var = {0,0};
    float cross = 0;
    float2 tps_b = {0,0};

#pragma unroll
    for(int i=0; i<5; ++i)
    {
#pragma unroll
        for(int j=0; j<5; ++j)
        {
            if(c_iomask[B.y][B.x][i][j] == 0)
                continue;

            int nbidx = mem_index(lvl,pos + make_int2(j,i)-2)+lvl.pagestride*page;

            float2 v = lvl.v[nbidx];

            assert(lvl.contains(pos.x+j-2,pos.y+i-2) || (v.x==0 && v.y==0));

            float2 tpos = make_float2((pos + make_int2(j,i) - 2)) + 0.5f;

            float2 luma;
            luma.x = tex2D(tex_img0, tpos.x - v.x, tpos.y - v.y),
            luma.y = tex2D(tex_img1, tpos.x + v.x, tpos.y + v.y);

            //assert(lvl.contains(pix.pos.x+j-2,pix.pos.y+i-2) || (luma.x==0 && luma.y==0));

            // this is the right thing to do, but result is better without it
            luma *= c_iomask[B.y][B.x][i][j];

            assert(lvl.contains(pos.x+j-2,pos.y+i-2) || c_tps_data[B.y][B.x][i][j]==0);
            tps_b += v*c_tps_data[B.y][B.x][i][j];


            assert(lvl.contains(pos.x+j-2,pos.y+i-2) || c_iomask[B.y][B.x][i][j]==0);
            counter += c_iomask[B.y][B.x][i][j];
            mean += luma;
            var += luma*luma;
            cross += luma.x*luma.y;

            if(i==2 && j==2)
                lvl.ssim.luma[nbidx] = luma;
        }
    }

    int idx = mem_index(lvl, pos)+lvl.pagestride*page;
    lvl.ssim.counter[idx] = counter;
    lvl.ssim.mean[idx] = mean;
    lvl.ssim.var[idx] = var;
    lvl.ssim.cross[idx] = cross;
    lvl.ssim.value[idx] = ssim(mean, var, cross, counter, ssim_clamp);

    lvl.tps.axy[idx] = c_tps_data[B.y][B.x][2][2]/2;
    lvl.tps.b[idx] = tps_b;
}

__global__
void init_improving_mask(unsigned int *impmask, int bw, int bh)
{
    int bx = blockIdx.x*blockDim.x + threadIdx.x,
        by = blockIdx.y*blockDim.y + threadIdx.y;

    if(bx >= bw || by >= bh)
        return;

    if(bx==0 || by==0 || bx == bw-1 || by == bh-1)
        impmask[by*bw+bx] = 0;
    else
        impmask[by*bw+bx] = (1<<25)-1;

}



void Morph::initialize_level(PyramidLevel &lvl,PyramidLevel &lv0)
{
    rod::Matrix<fmat5,5,5> tps;
    calc_tps_stencil(tps);
    copy_to_symbol(c_tps_data,tps);

    rod::Matrix<imat3,5,5> improvmask_check;
    rod::Matrix<int,3,3> improvmask_off;
    calc_nb_improvmask_check_stencil(lvl, improvmask_check, improvmask_off);
    copy_to_symbol(c_improvmask,improvmask_check);
    copy_to_symbol(c_improvmask_offset,improvmask_off);

    rod::Matrix<imat5,5,5> iomask;
    calc_nb_io_stencil(iomask);
    copy_to_symbol(c_iomask,iomask);

	size_t size = lvl.pagestride*lvl.depth;	

	lvl.ssim.cross.resize(size);
	lvl.ssim.luma.resize(size);
	lvl.ssim.mean.resize(size);
	lvl.ssim.var.resize(size);
	lvl.ssim.value.resize(size);
	lvl.ssim.counter.resize(size);

	lvl.tps.axy.resize(size);
	lvl.tps.b.resize(size);

	lvl.ui.axy.resize(size);
	lvl.ui.b.resize(size);

	lvl.temp.ref.resize(size);
	lvl.temp.mask.resize(size);
		
	lvl.improving_mask.resize(lvl.impmask_pagestride*lvl.depth);

    lvl.ssim.mean.fill(0);
    lvl.ssim.var.fill(0);
    lvl.ssim.luma.fill(0);
    lvl.ssim.cross.fill(0);
    lvl.ssim.value.fill(0);
    lvl.ssim.counter.fill(0);
	
    lvl.tps.axy.fill(0);
    lvl.tps.b.fill(0);

    lvl.ui.axy.fill(0);
    lvl.ui.b.fill(0);

	lvl.temp.ref.fill(0);
	lvl.temp.mask.fill(0);

    tex_img0.normalized = false;
    tex_img0.filterMode = cudaFilterModeLinear;
    tex_img0.addressMode[0] = tex_img0.addressMode[1] = cudaAddressModeClamp;

    tex_img1.normalized = false;
    tex_img1.filterMode = cudaFilterModeLinear;
    tex_img1.addressMode[0] = tex_img1.addressMode[1] = cudaAddressModeClamp;

     
   	for(int i=0;i<lvl.depth;i++)
	{	
		cudaBindTextureToArray(tex_img0, lvl.img0[i]);
		cudaBindTextureToArray(tex_img1, lvl.img1[i]);
		
		dim3 bdim(INIT_BW,INIT_BH),
         gdim((lvl.width+bdim.x-1)/bdim.x,
              (lvl.height+bdim.y-1)/bdim.y);

		kernel_initialize_level<<<gdim,bdim>>>(lvl, i,m_params.ssim_clamp);
		
		int2 blk = make_int2((lvl.width+4)/5+2, (lvl.height+4)/5+2);

		gdim = dim3((blk.x + bdim.x-1)/bdim.x,
                (blk.y + bdim.y-1)/bdim.y);

		init_improving_mask<<<gdim,bdim>>>(lvl.improving_mask+i*lvl.impmask_pagestride, blk.x, blk.y);
	}
	

    // initialize ui data in cpu since usually there aren't so much points

    std::vector<float> ui_axy(lvl.ui.axy.size(), 0);
    std::vector<float2> ui_b(lvl.ui.b.size(), make_float2(0,0)),  v;
 	lvl.v.copy_to_host(v);
	int factor=lv0.factor_d/lvl.factor_d;
	for(int z=0;z<lvl.depth;z++)
	{		
		int conz=min(z*factor,lv0.depth-1);
		for(int k=0;k<m_params.cnt.size();k++)
		for(int l=0;l<m_params.cnt[k].size();l++)
		{
			int2 li=m_params.cnt[k][l].li;
			int2 ri=m_params.cnt[k][l].ri;
			if(conz!=m_params.lp[li.x][li.y].p.z)
				continue;

			float x0=(m_params.lp[li.x][li.y].p.x+0.5)/lv0.width*lvl.width-0.5f;
			float y0=(m_params.lp[li.x][li.y].p.y+0.5)/lv0.height*lvl.height-0.5f;				
			float x1=(m_params.rp[ri.x][ri.y].p.x+0.5)/lv0.width*lvl.width-0.5f;
			float y1=(m_params.rp[ri.x][ri.y].p.y+0.5)/lv0.height*lvl.height-0.5f;
			float weight=MIN(m_params.lp[li.x][li.y].weight,m_params.rp[ri.x][ri.y].weight);

			float con_x=(x0+x1)/2.0f;
			float con_y=(y0+y1)/2.0f;			
			float vx=(x1-x0)/2.0f;
			float vy=(y1-y0)/2.0f;

			for(int y=floor(con_y);y<=ceil(con_y);y++)
				for(int x=floor(con_x);x<=ceil(con_x);x++)
				{
					if(x >=0 && x < lvl.width && y >= 0 && y < lvl.height)
					{
						int idx = mem_index(lvl, make_int2(x,y))+z*lvl.pagestride;
						float bilinear_w = (1 - fabs(y-con_y))*(1 - fabs(x-con_x))*weight;
						ui_axy[idx] += bilinear_w;
						ui_b[idx] += 2*bilinear_w*(v[idx] - make_float2(vx,vy));
					}
				}			
		}
	}
   
    lvl.ui.axy.copy_from_host(ui_axy);
    lvl.ui.b.copy_from_host(ui_b);  
	
}

void Morph::clear_level(PyramidLevel &lvl)
{
	
	lvl.ssim.cross.clear();
	lvl.ssim.luma.clear();
	lvl.ssim.mean.clear();
	lvl.ssim.var.clear();
	lvl.ssim.value.clear();
	lvl.ssim.counter.clear();

	lvl.tps.axy.clear();
	lvl.tps.b.clear();

	lvl.ui.axy.clear();
	lvl.ui.b.clear();

	lvl.temp.ref.clear();
	lvl.temp.mask.clear();
		
	lvl.improving_mask.clear();

	
}

#define next2power(x) (int)ceil(log((float)x)/log(2.0f))


void Morph::cpu_optimize_level(PyramidLevel &lvl,PyramidLevel &lv0)
{
	
	int w=lvl.width;
	int h=lvl.height;
	int d=lvl.depth;
	int factor=lv0.factor_d/lvl.factor_d;
	int num=w*h;	
	float2 *v=new float2[lvl.pagestride*d];
	lvl.v.resize(lvl.pagestride*d);
	lvl.v.fill(0);
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
		int conz=min(z*factor,lv0.depth-1);
		for(int k=0;k<m_params.cnt.size();k++)
		for(int l=0;l<m_params.cnt[k].size();l++)
		{
			int2 li=m_params.cnt[k][l].li;
			int2 ri=m_params.cnt[k][l].ri;
			if(conz!=m_params.lp[li.x][li.y].p.z)
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

__constant__ KernParameters c_params;

const int OPT_BW = 32,
          OPT_BH = 8,
          OPT_NB = 5;

const int SPACING = 5;

template <int BW, int BH>
struct SSIMData
{
    float2 mean[BH*2+4][BW*2+4],
           var[BH*2+4][BW*2+4];
    float  cross[BH*2+4][BW*2+4],
           value[BH*2+4][BW*2+4];

    int2 orig;
};

template <class T>
__device__ void swap(T &a, T &b)/*{{{*/
{
    T temp = a;
    a = b;
    b = temp;
}/*}}}*/

// returns -1 if pixel cannot improve due to neighbors (and itself) 
// not improving
__device__ int get_improve_mask_idx(const KernPyramidLevel &lvl, /*{{{*/
                            const int2 &p,const int &offt)
{
    int2 block = p/5;
    int2 offset = p%5;

    int begi = (offset.y >= 2 ? 1 : 0),
        begj = (offset.x >= 2 ? 1 : 0),
        endi = begi+2,
        endj = begj+2;

    int impmask_idx = offt*lvl.impmask_pagestride+(block.y+1)*lvl.impmask_rowstride + (block.x+1);

    for(int i=begi; i<endi; ++i)
    {
        for(int j=begj; j<endj; ++j)
        {
            int d = impmask_idx + c_improvmask_offset[i][j];

            if(lvl.improving_mask[d]&c_improvmask[offset.y][offset.x][i][j])
                return impmask_idx;
        }
    }

    return -1;
}/*}}}*/

__device__ bool pixel_on_border(const KernPyramidLevel &lvl, const int2 &p)/*{{{*/
{
    switch(c_params.bcond)
    {
    case BCOND_NONE:
        break;
    case BCOND_CORNER:
        if(p.x==0 && p.y==0 || p.x==0 && p.y==lvl.pixdim.y-1 ||
           p.x==lvl.pixdim.x-1 && p.y==0 && p.x==lvl.pixdim.x-1 && p.y==lvl.pixdim.y-1)
        {
            return true;
        }
        break;
    case BCOND_BORDER:
        if(p.x==0 || p.y==0 || p.x==lvl.pixdim.x-1 || p.y==lvl.pixdim.y-1)
            return true;
        break;
    }
    return false;
}/*}}}*/

// gradient calculation --------------------------

template <int BW, int BH>
__device__ float ssim_change(const KernPyramidLevel &lvl,/*{{{*/
                            const int2 &p,
							const int &offt,
                            float2 v, float2 old_luma, 
                            const SSIMData<BW,BH> &ssimdata)
{
    float2 luma;

    luma.x = tex2D(tex_img0, p.x-v.x + 0.5f, p.y-v.y + 0.5f),
    luma.y = tex2D(tex_img1, p.x+v.x + 0.5f, p.y+v.y + 0.5f);

    float change = 0;

    float2 dmean = luma - old_luma,
           dvar = pow2(luma) - pow2(old_luma);
    float  dcross = luma.x*luma.y - old_luma.x*old_luma.y;

    bool need_counter = p.x < 4 || p.x >= lvl.pixdim.x-4 ||
                        p.y < 4 || p.y >= lvl.pixdim.y-4;

    int idx = mem_index(lvl, p)+offt*lvl.pagestride;
    int2 B = calc_border(p, lvl.pixdim);

    for(int i=0; i<5; ++i)
    {
        int sy = p.y+i-2 - ssimdata.orig.y;
        assert(sy >= 0 && sy < OPT_BH*2+4);
        for(int j=0; j<5; ++j)
        {
            if(c_iomask[B.y][B.x][i][j] == 0)
                continue;

            int sx = p.x+j-2 - ssimdata.orig.x;

            int nb = mem_index(lvl, p + make_int2(j,i)-2)+lvl.pagestride*offt;

            float2 mean, var;
            float counter = need_counter ? lvl.ssim.counter[nb] : 25,
                  cross;

            assert(sx >= 0 && sx < OPT_BW*2+4);

            mean = ssimdata.mean[sy][sx];
            var = ssimdata.var[sy][sx];
            cross = ssimdata.cross[sy][sx];

            mean += dmean;
            var +=  dvar;
            cross += dcross;

            float new_ssim = ssim(mean,var,cross,counter,c_params.ssim_clamp);
            change += ssimdata.value[sy][sx] - new_ssim;
        }
    }

    return change;
}/*}}}*/

template <int BW, int BH>
__device__ float energy_change(const bool flag,
							   const KernPyramidLevel &lvl, /*{{{*/
                               const int2 &p,
							   const int &offt,
                               const float2 &v,
                               const float2 &old_luma,
                               const float2 &d,
                               const SSIMData<BW,BH> &ssimdata)
{
    float v_ssim = ssim_change(lvl, p, offt,v+d, old_luma, ssimdata);

    int idx = mem_index(lvl,p)+offt*lvl.pagestride;

    float v_tps = lvl.tps.axy[idx]*(d.x*d.x + d.y*d.y);
    v_tps += lvl.tps.b[idx].x*d.x;
    v_tps += lvl.tps.b[idx].y*d.y;

    float v_ui  = lvl.ui.axy[idx]*(d.x*d.x + d.y*d.y);
    v_ui += lvl.ui.b[idx].x*d.x;
    v_ui += lvl.ui.b[idx].y*d.y;

	float v_temp = 0.0;
	if(flag)
	{		v_temp += abs(v.x+d.x-lvl.temp.ref[idx].x)-abs(v.x-lvl.temp.ref[idx].x);
 			v_temp += abs(v.y+d.y-lvl.temp.ref[idx].y)-abs(v.y-lvl.temp.ref[idx].y);
	}


	return (c_params.w_ui*v_ui + c_params.w_ssim*v_ssim+ c_params.w_temp*v_temp*lvl.temp.mask[idx]*lvl.factor_d)*lvl.inv_wh
                + c_params.w_tps*v_tps;
}/*}}}*/

template <int BW, int BH>
__device__ float2 compute_gradient(const bool flag,
								   const KernPyramidLevel &lvl, /*{{{*/
                                   const int2 &p,
								   const int &offt,
                                   const float2 &v,
                                   const float2 &old_luma,
                                   const SSIMData<BW,BH> &ssimdata)
{
    float2 g;
    g.x = energy_change(flag,lvl,p,offt,v,old_luma,make_float2(c_params.eps,0),ssimdata)-
          energy_change(flag,lvl,p,offt,v,old_luma,make_float2(-c_params.eps,0),ssimdata);
    g.y = energy_change(flag,lvl,p,offt,v,old_luma,make_float2(0,c_params.eps),ssimdata)-
          energy_change(flag,lvl,p,offt,v,old_luma,make_float2(0,-c_params.eps),ssimdata);
    return -g;
}/*}}}*/

// foldover --------------------------------

template <int X, int Y, int SIGN>
__device__ float2 fover_calc_vtx(const KernPyramidLevel &lvl,/*{{{*/
                                 const int2 &p,  const int &offt, float2 v)
{
    const int2 off = make_int2(X,Y);

    if(lvl.contains(p+off))
        v = SIGN*lvl.v[mem_index(lvl,p+off)+offt*lvl.pagestride];

     return v + (p-off);
}/*}}}*/

__device__ void fover_update_isec_min(float2 c, float2 grad,/*{{{*/
                                      float2 e0, float2 e1,
                                      float &t_min)
{
    float2 de = e1-e0,
           dce = c-e0;

    // determinant
    float d  = de.y*grad.x - de.x*grad.y;

    // signals that we don't have an intersection (yet)
    // t = td/d
    float td = -1;

    // u = ud/d
    // e0 + u*(e1-e0) = intersection point
    float ud = grad.x*dce.y - grad.y*dce.x;

    int sign = signbit(d);

    // this is faster than multiplying ud and d by sign
    if(sign)
    {
        ud = -ud;
        d = -d;
    }

    // line by c0 and c1 intersects segment [e0,e1] ?
    if(ud >= 0 && ud <= d) // u >= 0 && u <= 1
    {
        // c0 + t*(c1-c0) = intersection point
        td = de.x*dce.y - de.y*dce.x;
        td *= (-sign*2+1);

        if(td >= 0 && td < t_min*d)
            t_min = td/d;
    }
}/*}}}*/

template <int SIGN>
__device__ void fover_calc_isec_min(const KernPyramidLevel &lvl, /*{{{*/
                                    const int2 &p,  const int &offt,
                                    float2 v, float2 grad, 
                                    float &t_min)
{
    // edge segment, start from upper left (-1,-1), go cw around center
    // pixel testing whether pixel will intersect the edge or not
    float2 e[2] = { fover_calc_vtx<-1,-1,SIGN>(lvl, p, offt, v),
                    fover_calc_vtx< 0,-1,SIGN>(lvl, p, offt, v)};

    float2 efirst = e[0];

    // pixel displacement (c0 -> c1)
    float2 c = p + v;

    fover_update_isec_min(c,grad,e[0],e[1],t_min);

    e[0]  = fover_calc_vtx<1,-1,SIGN>(lvl, p, offt, v);
    fover_update_isec_min(c,grad,e[1],e[0],t_min);

    e[1]  = fover_calc_vtx<1,0,SIGN>(lvl, p, offt, v);
    fover_update_isec_min(c,grad,e[0],e[1],t_min);

    e[0]  = fover_calc_vtx<1,1,SIGN>(lvl, p, offt, v);
    fover_update_isec_min(c,grad,e[1],e[0],t_min);

    e[1]  = fover_calc_vtx<0,1,SIGN>(lvl, p, offt, v);
    fover_update_isec_min(c,grad,e[0],e[1],t_min);

    e[0]  = fover_calc_vtx<-1,1,SIGN>(lvl, p, offt, v);
    fover_update_isec_min(c,grad,e[1],e[0],t_min);

    e[1]  = fover_calc_vtx<-1,0,SIGN>(lvl, p, offt, v);
    fover_update_isec_min(c,grad,e[0],e[1],t_min);

    fover_update_isec_min(c,grad,e[1],efirst,t_min);
}/*}}}*/

__device__ float prevent_foldover(const KernPyramidLevel &lvl,/*{{{*/
                                  const int2 &p, 
								   const int &offt, 
                                  float2 v, float2 grad)
{
    float t_min = 10;

    fover_calc_isec_min<-1>(lvl, p,offt, -v, -grad, t_min);
    fover_calc_isec_min<1>(lvl, p, offt,v, grad, t_min);

    return max(t_min-c_params.eps,0.0f);
}/*}}}*/

template <int BW, int BH>
__device__ void golden_section_search(const bool flag,
									  const KernPyramidLevel &lvl,/*{{{*/
                                      const int2 &p, const int &offt,
                                      float a, float c,
                                      float2 v, float2 grad,
                                      float2 old_luma,
                                      const SSIMData<BW,BH> &ssimdata,
                                      float &fmin, float &tmin)
{
    const float R = 0.618033989f,
                C = 1.0f - R;


    float b = a*R + c*C,  // b between [a,c>
          x = b*R + c*C;  // x between [b,c>

    float fb = energy_change(flag,lvl, p, offt,v, old_luma, grad*b, ssimdata),
          fx = energy_change(flag,lvl, p, offt,v, old_luma, grad*x, ssimdata);

#pragma unroll 4
    while(c - a > c_params.eps)
    {
        if(fx < fb) // bracket is [b,x,c] ?
        {
            // [a,b,c] = [b,x,c]
            a = b;
            b = x;
            x = b*R + c*C; // x between [b,c>
        }
        else // bracket is [a,b,x] ?
        {
            // [a,b,c] = [a,b,x]
            c = x;
            x = b*R + a*C; // x between <a,b]
        }

        float f = energy_change(flag,lvl, p, offt, v, old_luma, grad*x, ssimdata);

        if(fx < fb)
        {
            fb = fx;
            fx = f;
        }
        else
        {
            swap(b,x);
            fx = fb;
            fb = f;
        }
    }

    if(fx < fb)
    {
        tmin = x;
        fmin = fx;
    }
    else
    {
        tmin = b;
        fmin = fb;
    }
}/*}}}*/

// update --------------------------------

template <int BW, int BH>
__device__ void ssim_update(KernPyramidLevel &lvl,/*{{{*/
                            const int2 &p, 
							const int &offt,
                            float2 v, float2 old_luma,
                            SSIMData<BW,BH> &ssimdata)
{
    float2 luma;

    luma.x = tex2D(tex_img0, p.x-v.x + 0.5f, p.y-v.y + 0.5f),
    luma.y = tex2D(tex_img1, p.x+v.x + 0.5f, p.y+v.y + 0.5f);

    int idx = mem_index(lvl,p)+offt*lvl.pagestride;

    lvl.ssim.luma[idx] = luma;

    float2 dmean = luma - old_luma,
           dvar = pow2(luma) - pow2(old_luma);
    float  dcross = luma.x*luma.y - old_luma.x*old_luma.y;

    int2 B = calc_border(p, lvl.pixdim);

    for(int i=0; i<5; ++i)
    {
        int sy = p.y+i-2 - ssimdata.orig.y;
        for(int j=0; j<5; ++j)
        {
            if(c_iomask[B.y][B.x][i][j])
            {
                int sx = p.x+j-2 - ssimdata.orig.x;

                atomicAdd(&ssimdata.mean[sy][sx], dmean);
                atomicAdd(&ssimdata.var[sy][sx], dvar);
                atomicAdd(&ssimdata.cross[sy][sx], dcross);
            }
        }
    }
}/*}}}*/

template <int BW, int BH>
__device__ void commit_pixel_motion(const bool flag,
									KernPyramidLevel &lvl, /*{{{*/
                                    const int2 &p,
									const int&offt,
                                    const float2 &newv,
                                    const float2 &old_luma,
                                    const float2 &grad,
                                    SSIMData<BW,BH> &ssimdata)

{
    ssim_update(lvl, p, offt,newv, old_luma, ssimdata);

    int2 B = calc_border(p, lvl.pixdim);

    // tps update
    for(int i=0; i<5; ++i)
    {
        for(int j=0; j<5; ++j)
        {
            assert(lvl.contains(p.x+j-2,p.y+i-2) || c_tps_data[B.y][B.x][i][j] == 0);

            int nb = mem_index(lvl, p + make_int2(j,i)-2)+offt*lvl.pagestride;
            atomicAdd(&lvl.tps.b[nb], grad*c_tps_data[B.y][B.x][i][j]);
        }
    }

    int idx = mem_index(lvl,p)+offt*lvl.pagestride;

    // ui update
    lvl.ui.b[idx] += 2*grad*lvl.ui.axy[idx];

	//temp update

    // vector update
    lvl.v[idx] = newv;
}/*}}}*/

// optimization kernel --------------------------

template <int BW, int BH>
__device__ bool optimize_pixel(const bool flag,
							   const KernPyramidLevel &lvl,/*{{{*/
                               const int2 &pos,
							   const int& offt,
                               const SSIMData<BW,BH> &ssim,
                               float2 &old_luma,
                               float2 &v,
                               float2 &grad,
                               int &impmask_idx)
{
    if(lvl.contains(pos))
    {
        int idx = mem_index(lvl,pos)+offt*lvl.pagestride;

        v = lvl.v[idx],
        old_luma = lvl.ssim.luma[idx];

        impmask_idx = get_improve_mask_idx(lvl, pos, offt);

		assert(lvl.contains(pos) || lvl.improving_mask[impmask_idx] == 0);

		if(impmask_idx >= 0)
        {
            if(!pixel_on_border(lvl, pos))
            {
                grad = compute_gradient(flag, lvl, pos, offt,v, old_luma, ssim);

            //    float ng = hypot(grad.x,grad.y); // slower
                float ng = sqrt(pow2(grad.x)+pow2(grad.y));

                if(ng != 0)
                {
                    grad /= ng;

                    float t = prevent_foldover(lvl, pos, offt,v, grad);

                    float tmin, fmin;

                    golden_section_search(flag, lvl, pos, offt, 0, t,
                                          v, grad, old_luma, ssim, fmin, tmin);

                    if(fmin < 0)
                    {
                        grad *= tmin;
                        v += grad;
                        return true;
                    }
                }
            }
        }
    }
    return false;
}/*}}}*/

template <template<int,int> class F>
__device__ void process_shared_state(F<8,8> fun, const KernPyramidLevel &lvl,/*{{{*/
                                     const int2 &block_orig)
{
    const int BW = 8, BH = 8;

    /*     BW      BW      4
       -----------------------
       |        |        |   | BH
       |   1    |   2    | 6 |
       |-----------------|---|
       |        |        |   | BH
       |   4    |   3    | 6 |
       |-----------------|---|
       |   5    |   5    | 7 | 4
       -----------------------
    */

    // area 1
    int sx = threadIdx.x,
        sy = threadIdx.y;
    int2 pix = block_orig + make_int2(sx,sy);
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 2
    pix.x += BW;
    sx += BW;
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 3
    pix.y += BH;
    sy += BH;
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 4
    pix.x -= BW;
    sx -= BW;
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 5
    sx = (threadIdx.y/4)*BW + threadIdx.x;
    sy = threadIdx.y%4 + BH*2;
    pix.x = block_orig.x+sx;
    pix.y = block_orig.y+sy;
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 6
    sx = threadIdx.x%4 + BW*2;
    sy = threadIdx.y*(BW/4) + threadIdx.x/4;
    pix.x = block_orig.x+sx;
    pix.y = block_orig.y+sy;
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 7
    sy += BH*2;
    pix.y += BH*2;
    if(lvl.contains(pix) && sy < BH*2+4)
        fun(pix, sx, sy);
}/*}}}*/

template <template<int,int> class F>
__device__ void process_shared_state(F<32,8> fun, const KernPyramidLevel &lvl,/*{{{*/
                                     const int2 &block_orig)
{
    const int BW = 32, BH = 8;

    int sx = threadIdx.x,
        sy = threadIdx.y;

    /*     BW      BW      4
       -----------------------
       |        |        |   | BH
       |   1    |   2    | 6 |
       |-----------------|---|
       |        |        |   | BH
       |   4    |   3    | 6 |
       |-----------------|---|
       |   5    |   5    | 6 | 4
       -----------------------
    */

    // area 1
    int2 pix = block_orig + make_int2(sx,sy);
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 2
    pix.x += BW;
    sx += BW;
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 3

    pix.y += BH;
    sy += BH;
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 4

    pix.x -= BW;
    sx -= BW;
    if(lvl.contains(pix))
        fun(pix, sx, sy);

    // area 5
    sx = (threadIdx.y/4)*BW + threadIdx.x;
    sy = threadIdx.y%4 + BH*2;
    pix.x = block_orig.x+sx;
    pix.y = block_orig.y+sy;
    if(lvl.contains(pix) && sx < BW*2+4 && sy < BH*2+4)
        fun(pix, sx, sy);

    // area 6
    sx = threadIdx.x%4 + BW*2;
    sy = threadIdx.y*8 + threadIdx.x/4;
    pix.x = block_orig.x+sx;
    pix.y = block_orig.y+sy;
    if(lvl.contains(pix) && sx < BW*2+4 && sy < BH*2+4)
        fun(pix, sx, sy);
}/*}}}*/

template <int BW, int BH>
class LoadSSIM/*{{{*/
{
public:
    __device__ LoadSSIM(const KernPyramidLevel &lvl, SSIMData<BW,BH> &ssim,int offt)
        : m_level(lvl), m_ssim(ssim),m_offt(offt) {}

    __device__ void operator()(const int2 &pix, int sx, int sy)
    {
        int idx = mem_index(m_level, pix)+m_offt*m_level.pagestride;
        m_ssim.mean[sy][sx] = m_level.ssim.mean[idx];
        m_ssim.var[sy][sx] = m_level.ssim.var[idx];
        m_ssim.cross[sy][sx] = m_level.ssim.cross[idx];
        m_ssim.value[sy][sx] = m_level.ssim.value[idx];
    }

private:
    const KernPyramidLevel &m_level;
    SSIMData<BW,BH> &m_ssim;
	int m_offt;
};/*}}}*/

template <int BW, int BH>
class SaveSSIM/*{{{*/
{
public:
    __device__ SaveSSIM(KernPyramidLevel &lvl, const SSIMData<BW,BH> &ssim,int offt)
        : m_level(lvl), m_ssim(ssim),m_offt(offt) {}

    __device__ void operator()(const int2 &pix, int sx, int sy)
    {
        int idx = mem_index(m_level, pix)+m_offt*m_level.pagestride;
        m_level.ssim.mean[idx] = m_ssim.mean[sy][sx];
        m_level.ssim.var[idx] = m_ssim.var[sy][sx];
        m_level.ssim.cross[idx] = m_ssim.cross[sy][sx];
        m_level.ssim.value[idx] = m_ssim.value[sy][sx];
    }

private:
    KernPyramidLevel &m_level;
    const SSIMData<BW,BH> &m_ssim;
	int m_offt;
};/*}}}*/

template <int BW, int BH>
class UpdateSSIM/*{{{*/
{
public:
    __device__ UpdateSSIM(const KernPyramidLevel &lvl, SSIMData<BW,BH> &ssim,int offt)
        : m_level(lvl), m_ssim(ssim),m_offt(offt) {}

    __device__ void operator()(const int2 &pix, int sx, int sy)
    {
        int idx = mem_index(m_level, pix)+m_offt*m_level.pagestride;
        m_ssim.value[sy][sx] = ssim(m_ssim.mean[sy][sx],
                                    m_ssim.var[sy][sx],
                                    m_ssim.cross[sy][sx],
                                    m_level.ssim.counter[idx],
                                    c_params.ssim_clamp);
    }

private:
    const KernPyramidLevel &m_level;
	int m_offt;
    SSIMData<BW,BH> &m_ssim;
};/*}}}*/

__global__
__launch_bounds__(OPT_BW*OPT_BH, OPT_NB)
void kernel_optimize_level(const bool flag, KernPyramidLevel lvl,/*{{{*/
                           int offx, int offy,int offt,
                           bool *out_improving)
{

    __shared__ SSIMData<OPT_BW,OPT_BH> ssim;

    {
        int2 block_orig = make_int2(blockIdx.x*(OPT_BW*2+SPACING)+offx-2,
                                    blockIdx.y*(OPT_BH*2+SPACING)+offy-2);

        if(threadIdx.x == 0 && threadIdx.y == 0)
            ssim.orig = block_orig;

        process_shared_state(LoadSSIM<OPT_BW,OPT_BH>(lvl, ssim, offt), lvl, block_orig);
    }

    bool improving = false;

    __syncthreads();

    for(int i=0; i<2; ++i)
    {
        for(int j=0; j<2; ++j)
        {
            int2 p = ssim.orig + make_int2(threadIdx.x*2+j+2,
                                           threadIdx.y*2+i+2);

            float2 old_luma, v, grad;
            int impmask_idx = -1;
            bool ok = optimize_pixel(flag,lvl, p, offt, ssim, old_luma, v, grad, 
                                     impmask_idx);


            int2 offset = p%5;
            __syncthreads();

            if(ok)
            {
                commit_pixel_motion(flag, lvl, p, offt,v, old_luma, grad, ssim);

                improving = true;
                atomicOr(&lvl.improving_mask[impmask_idx], 
                         1 << (offset.x + offset.y*5));
            }
            else if(impmask_idx >= 0)
            {
                atomicAnd(&lvl.improving_mask[impmask_idx], 
                          ~(1 << (offset.x + offset.y*5)));
            }
            __syncthreads();

            process_shared_state(UpdateSSIM<OPT_BW,OPT_BH>(lvl, ssim, offt), lvl, ssim.orig);

            __syncthreads();
        }
    }

    process_shared_state(SaveSSIM<OPT_BW,OPT_BH>(lvl, ssim,offt), lvl, ssim.orig);

    if(improving)
        *out_improving = true;
}/*}}}*/

template <class T>
T *addressof(T &v)
{
    return reinterpret_cast<T*>(&const_cast<char &>(reinterpret_cast<const volatile char &>(v)));
}

void Morph::optimize_level(PyramidLevel &lvl)
{      
    KernPyramidLevel klvl(lvl);
    KernParameters kparams(m_params);
	
    rod::copy_to_symbol(c_params,kparams);

    bool *improving = NULL;
    cudaHostAlloc(&improving, sizeof(bool), cudaHostAllocMapped);
    rod::check_cuda_error("cudaHostAlloc");
    assert(improving != NULL);

    bool *dimproving = NULL;
    cudaHostGetDevicePointer(&dimproving, improving, 0);
    rod::check_cuda_error("cudaHostGetDevicePointer");

	//middle frame
	
	dim3 bdim(OPT_BW,OPT_BH),
         gdim((lvl.width+OPT_BW*2+SPACING-1)/(OPT_BW*2+SPACING),
              (lvl.height+OPT_BH*2+SPACING-1)/(OPT_BH*2+SPACING));
	int i=lvl.depth/2;
	cudaBindTextureToArray(tex_img0, lvl.img0[i]);
	cudaBindTextureToArray(tex_img1, lvl.img1[i]);		 	
	int iter = 0;
	do
	{
		*improving = false;
		
		kernel_optimize_level<<<gdim,bdim>>>(false,klvl, 0,0, i,dimproving);
		kernel_optimize_level<<<gdim,bdim>>>(false,klvl, OPT_BW*2, 0,i, dimproving);
		kernel_optimize_level<<<gdim,bdim>>>(false,klvl, 0, OPT_BH*2, i,dimproving);
		kernel_optimize_level<<<gdim,bdim>>>(false,klvl, OPT_BW*2, OPT_BH*2,i, dimproving);
		
		cudaDeviceSynchronize();
		iter++;
		_current_iter+=lvl.width*lvl.height;
	}while(iter <_max_iter&&m_cb&&*improving);
	_current_iter=_current_iter-(lvl.width*lvl.height)*iter+(lvl.width*lvl.height)*_max_iter;
 	

	for(int i=lvl.depth/2+1;i<lvl.depth;i++)
				{
					if(!m_cb)	
						break;		
					
					initialize_temp(lvl,i,-1);
			
					cudaBindTextureToArray(tex_img0, lvl.img0[i]);
					cudaBindTextureToArray(tex_img1, lvl.img1[i]);		 
					int iter = 0;
					do
					{		
 						*improving = false;
						kernel_optimize_level<<<gdim,bdim>>>(true,klvl, 0,0, i,dimproving);
						kernel_optimize_level<<<gdim,bdim>>>(true,klvl, OPT_BW*2, 0,i, dimproving);
						kernel_optimize_level<<<gdim,bdim>>>(true,klvl, 0, OPT_BH*2, i,dimproving);
						kernel_optimize_level<<<gdim,bdim>>>(true,klvl, OPT_BW*2, OPT_BH*2,i, dimproving);
						cudaDeviceSynchronize();
						iter++;
						_current_iter+=lvl.width*lvl.height;				
					}while(iter <_max_iter&&m_cb&&*improving);
					_current_iter=_current_iter-(lvl.width*lvl.height)*iter+(lvl.width*lvl.height)*_max_iter;
		 		}
		for(int i=lvl.depth/2-1;i>=0;i--)
				{
					if(!m_cb)	
						break;		
					
					initialize_temp(lvl,i,1);
			
					cudaBindTextureToArray(tex_img0, lvl.img0[i]);
					cudaBindTextureToArray(tex_img1, lvl.img1[i]);		 
					int iter = 0;
					do
					{		
 						*improving = false;
						kernel_optimize_level<<<gdim,bdim>>>(true,klvl, 0,0, i,dimproving);
						kernel_optimize_level<<<gdim,bdim>>>(true,klvl, OPT_BW*2, 0,i, dimproving);
						kernel_optimize_level<<<gdim,bdim>>>(true,klvl, 0, OPT_BH*2, i,dimproving);
						kernel_optimize_level<<<gdim,bdim>>>(true,klvl, OPT_BW*2, OPT_BH*2,i, dimproving);
						cudaDeviceSynchronize();
						iter++;
						_current_iter+=lvl.width*lvl.height;				
					}while(iter <_max_iter&&m_cb&&*improving);
					_current_iter=_current_iter-(lvl.width*lvl.height)*iter+(lvl.width*lvl.height)*_max_iter;
		 		}
	  cudaFreeHost(improving);  
 }
