#include <cuda.h>
#include <util/dimage.h>
#include <util/timer.h>
#include <util/image_ops.h>
#include "pyramid.h"

extern texture<float2, 2, cudaReadModeElementType> tex_f0, tex_f1;

__global__ void conv_to_block_of_arrays(float2 *v,
                                        rod::dimage_ptr<const float2> in,
                                        KernPyramidLevel lvl,
                                        float2 m)
{
    int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

    if(!in.is_inside(pos))
        return;

    in += in.offset_at(pos);
    v += mem_index(lvl, pos);

    *v = *in * m;
}

__global__ void temp_ref(float2 *v_prev,float2 *v_cur,float* weight,float* ssim,KernPyramidLevel lvl)
{
    int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

    if(!lvl.contains(pos))
        return;
	
	float2 p=make_float2(pos.x,pos.y);
	float2 v=v_prev[mem_index(lvl, pos)];
	float2 f0=tex2D(tex_f0, p.x -v.x + 0.5f, p.y - v.y + 0.5f);
	float2 f1=tex2D(tex_f1, p.x +v.x + 0.5f, p.y + v.y + 0.5f);    

	float2 p_ref=p+0.5*(f0+f1);
	float2 v_ref=v+0.5*(f1-f0);

	int xx=floor(p_ref.x);
	int yy=floor(p_ref.y);

	for(int y=yy;y<=yy+1;y++)
		for(int x=xx;x<=xx+1;x++)
		{
			if(!lvl.contains(make_int2(x,y)))
				continue;
			float ssim_fa=1;
			if(ssim)
				ssim_fa=ssim[mem_index(lvl, make_int2(pos.x,pos.y))];
			float fa=ssim_fa*(1.0-abs((float)x-p_ref.x))*(1.0-abs((float)y-p_ref.y));
			atomicAdd(v_cur+mem_index(lvl, make_int2(x,y)),v_ref*fa);
			atomicAdd(weight+mem_index(lvl, make_int2(x,y)),fa);			
		}

}

__global__ void interpolate_temp_ref(float2 *v_cur,float* weight,KernPyramidLevel lvl)
{
    int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

    if(!lvl.contains(pos))
        return;
	int idx=mem_index(lvl, pos);
	if(weight[idx]>0)	
		v_cur[idx]/=weight[idx];
	
}


__global__ void smooth(float2 *v_out, float2 *v_cur,float* weight,KernPyramidLevel lvl)
{
    int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

    if(!lvl.contains(pos))
        return;
	
	float ww=0.0;
	float2 v=make_float2(0,0);
	int r=1;
	for(int y=pos.y-r;y<=pos.y+r;y++)
		for(int x=pos.x-r;x<=pos.x+r;x++)
		{
			if(!lvl.contains(make_int2(x,y)))
				continue;
			int idx=mem_index(lvl,make_int2(x,y));
			if(weight[idx]>0)
				{
					ww+=1;
					v+=v_cur[idx];
				}
		}

	if(ww>0)	
		v_out[mem_index(lvl,pos)]=v/ww;
	
	
	
}



__global__ void fill_zeros_x(float2 *v_out, float* weight,KernPyramidLevel lvl)
{
    int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

    if(!lvl.contains(pos))
        return;
	int idx=mem_index(lvl, pos);
	
	float ww=0.0;
	float2 v=make_float2(0,0);
	
	if(weight[idx]>0)
		return;

	for(int x=pos.x;x>=0;x--)
		if(weight[mem_index(lvl,make_int2(x,pos.y))]>0)
		{
			ww+=1.0/(pos.x-x);
			v+=v_out[mem_index(lvl,make_int2(x,pos.y))];
			break;
		}
	for(int x=pos.x;x<lvl.pixdim.x;x++)
		if(weight[mem_index(lvl,make_int2(x,pos.y))]>0)
		{
			ww+=1.0/(x-pos.x);
			v+=v_out[mem_index(lvl,make_int2(x,pos.y))];
			break;
		}

	if(ww>0)	
		v_out[idx]=v/ww;
		
	
}

__global__ void fill_zeros_y(float2 *v_out, float* weight,KernPyramidLevel lvl)
{
    int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

    if(!lvl.contains(pos))
        return;
	int idx=mem_index(lvl, pos);
	
	float ww=0.0;
	float2 v=make_float2(0,0);
	
	if(weight[idx]>0)
		return;

	for(int y=pos.y;y>=0;y--)
		if(weight[mem_index(lvl,make_int2(pos.x,y))]>0)
		{
			ww+=1.0/(pos.y-y);
			v+=v_out[mem_index(lvl,make_int2(pos.x,y))];
			break;
		}
	for(int y=pos.y;y<lvl.pixdim.y;y++)
		if(weight[mem_index(lvl,make_int2(pos.x,y))]>0)
		{
			ww+=1.0/(y-pos.y);
			v+=v_out[mem_index(lvl,make_int2(pos.x,y))];
			break;
		}

	if(ww>0)	
		weight[idx]=1;
	
	
}
__global__ void kernel_initialize_temp(KernPyramidLevel lvl,int page,float2* v_ref, float* weight)
{
	int tx = threadIdx.x, ty = threadIdx.y,
		  bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

    if(!lvl.contains(pos))
        return;

	 int idx = mem_index(lvl, pos)+lvl.pagestride*page;


	if(weight[idx-page*lvl.pagestride]>0)
	{
		lvl.temp.ref[idx]=v_ref[idx-page*lvl.pagestride];	
		lvl.temp.mask[idx]=weight[idx-page*lvl.pagestride];
	}
	else
		lvl.temp.mask[idx]=0.0;
		
}


void initialize_temp(PyramidLevel &lvl,int i,int dir)
{
	dim3 bdim(32,8),
			gdim((lvl.width+bdim.x-1)/bdim.x,
                    (lvl.height+bdim.y-1)/bdim.y);
		
	rod::dvector<float> weight;
	weight.resize(lvl.pagestride);
	weight.fill(0);
	rod::dvector<float2> ref_v;
	ref_v.resize(lvl.pagestride);
	ref_v.fill(0);
	
	tex_f0.normalized = false;
    tex_f0.filterMode = cudaFilterModeLinear;
    tex_f0.addressMode[0] = tex_f0.addressMode[1] = cudaAddressModeClamp;

    tex_f1.normalized = false;
    tex_f1.filterMode = cudaFilterModeLinear;
    tex_f1.addressMode[0] = tex_f1.addressMode[1] = cudaAddressModeClamp;

	if(dir<0)
	{
		cudaBindTextureToArray(tex_f0, lvl.f0[i+dir]);
		cudaBindTextureToArray(tex_f1, lvl.f1[i+dir]);			
	}
	else
	{
		cudaBindTextureToArray(tex_f0, lvl.b0[i+dir]);
		 cudaBindTextureToArray(tex_f1, lvl.b1[i+dir]);			
	}
   	temp_ref<<<gdim, bdim>>>(lvl.v.data()+(i+dir)*lvl.pagestride,ref_v,weight,lvl.ssim.value.data()+(i+dir)*lvl.pagestride,lvl);


			
   	interpolate_temp_ref<<<gdim, bdim>>>(ref_v,weight,lvl);   	
	


 	kernel_initialize_temp<<<gdim, bdim>>>(lvl,i,ref_v,weight);


	weight.clear();	
	ref_v.clear();
}

void upsample(PyramidLevel &dest, PyramidLevel &orig)
{
   dest.v.resize(dest.pagestride*dest.depth);
	dest.v.fill(0);
	
    dim3 bdim(32,8),
			gdim((dest.width+bdim.x-1)/bdim.x,
                    (dest.height+bdim.y-1)/bdim.y);
	int factor=1;
	if(dest.depth>orig.depth)
		factor=2;

    for(int i=0;i<orig.depth;i++)
    {
        rod::dimage<float2> vec_orig;
        internal_vector_to_image(vec_orig, orig.v.data()+i*orig.pagestride, orig, 
                             make_float2(1,1));			

        rod::dimage<float2> vec_dest(dest.width, dest.height);

        rod::upsample(&vec_dest, &vec_orig, rod::INTERP_LINEAR);       
		
        conv_to_block_of_arrays<<<gdim, bdim>>>(dest.v.data()+min((i*factor),dest.depth-1)*dest.pagestride, &vec_dest, dest,
                      make_float2((float)dest.width/orig.width,
                                  (float)dest.height/orig.height));			
    }
	//orig.v.clear();


	tex_f0.normalized = false;
    tex_f0.filterMode = cudaFilterModeLinear;
    tex_f0.addressMode[0] = tex_f0.addressMode[1] = cudaAddressModeClamp;

    tex_f1.normalized = false;
    tex_f1.filterMode = cudaFilterModeLinear;
    tex_f1.addressMode[0] = tex_f1.addressMode[1] = cudaAddressModeClamp;

	if(factor>1)//upsample based on optical flow
   	{
   		 for(int i=1;i<dest.depth;i+=factor)
   		{
   			
   			if(i==dest.depth-1)
   				continue;

			rod::dvector<float> weight;
			weight.resize(dest.pagestride);
			weight.fill(0);
						
		
   			cudaBindTextureToArray(tex_f0, dest.f0[i-1]);
    		cudaBindTextureToArray(tex_f1, dest.f1[i-1]);			
   			temp_ref<<<gdim, bdim>>>(dest.v.data()+(i-1)*dest.pagestride,dest.v.data()+i*dest.pagestride,weight,NULL,dest);
   			//cudaDeviceSynchronize();
			
					
   			cudaBindTextureToArray(tex_f0, dest.b0[i+1]);
    		cudaBindTextureToArray(tex_f1, dest.b1[i+1]);			
   			temp_ref<<<gdim, bdim>>>(dest.v.data()+(i+1)*dest.pagestride,dest.v.data()+i*dest.pagestride,weight,NULL,dest);
					
			
			interpolate_temp_ref<<<gdim, bdim>>>(dest.v.data()+i*dest.pagestride,weight,dest);   	
		
			
			rod::dvector<float2> v;
			v.resize(dest.pagestride);
			v.fill(0);
 			smooth<<<gdim, bdim>>>(v.data(),dest.v.data()+i*dest.pagestride,weight,dest); 	
			fill_zeros_x<<<gdim, bdim>>>(v.data(),weight,dest);			
			fill_zeros_y<<<gdim, bdim>>>(v.data(),weight,dest);
	

			cudaMemcpy(dest.v.data()+i*dest.pagestride,v.data(),dest.pagestride*sizeof(float2),cudaMemcpyDeviceToDevice);
			
			weight.clear();
			v.clear();		
		
   		}
	}
     
}

 
texture<float, 2, cudaReadModeElementType> tex_val;

__global__ void Kernel_upsample(float* dest, int w, int h,float ratio)
{
	int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

	float2 p=make_float2((pos.x+0.5)/(float)w,(pos.y+0.5)/(float)h);
	
	dest+= pos.y*w+pos.x;
	*dest=tex2D(tex_val,p.x,p.y)*ratio;
}

void upsample(float *dest, int dw,int dh, float *orig, int sw,int sh,float ratio)
{
	dim3 bdim(32,8),
			gdim((dw+bdim.x-1)/bdim.x,
                    (dh+bdim.y-1)/bdim.y);

		cudaArray* a_val;
	cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float>();
	cudaMallocArray(&a_val, &ccd, sw, sh);	
 	cudaMemcpy2DToArray(a_val, 0, 0, orig, sw*sizeof(float), sw*sizeof(float), sh,cudaMemcpyDeviceToDevice); 

	tex_val.normalized = true;
    tex_val.filterMode = cudaFilterModeLinear;
    tex_val.addressMode[0] = tex_val.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(tex_val,a_val);

	Kernel_upsample<<<gdim,bdim>>>(dest, dw, dh,ratio);	

	cudaFreeArray(a_val);
}