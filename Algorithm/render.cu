#include <cuda.h>
#include <util/dimage.h>
#include <util/timer.h>
#include <util/dmath.h>

#define MAX_FRAME 200

                                          
texture<float2, 2, cudaReadModeElementType> tex_vector,tex_qpath;
texture<float4, 2, cudaReadModeElementType>tex_vector_3d;
texture<float4, 2, cudaReadModeElementType> tex_ext0, tex_ext1;
texture<float2, cudaTextureType2DLayered, cudaReadModeElementType> tex_forw0,tex_forw1;
texture<float4, cudaTextureType2DLayered, cudaReadModeElementType> tex_video0,tex_video1;


__global__ void kernel_render_halfway_image(uchar3* out, int rowstride, int width, int height, int ex,
                                            float color_fa,float geo_fa,int color_from)
{
    int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

    if(pos.x >= width|| pos.y >= height)
        return;

	float2 p,q,v,u;
	p=q=make_float2(pos.x,pos.y);
	v=tex2D(tex_vector,p.x+0.5f, p.y+0.5f);
	u=tex2D(tex_qpath, p.x+0.5f, p.y+0.5f);
	float alpha=0.8;
	for(int i=0;i<20;i++)
 		{	 			
 			p=q-(2*geo_fa-1)*v-(4*geo_fa-4*geo_fa*geo_fa)*u;
 			//p=q-(2*t_geo-1)*v;		 		
 			v=alpha*tex2D(tex_vector, p.x+0.5f, p.y+0.5f)+(1-alpha)*v;
 			u=alpha*tex2D(tex_qpath, p.x+0.5f, p.y+0.5f)+(1-alpha)*u;				
 		}
	
    float4 c0,c1;   
    c0 = tex2D(tex_ext0, p.x-v.x+ex+0.5f, p.y-v.y+ex+0.5f);    
    c1 = tex2D(tex_ext1, p.x+v.x+ex+0.5f, p.y+v.y+ex+0.5f);
       
	out += pos.y*rowstride+pos.x;	
	switch (color_from)
	{

	case 0:
		*out=make_uchar3(c0.x+0.5,c0.y+0.5,c0.z+0.5);
		break;
	case 1:
		*out=make_uchar3(c0.x*(1-color_fa)+c1.x*color_fa+0.5,c0.y*(1-color_fa)+c1.y*color_fa+0.5,c0.z*(1-color_fa)+c1.z*color_fa+0.5);		
		break;
	case 2:
		*out=make_uchar3(c1.x+0.5,c1.y+0.5,c1.z+0.5);
		break;
	}
	
   
}

void render_halfway_image(rod::dvector<uchar3> &out, int rowstride, int width, int height, int ex,
						  float color_fa, float geo_fa,int color_from,
                          const cudaArray *img0, 
						  const cudaArray *img1,
                          const cudaArray *vector, 
						  const cudaArray *qpath)
{
    tex_ext0.normalized = false;
    tex_ext0.filterMode = cudaFilterModeLinear;
    tex_ext0.addressMode[0] = tex_ext0.addressMode[1] = cudaAddressModeClamp;

    tex_ext1.normalized = false;
    tex_ext1.filterMode = cudaFilterModeLinear;
    tex_ext1.addressMode[0] = tex_ext1.addressMode[1] = cudaAddressModeClamp;

	tex_vector.normalized = false;
    tex_vector.filterMode = cudaFilterModeLinear;
    tex_vector.addressMode[0] = tex_vector.addressMode[1] = cudaAddressModeClamp;
	
	tex_qpath.normalized = false;
    tex_qpath.filterMode = cudaFilterModeLinear;
    tex_qpath.addressMode[0] = tex_qpath.addressMode[1] = cudaAddressModeClamp;
   

    cudaBindTextureToArray(tex_ext0, img0);
    cudaBindTextureToArray(tex_ext1, img1);
	cudaBindTextureToArray(tex_vector, vector);
    cudaBindTextureToArray(tex_qpath, qpath);

	 dim3 bdim(32,8),
     gdim((width+bdim.x-1)/bdim.x,
              (height+bdim.y-1)/bdim.y);

    kernel_render_halfway_image<<<gdim, bdim>>>(out, rowstride,width,height, ex, color_fa,geo_fa,color_from);
}


__global__ void kernel_render_resample_image0(uchar3* out, int rowstride, int width, int height, int depth,float fa,int frame)
{
	 int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

     if(pos.x >= width|| pos.y >= height)
        return;
	
	float4 c;   
    float4 p,q,v;		
	p=q=make_float4(pos.x,pos.y,frame,0);
	v=tex2D(tex_vector_3d,p.x+0.5f, p.y+0.5f);
	v.z=0;
	float alpha=0.5;
	for(int i=0;i<50;i++)
	{
		p=q+v;	
		v=alpha*tex2D(tex_vector_3d, p.x+0.5f, p.y+0.5f)+(1-alpha)*v;
		v.z=0;
	}	
	v=tex2D(tex_vector_3d, p.x+0.5f, p.y+0.5f);
	q.z-=v.z;
		
	if(q.z<=0)
		c=tex2DLayered(tex_video0,q.x+0.5,q.y+0.5,0);
	else if (q.z>=depth-1)
		c=tex2DLayered(tex_video0,q.x+0.5,q.y+0.5,depth-1);
	else 
	{		
		p=q;
		p.z=floor(q.z);
		float fa_z=q.z-p.z;
		float2 f=tex2DLayered(tex_forw0,p.x+0.5,p.y+0.5,(int)(p.z+0.5));
		for(int i=0;i<50;i++)
 			{
 				p=q-make_float4(f.x,f.y,1,0)*fa_z;	
 				f=alpha*tex2DLayered(tex_forw0,p.x+0.5,p.y+0.5,(int)(p.z+0.5))+(1-alpha)*f;
 			}
		
			
		c=tex2DLayered(tex_video0,p.x+0.5,p.y+0.5,(int)(p.z+0.5))*(1-fa_z)+tex2DLayered(tex_video0,p.x+0.5+f.x,p.y+0.5+f.y, (int)(p.z+0.5+1))*fa_z;
	}
	
	out += pos.y*rowstride+pos.x;
	*out=make_uchar3((*out).x+(c.x+0.5)*(1-fa),(*out).y+(c.y+0.5)*(1-fa),(*out).z+(c.z+0.5)*(1-fa));
		
}
 
__global__ void kernel_render_resample_image1(uchar3* out, int rowstride, int width, int height, int depth,float fa,int frame)
{
	 int tx = threadIdx.x, ty = threadIdx.y,
        bx = blockIdx.x, by = blockIdx.y;

    int2 pos = make_int2(bx*blockDim.x + tx, by*blockDim.y + ty);

    if(pos.x >= width || pos.y >= height)
        return;
	
	float4 c;      
	
	float4 p,q,v;		
	p=q=make_float4(pos.x,pos.y,frame,0);
	v=tex2D(tex_vector_3d,p.x+0.5f, p.y+0.5f);
	v.z=0;
	float alpha=0.5;
	for(int i=0;i<50;i++)
	{
		p=q-v;	
		v=alpha*tex2D(tex_vector_3d, p.x+0.5f, p.y+0.5f)+(1-alpha)*v;
		v.z=0;
	}	
	v=tex2D(tex_vector_3d, p.x+0.5f, p.y+0.5f);
	q.z+=v.z;
	
	if(q.z<=0)
		c=tex2DLayered(tex_video1,q.x+0.5,q.y+0.5,0);
	else if (q.z>=depth-1)
		c=tex2DLayered(tex_video1,q.x+0.5,q.y+0.5,depth-1);
	else 
	{		
		p=q;
		p.z=floor(q.z);
		float fa_z=q.z-p.z;
		float2 f=tex2DLayered(tex_forw1,p.x+0.5,p.y+0.5,(int)(p.z+0.5));
		
 		for(int i=0;i<50;i++)
 			{
 				p=q-make_float4(f.x,f.y,1,0)*fa_z;	
 				f=alpha*tex2DLayered(tex_forw1,p.x+0.5,p.y+0.5,(int)(p.z+0.5))+(1-alpha)*f;
 			}
		
		c=tex2DLayered(tex_video1,p.x+0.5,p.y+0.5,(int)(p.z+0.5))*(1-fa_z)+tex2DLayered(tex_video1,p.x+0.5+f.x,p.y+0.5+f.y,(int)(p.z+0.5+1))*fa_z;
	}	
	
	out += pos.y*rowstride+pos.x;	
	*out=make_uchar3((*out).x+(c.x+0.5)*fa,(*out).y+(c.y+0.5)*fa,(*out).z+(c.z+0.5)*fa);
}
 


void render_resample_image(rod::dvector<uchar3> &out, int rowstride, int width, int height, int depth,
						   float fa, int frame,
						   const cudaArray* img0, 
						  const cudaArray* img1,
						   const cudaArray *vector, 
						   const cudaArray* f0,
						   const cudaArray* f1)

{
	

	tex_vector_3d.normalized = false;
    tex_vector_3d.filterMode = cudaFilterModeLinear;
    tex_vector_3d.addressMode[0] = tex_vector_3d.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(tex_vector_3d, vector);

	tex_video0.normalized = false;
    tex_video0.filterMode = cudaFilterModeLinear;
    tex_video0.addressMode[0] = tex_video0.addressMode[1] = tex_video0.addressMode[2]= cudaAddressModeClamp;
	cudaBindTextureToArray(tex_video0, img0);

	tex_video1.normalized = false;
    tex_video1.filterMode = cudaFilterModeLinear;
    tex_video1.addressMode[0] = tex_video1.addressMode[1] = tex_video0.addressMode[2]=cudaAddressModeClamp;
	cudaBindTextureToArray(tex_video1, img1);

	tex_forw0.normalized = false;
    tex_forw0.filterMode = cudaFilterModeLinear;
    tex_forw0.addressMode[0] = tex_forw0.addressMode[1] = tex_forw0.addressMode[2]=cudaAddressModeClamp;
	cudaBindTextureToArray(tex_forw0, f0);

	tex_forw1.normalized = false;
    tex_forw1.filterMode = cudaFilterModeLinear;
    tex_forw1.addressMode[1] = tex_forw1.addressMode[1] = tex_forw1.addressMode[2]=cudaAddressModeClamp;
	cudaBindTextureToArray(tex_forw1, f1);

	out.fill(0);

	dim3 bdim(32,8),
         gdim((width+bdim.x-1)/bdim.x,
              (height+bdim.y-1)/bdim.y);
	if(fa<1)
    kernel_render_resample_image0<<<gdim, bdim>>>(out,rowstride, width, height, depth,fa,frame);
	if(fa>0)
	kernel_render_resample_image1<<<gdim, bdim>>>(out,rowstride, width, height, depth,fa,frame);
	
}