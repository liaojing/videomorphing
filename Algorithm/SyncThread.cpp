#include "SyncThread.h"
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <helper_cuda.h>       // helper for CUDA error checking

extern void upsample(float *dest, int dw,int dh, float *orig, int sw,int sh,float ratio);
CSyncThread::CSyncThread(Parameters& parameters,Pyramid &pyramids):_pyramids(pyramids),_parameters(parameters)
{
	runflag=true;
	
	percentage=0.0f;
	_total_l=_pyramids.size()-1;
	_current_l=_total_l;
	_total_iter=_current_iter=0;
	_max_iter=_parameters.max_iter*10;
	int iter_num = _parameters.max_iter * 10;
	for (int el=_total_l;el>=0;el--)
	{
		if(el>0)
		{
			_total_iter+=iter_num*_pyramids[el].width*_pyramids[el].height*_pyramids[el].depth;
			iter_num/=_parameters.max_iter_drop_factor;
		}
		
		float *dx,*dy,*dz;
		dx=dy=dz=NULL;
		d_x.push_back(dx);
		d_y.push_back(dy);
		d_z.push_back(dz);
	}

	
	_timer=NULL;
	_timer=new QTimer(this);
	connect(_timer,SIGNAL(timeout()), this, SLOT(update_result()) );
	_timer->start(1000);
}

CSyncThread::~CSyncThread()
{
	if(_timer)
		delete _timer;
	for(int i=0;i<d_x.size();i++)
	{
		if(d_x[i])
			cudaFree(d_x[i]);
		if(d_y[i])
			cudaFree(d_y[i]);
		if(d_z[i])
			cudaFree(d_z[i]);
	}
	d_x.clear();
	d_y.clear();
	d_z.clear();
}

void  CSyncThread::run()
{
	clock_t start, finish;
	start = clock();
	for(_current_l=_total_l;_current_l>0;_current_l--)
	{
		int el=_current_l;
		_timer->stop();
		if(el==_total_l)
			load_identity(el);
		else
			upsample_level(el,el+1);
		_timer->start();

		optimize_level(el);
		
		_max_iter/=2;
		if(!runflag)
			break;
	}
	finish = clock();
	_timer->stop();
	run_time = (float)(finish - start) / CLOCKS_PER_SEC;
	update_result();
	emit sigFinished();
}

void CSyncThread::load_identity(int el)
{
	int N=_pyramids[el].width*_pyramids[el].height*_pyramids[el].depth;

	checkCudaErrors(cudaMalloc((void **)&d_x[el], N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_y[el], N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_z[el], N*sizeof(float)));
	checkCudaErrors(cudaMemset(d_x[el], 0, N*sizeof(float)));
	checkCudaErrors(cudaMemset(d_y[el], 0, N*sizeof(float)));
	checkCudaErrors(cudaMemset(d_z[el], 0, N*sizeof(float)));
}
void CSyncThread::upsample_level(int el,int pel)
{
	int N=_pyramids[el].width*_pyramids[el].height*_pyramids[el].depth;

	float ratio_x=(float)_pyramids[el].width/(float)_pyramids[pel].width;
	float ratio_y=(float)_pyramids[el].height/(float)_pyramids[pel].height;

	checkCudaErrors(cudaMalloc((void **)&d_x[el], N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_y[el], N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_z[el], N*sizeof(float)));
	checkCudaErrors(cudaMemset(d_x[el], 0, N*sizeof(float)));
	checkCudaErrors(cudaMemset(d_y[el], 0, N*sizeof(float)));
	checkCudaErrors(cudaMemset(d_z[el], 0, N*sizeof(float)));

	for (int i=0;i<_pyramids[el].depth;i++)
	{
		upsample(d_x[el]+i*_pyramids[el].width*_pyramids[el].height,_pyramids[el].width,_pyramids[el].height,
			d_x[pel]+i*_pyramids[pel].width*_pyramids[pel].height,_pyramids[pel].width,_pyramids[pel].height,ratio_x);
		upsample(d_y[el]+i*_pyramids[el].width*_pyramids[el].height,_pyramids[el].width,_pyramids[el].height,
			d_y[pel]+i*_pyramids[pel].width*_pyramids[pel].height,_pyramids[pel].width,_pyramids[pel].height,ratio_y);
		upsample(d_z[el]+i*_pyramids[el].width*_pyramids[el].height,_pyramids[el].width,_pyramids[el].height,
			d_z[pel]+i*_pyramids[pel].width*_pyramids[pel].height,_pyramids[pel].width,_pyramids[pel].height,1.0f);
	}
	

	if(d_x[pel])
		{cudaFree(d_x[pel]);d_x[pel]=NULL;};
	if(d_y[pel])
		{cudaFree(d_y[pel]);d_y[pel]=NULL;};
	if(d_z[pel])
		{cudaFree(d_z[pel]);d_z[pel]=NULL;};


}
void CSyncThread::genMatrix(int *row_ptr, int *col_ind, float *val, int N, int  nz, int* pagestride, float *rhs_x,float *rhs_y,float *rhs_z,int el)
{
	int width=_pyramids[el].width;
	int height=_pyramids[el].height;
	int depth=_pyramids[el].depth;	
	float ratio_x=(float)_pyramids[el].width/(float)_pyramids[0].width;
	float ratio_y=(float)_pyramids[el].height/(float)_pyramids[0].height;
	
	
	// loop over degrees of freedom
	#pragma omp parallel for
	for (int z=0; z<depth; z++)
	{
		int idx;
		if(z==0)
			idx=0;
		else if(z==1)
			idx=pagestride[0];
		else if(z==depth-1)
			idx=pagestride[0]+2*pagestride[1]+(z-3)*pagestride[2];
		else
			idx=pagestride[0]+pagestride[1]+(z-2)*pagestride[2];			
	
	for (int y=0; y<height; y++)
	for (int x=0; x<width; x++)
	{	
		int index=z*width*height+y*width+x;
		row_ptr[index] = idx;
		float data[5][5][5];
		memset((float*)data,0,5*5*5*sizeof(float));

		//UI		
		for(int k=0;k<_parameters.cnt.size();k++)
			for(int l=0;l<_parameters.cnt[k].size();l++)
			{
				int2 li=_parameters.cnt[k][l].li;
				int2 ri=_parameters.cnt[k][l].ri;
				float x0=_parameters.lp[li.x][li.y].p.x*ratio_x;
				float y0=_parameters.lp[li.x][li.y].p.y*ratio_y;
				float z0=_parameters.lp[li.x][li.y].p.z;

				float x1=_parameters.rp[ri.x][ri.y].p.x*ratio_x;
				float y1=_parameters.rp[ri.x][ri.y].p.y*ratio_y;
				float z1=_parameters.rp[ri.x][ri.y].p.z;

				float con_x=(x0+x1)/2.0f;
				float con_y=(y0+y1)/2.0f;
				float con_z=(z0+z1)/2.0f;
				float vx=(x1-x0)/2.0f;
				float vy=(y1-y0)/2.0f;
				float vz=(z1-z0)/2.0f;

				float faz=fabs(z-con_z);
				float fay=fabs(y-con_y);
				float fax=fabs(x-con_x);
				
				if(faz<1&&fay<1&&fax<1)
				{
					float bilinear_w=(1.0-fax)*(1.0-fay)*(1.0-faz);
					data[2][2][2]+=bilinear_w*_parameters.w_ui;
					rhs_x[index]+=bilinear_w*vx*_parameters.w_ui;
					rhs_y[index]+=bilinear_w*vy*_parameters.w_ui;
					rhs_z[index]+=bilinear_w*vz*_parameters.w_ui;
				}
			}
		

		//TPS
		//dxx
		if(x>1)
			data[2][2][0]+=1.0f*2.0f*_parameters.w_tps,	data[2][2][1]+=-2.0f*2.0f*_parameters.w_tps,	data[2][2][2]+=1.0f*2.0f*_parameters.w_tps;
		if(x>0&&x<width-1)
			data[2][2][1]+=-2.0f*2.0f*_parameters.w_tps, data[2][2][2]+=4.0f*2.0f*_parameters.w_tps,	data[2][2][3]+=-2.0f*2.0f*_parameters.w_tps;
		if(x<width-2)
			data[2][2][2]+=1.0f*2.0f*_parameters.w_tps,	data[2][2][3]+=-2.0f*2.0f*_parameters.w_tps,	data[2][2][4]+=1.0f*2.0f*_parameters.w_tps;
		//dyy
		if(y>1)
			data[2][0][2]+=1.0f*2.0f*_parameters.w_tps, data[2][1][2]+=-2.0f*2.0f*_parameters.w_tps,	data[2][2][2]+=1.0f*2.0f*_parameters.w_tps;
		if(y>0&&y<height-1)
			data[2][1][2]+=-2.0f*2.0f*_parameters.w_tps, data[2][2][2]+=4.0f*2.0f*_parameters.w_tps,	data[2][3][2]+=-2.0f*2.0f*_parameters.w_tps;
		if(y<height-2)
			data[2][2][2]+=1.0f*2.0f*_parameters.w_tps,	 data[2][3][2]+=-2.0f*2.0f*_parameters.w_tps,	data[2][4][2]+=1.0f*2.0f*_parameters.w_tps;
		//dzz
		if(z>1)
			data[0][2][2]+=1.0f*2.0f*_parameters.w_tps, data[1][2][2]+=-2.0f*2.0f*_parameters.w_tps,	data[2][2][2]+=1.0f*2.0f*_parameters.w_tps;
		if(z>0&&z<depth-1)
			data[1][2][2]+=-2.0f*2.0f*_parameters.w_tps, data[2][2][2]+=4.0f*2.0f*_parameters.w_tps,	data[3][2][2]+=-2.0f*2.0f*_parameters.w_tps;
		if(z<depth-2)
			data[2][2][2]+=1.0f*2.0f*_parameters.w_tps,	 data[3][2][2]+=-2.0f*2.0f*_parameters.w_tps,	data[4][2][2]+=1.0f*2.0f*_parameters.w_tps;

		//dxy
		if(x>0&&y>0)
			data[2][1][1]+=2.0f*2.0f*_parameters.w_tps,	data[2][1][2]+=-2.0f*2.0f*_parameters.w_tps,	data[2][2][1]+=-2.0f*2.0f*_parameters.w_tps,data[2][2][2]+=2.0f*2.0f*_parameters.w_tps;
		if(x<width-1&&y>0)
			data[2][1][2]+=-2.0f*2.0f*_parameters.w_tps,data[2][1][3]+=2.0f*2.0f*_parameters.w_tps,	data[2][2][2]+=2.0f*2.0f*_parameters.w_tps,	data[2][2][3]+=-2.0f*2.0f*_parameters.w_tps;
		if(x>0&&y<height-1)
			data[2][2][1]+=-2.0f*2.0f*_parameters.w_tps,data[2][2][2]+=2.0f*2.0f*_parameters.w_tps,	data[2][3][1]+=2.0f*2.0f*_parameters.w_tps,	data[2][3][2]+=-2.0f*2.0f*_parameters.w_tps;
		if(x<width-1&&y<height-1)
			data[2][2][2]+=2.0f*2.0f*_parameters.w_tps,	data[2][2][3]+=-2.0f*2.0f*_parameters.w_tps,	data[2][3][2]+=-2.0f*2.0f*_parameters.w_tps,data[2][3][3]+=2.0f*2.0f*_parameters.w_tps;

		//dyz
		if(z>0&&y>0)
			data[1][1][2]+=2.0f*2.0f*_parameters.w_tps,data[1][2][2]+=-2.0f*2.0f*_parameters.w_tps,data[2][1][2]+=-2.0f*2.0f*_parameters.w_tps, data[2][2][2]+=2.0f*2.0f*_parameters.w_tps;
		if(z>0&&y<height-1)
			data[1][2][2]+=-2.0f*2.0f*_parameters.w_tps,data[1][3][2]+=2.0f*2.0f*_parameters.w_tps,data[2][2][2]+=2.0f*2.0f*_parameters.w_tps,data[2][3][2]+=-2.0f*2.0f*_parameters.w_tps;
		if(z<depth-1&&y>0)
			data[2][1][2]+=-2.0f*2.0f*_parameters.w_tps,data[2][2][2]+=2.0f*2.0f*_parameters.w_tps,data[3][1][2]+=2.0f*2.0f*_parameters.w_tps,data[3][2][2]+=-2.0f*2.0f*_parameters.w_tps;
		if(z<depth-1&&y<height-1)
			data[2][2][2]+=2.0f*2.0f*_parameters.w_tps,data[2][3][2]+=-2.0f*2.0f*_parameters.w_tps,data[3][2][2]+=-2.0f*2.0f*_parameters.w_tps,data[3][3][2]+=2.0f*2.0f*_parameters.w_tps;
		//dzx
		if(x>0&&z>0)
			data[1][2][1]+=2.0f*2.0f*_parameters.w_tps,data[2][2][1]+=-2.0f*2.0f*_parameters.w_tps,	data[1][2][2]+=-2.0f*2.0f*_parameters.w_tps,	data[2][2][2]+=2.0f*2.0f*_parameters.w_tps;
		if(x>0&&z<depth-1)
			data[2][2][1]+=-2.0f*2.0f*_parameters.w_tps,data[3][2][1]+=2.0f*2.0f*_parameters.w_tps,	data[2][2][2]+=2.0f*2.0f*_parameters.w_tps,	data[3][2][2]+=-2.0f*2.0f*_parameters.w_tps;
		if(x<width-1&&z>0)
			data[1][2][2]+=-2.0f*2.0f*_parameters.w_tps,data[2][2][2]+=2.0f*2.0f*_parameters.w_tps,	data[1][2][3]+=2.0f*2.0f*_parameters.w_tps,	data[2][2][3]+=-2.0f*2.0f*_parameters.w_tps;
		if(x<width-1&&z<depth-1)
			data[2][2][2]+=2.0f*2.0f*_parameters.w_tps,data[3][2][2]+=-2.0f*2.0f*_parameters.w_tps,	data[2][2][3]+=-2.0f*2.0f*_parameters.w_tps,	data[3][2][3]+=2.0f*2.0f*_parameters.w_tps;

		//dx
		/*if(x>0)
			data[2][2][1]+=-1.0f*_parameters.w_tps*0.5*2.0f,	data[2][2][2]+=1.0f*_parameters.w_tps*0.5*2.0f;
		if(x<width-1)
			data[2][2][3]+=-1.0f*_parameters.w_tps*0.5*2.0f, data[2][2][2]+=1.0f*_parameters.w_tps*0.5*2.0f;

		//dy
		if(y>0)
			data[2][1][2]+=-1.0f*_parameters.w_tps*0.5*2.0f, data[2][2][2]+=1.0f*_parameters.w_tps*0.5*2.0f;
		if(y<height-1)
			data[2][3][2]+=-1.0f*_parameters.w_tps*0.5*2.0f, data[2][2][2]+=1.0f*_parameters.w_tps*0.5*2.0f;	*/	

		for (int i=-2; i<=2; i++)
			for (int j=-2; j<=2; j++)
				for (int k=-2; k<=2; k++)
				{
					if(data[i+2][j+2][k+2]!=0.0f)
					{
						val[idx] = data[i+2][j+2][k+2];
						col_ind[idx] = index+i*width*height+j*width+k;
						idx++;
					}
				}					
			}	
	}

	row_ptr[N] = nz;
}
void CSyncThread::optimize_level(int el)
{

	int k, N = 0, nz = 0, *I = NULL, *J = NULL;
	int *d_col, *d_row;	
	
	float *rhs_X,*rhs_Y,*rhs_Z;
	float r0_x, r1_x,r0_y, r1_y,r0_z, r1_z;
	float alpha, beta;
	float *d_val;
	float *d_rx,*d_ry,*d_rz;
	float *d_px,*d_py,*d_pz;
	float *d_omegax,*d_omegay,*d_omegaz;
	float *val = NULL;
	float dot,  nalpha;
	const float floatone = 1.0;
	const float floatzero = 0.0;
	const float tol = 1e-12f;
	

	/* Generate a random tridiagonal symmetric matrix in CSR (Compressed Sparse Row) format */
	int w=_pyramids[el].width;
	int h=_pyramids[el].height;
	int d=_pyramids[el].depth;
	N = w*h*d;
	int page_stride[3];	
	page_stride[0]=19*w*h-12*(w+h)+4;
	page_stride[1]=24*w*h-14*(w+h)+4;
	page_stride[2]=25*w*h-14*(w+h)+4;
	nz = page_stride[0]*2+page_stride[1]*2+page_stride[2]*(d-4);
	I = (int *)malloc(sizeof(int)*(N+1));                              // csr row pointers for matrix A
	J = (int *)malloc(sizeof(int)*nz);                                 // csr column indices for matrix A
	val = (float *)malloc(sizeof(float)*nz);                           // csr values for matrix A
	rhs_X = (float *)malloc(sizeof(float)*N);
	rhs_Y = (float *)malloc(sizeof(float)*N);
	rhs_Z = (float *)malloc(sizeof(float)*N);
	memset(I,0,sizeof(int)*(N+1));
	memset(J,0,sizeof(int)*nz);
	memset(val,0,sizeof(int)*nz);
	memset(rhs_X,0,sizeof(float)*N);
	memset(rhs_Y,0,sizeof(float)*N);
	memset(rhs_Z,0,sizeof(float)*N);

	//create matrix
	genMatrix(I, J, val, N, nz,page_stride,rhs_X,rhs_Y,rhs_Z,el);
	
	//solve
	/* Create CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);

	checkCudaErrors(cublasStatus);

	/* Create CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;
	cusparseStatus_t cusparseStatus;
	cusparseStatus = cusparseCreate(&cusparseHandle);

	checkCudaErrors(cusparseStatus);

	/* Description of the A matrix*/
	cusparseMatDescr_t descr = 0;
	cusparseStatus = cusparseCreateMatDescr(&descr);

	checkCudaErrors(cusparseStatus);

	/* Define the properties of the matrix */
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

	/* Allocate required memory */
	checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(float)));	
	checkCudaErrors(cudaMalloc((void **)&d_rx, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_ry, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_rz, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_px, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_py, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_pz, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_omegax, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_omegay, N*sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_omegaz, N*sizeof(float)));

	cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, val, nz*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy(d_rx, rhs_X, N*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy(d_ry, rhs_Y, N*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy(d_rz, rhs_Z, N*sizeof(float), cudaMemcpyHostToDevice);

	
	/* Conjugate gradient without preconditioning.
	   ------------------------------------------
	   Follows the description by Golub & Van Loan, "Matrix Computations 3rd ed.", Section 10.2.6  */
   
	k = 0;
	r0_x = r0_y=r0_z=0;
	cublasSdot(cublasHandle, N, d_rx, 1, d_rx, 1, &r1_x);
	cublasSdot(cublasHandle, N, d_ry, 1, d_ry, 1, &r1_y);
	cublasSdot(cublasHandle, N, d_rz, 1, d_rz, 1, &r1_z);

	while (k <= _max_iter&&runflag)
	{
		k++;

		if(r1_x > tol*tol )
		{
			if (k == 1)
			{
				cublasScopy(cublasHandle, N, d_rx, 1, d_px, 1);			
			}
			else
			{
				beta = r1_x/r0_x;
				cublasSscal(cublasHandle, N, &beta, d_px, 1);
				cublasSaxpy(cublasHandle, N, &floatone, d_rx, 1, d_px, 1) ;
			}

			cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &floatone, descr, d_val, d_row, d_col, d_px, &floatzero, d_omegax);
			cublasSdot(cublasHandle, N, d_px, 1, d_omegax, 1, &dot);
			alpha = r1_x/dot;
			cublasSaxpy(cublasHandle, N, &alpha, d_px, 1, d_x[el], 1);
			nalpha = -alpha;
			cublasSaxpy(cublasHandle, N, &nalpha, d_omegax, 1, d_rx, 1);
			r0_x = r1_x;
			cublasSdot(cublasHandle, N, d_rx, 1, d_rx, 1, &r1_x);
		}

		if(r1_y > tol*tol )
		{
			if (k == 1)
			{
				cublasScopy(cublasHandle, N, d_ry, 1, d_py, 1);			
			}
			else
			{
				beta = r1_y/r0_y;
				cublasSscal(cublasHandle, N, &beta, d_py, 1);
				cublasSaxpy(cublasHandle, N, &floatone, d_ry, 1, d_py, 1) ;
			}

			cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &floatone, descr, d_val, d_row, d_col, d_py, &floatzero, d_omegay);
			cublasSdot(cublasHandle, N, d_py, 1, d_omegay, 1, &dot);
			alpha = r1_y/dot;
			cublasSaxpy(cublasHandle, N, &alpha, d_py, 1, d_y[el], 1);
			nalpha = -alpha;
			cublasSaxpy(cublasHandle, N, &nalpha, d_omegay, 1, d_ry, 1);
			r0_y = r1_y;
			cublasSdot(cublasHandle, N, d_ry, 1, d_ry, 1, &r1_y);
		
		}
		
		if(r1_z > tol*tol )
		{
			if (k == 1)
			{
				cublasScopy(cublasHandle, N, d_rz, 1, d_pz, 1);			
			}
			else
			{
				beta = r1_z/r0_z;
				cublasSscal(cublasHandle, N, &beta, d_pz, 1);
				cublasSaxpy(cublasHandle, N, &floatone, d_rz, 1, d_pz, 1) ;
			}

			cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &floatone, descr, d_val, d_row, d_col, d_pz, &floatzero, d_omegaz);
			cublasSdot(cublasHandle, N, d_pz, 1, d_omegaz, 1, &dot);
			alpha = r1_z/dot;
			cublasSaxpy(cublasHandle, N, &alpha, d_pz, 1, d_z[el], 1);
			nalpha = -alpha;
			cublasSaxpy(cublasHandle, N, &nalpha, d_omegaz, 1, d_rz, 1);
			r0_z = r1_z;
			cublasSdot(cublasHandle, N, d_rz, 1, d_rz, 1, &r1_z);
		}
		_current_iter+=N;
		
	}
	

  	/* Destroy contexts */
	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);

	/* Free device memory */
	free(I);
	free(J);
	free(val);
	free(rhs_X);	
	free(rhs_Y);
	free(rhs_Z);
	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);	
	cudaFree(d_px);
	cudaFree(d_py);
	cudaFree(d_pz);
	cudaFree(d_rx);
	cudaFree(d_ry);
	cudaFree(d_rz);
	cudaFree(d_omegax);
	cudaFree(d_omegay);
	cudaFree(d_omegaz);	
};

void CSyncThread::update_result()
{
	int el=_current_l;
	if(el<1) el=1;
	if(d_x[el]&&d_y[el]&&d_z[el])
	{
		int N = _pyramids[el].width*_pyramids[el].height*_pyramids[el].depth;
		float *X,*Y,*Z;
		X = (float *)malloc(sizeof(float)*N);
		Y = (float *)malloc(sizeof(float)*N);
		Z = (float *)malloc(sizeof(float)*N);
		checkCudaErrors(cudaMemcpy(X, d_x[el], N*sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(Y, d_y[el], N*sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(Z, d_z[el], N*sizeof(float), cudaMemcpyDeviceToHost));

		float ratio_x=(float)_pyramids[0].width/(float)_pyramids[el].width;
		float ratio_y=(float)_pyramids[0].height/(float)_pyramids[el].height;

		#pragma omp parallel for
		for (int z=0; z<_pyramids[el].depth; z++)
		{
			Mat mat=Mat(_pyramids[el].height,_pyramids[el].width,CV_32FC4);
			for (int y=0; y<_pyramids[el].height; y++)
				for (int x=0; x<_pyramids[el].width; x++)
				{	
					int index=z*_pyramids[el].width*_pyramids[el].height+y*_pyramids[el].width+x;
					mat.at<Vec4f>(y,x)=Vec4f(X[index]*ratio_x,Y[index]*ratio_y,Z[index],0.0f);
				}

				resize(mat,_pyramids._vector[z],Size(_pyramids[0].width,_pyramids[0].height));
		}
		free(X);	
		free(Y);
		free(Z);
		percentage=(float)_current_iter/(float)_total_iter*100;
		emit sigUpdate();
	}

}


