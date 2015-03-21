#include "QuadraticPath.h"
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <helper_cuda.h>       // helper for CUDA error checking

CQuadraticPath::CQuadraticPath(Pyramid &pyramid):_vector(pyramid._vector),_qpath(pyramid._qpath)
{
	times=_vector.size();
	cols=_vector[0].cols;
	rows=_vector[0].rows;
}

void CQuadraticPath::run()
{
	 clock_t start, finish;
	 start=clock();
     optimize();
	finish=clock();
	_runtime= (float)(finish - start)/ CLOCKS_PER_SEC; 
	emit sigFinished();
}


void CQuadraticPath::optimize()
{
	//prepare
	for(int z=0;z<times;z++)
	{
		int size=cols*rows;
		float *j_opt=new float[size*4];
		#pragma omp parallel for
		for(int y=0;y<rows;y++)
			for (int x=0;x<cols;x++)
			{
				float j0[4],j1[4];			
				float vx_x,vy_x;
				float vx_y,vy_y;
				if(x==0)
				{
					vx_x=_vector[z].at<Vec2f>(y,x+1)[0]-_vector[z].at<Vec2f>(y,x)[0];
					vy_x=_vector[z].at<Vec2f>(y,x+1)[1]-_vector[z].at<Vec2f>(y,x)[1];
				}
				else 
				{
					vx_x=_vector[z].at<Vec2f>(y,x)[0]-_vector[z].at<cv::Vec2f>(y,x-1)[0];
					vy_x=_vector[z].at<Vec2f>(y,x)[1]-_vector[z].at<cv::Vec2f>(y,x-1)[1];
				}

				j0[0]=1.0f-vx_x;
				j0[2]=-vy_x;
				j1[0]=1.0f+vx_x;
				j1[2]=vy_x;

				if(y==0)
				{
					vx_y=_vector[z].at<Vec2f>(y+1,x)[0]-_vector[z].at<Vec2f>(y,x)[0];
					vy_y=_vector[z].at<Vec2f>(y+1,x)[1]-_vector[z].at<Vec2f>(y,x)[1];

				}
				else 
				{
					vx_y=_vector[z].at<Vec2f>(y,x)[0]-_vector[z].at<Vec2f>(y-1,x)[0];
					vy_y=_vector[z].at<Vec2f>(y,x)[1]-_vector[z].at<Vec2f>(y-1,x)[1];
				}


				j0[1]=-vx_y;
				j0[3]=1.0f-vy_y;
				j1[1]=vx_y;
				j1[3]=1.0f+vy_y;

				//optimal J
				float nj0[4],nj1[4];
				float la0,lb0,la1,lb1;
				la0=sqrt(j0[0]*j0[0]+j0[2]*j0[2]);	
				lb0=sqrt(j0[1]*j0[1]+j0[3]*j0[3]);
				nj0[0]=j0[0]/la0;
				nj0[2]=j0[2]/la0;
				nj0[1]=j0[1]/lb0;
				nj0[3]=j0[3]/lb0;

				la1=sqrt(j1[0]*j1[0]+j1[2]*j1[2]);	
				lb1=sqrt(j1[1]*j1[1]+j1[3]*j1[3]);
				nj1[0]=j1[0]/la1;
				nj1[2]=j1[2]/la1;
				nj1[1]=j1[1]/lb1;
				nj1[3]=j1[3]/lb1;

				//rotate
				float nj_opt[4];
				for(int i=0;i<4;i++)
					nj_opt[i]=nj0[i]+nj1[i];
				float la_opt=sqrt(nj_opt[0]*nj_opt[0]+nj_opt[2]*nj_opt[2]);
				float lb_opt=sqrt(nj_opt[1]*nj_opt[1]+nj_opt[3]*nj_opt[3]);			


				nj_opt[0]/=la_opt;
				nj_opt[2]/=la_opt;
				nj_opt[1]/=lb_opt;
				nj_opt[3]/=lb_opt;

				//scale
				la_opt=sqrt(la0*la1);
				lb_opt=sqrt(lb0*lb1);

				int index=y*cols*4+x*4;
				j_opt[index+0]=nj_opt[0]*la_opt;
				j_opt[index+2]=nj_opt[2]*la_opt;
				j_opt[index+1]=nj_opt[1]*lb_opt;
				j_opt[index+3]=nj_opt[3]*lb_opt;
			}

			//matrix	
			float *A=new float[5*size];
			int  *columns=new int [5*size];
			int  *rowindex=new int [size+1];
			float *Bx=new float[size];
			float *By=new float[size];		
			float *X=new float[size];
			float *Y=new float[size];

			int  nNonZeros=0;

			memset(A,0,5*size*sizeof(float));
			memset(columns,0,5*size*sizeof(int));
			memset(rowindex,0,(size+1)*sizeof(int));
			memset(Bx,0,size*sizeof(float));
			memset(By,0,size*sizeof(float));		
			memset(X,0,size*sizeof(float));
			memset(Y,0,size*sizeof(float));

			rowindex[0]=0;

			for(int y=0;y<rows;y++)
				for (int x=0;x<cols;x++)
				{
					float a[5];
					a[0]=a[1]=a[2]=a[3]=a[4]=0.0f;

					int ii=y*cols+x;
					if(y-1>=0) 
					{
						a[2]+=1.0f;
						a[0]-=1.0f;	
						Bx[ii]+=j_opt[y*cols*4+x*4+1];
						By[ii]+=j_opt[y*cols*4+x*4+3]-1.0f;		
					}
					if(x-1>=0) 
					{
						a[2]+=1.0f;
						a[1]-=1.0f;	
						Bx[ii]+=j_opt[y*cols*4+x*4+0]-1.0f;
						By[ii]+=j_opt[y*cols*4+x*4+2];

					}
					if(x+1<cols) 
					{
						a[2]+=1.0f;
						a[3]-=1.0f;	
						Bx[ii]-=j_opt[y*cols*4+(x+1)*4+0]-1.0f;
						By[ii]-=j_opt[y*cols*4+(x+1)*4+2];
					}		
					if(y+1<rows) 
					{
						a[2]+=1.0f;
						a[4]-=1.0f;	
						Bx[ii]-=j_opt[(y+1)*cols*4+x*4+1];
						By[ii]-=j_opt[(y+1)*cols*4+x*4+3]-1.0f;		
					}
					//put into A
					if(a[0]!=0)
					{
						A[nNonZeros]=a[0];
						columns[nNonZeros]=ii-cols;
						nNonZeros++;
					}
					if(a[1]!=0)
					{
						A[nNonZeros]=a[1];
						columns[nNonZeros]=ii-1;
						nNonZeros++;
					}
					if(a[2]!=0)
					{
						A[nNonZeros]=a[2];
						columns[nNonZeros]=ii;
						nNonZeros++;
					}
					if(a[3]!=0)
					{
						A[nNonZeros]=a[3];
						columns[nNonZeros]=ii+1;
						nNonZeros++;
					}
					if(a[4]!=0)
					{
						A[nNonZeros]=a[4];
						columns[nNonZeros]=ii+cols;
						nNonZeros++;
					}
					rowindex[ii+1]=nNonZeros;
				}

				//SOLVER
				cudaSolver(A,rowindex,columns,size,nNonZeros,Bx,X);
				cudaSolver(A,rowindex,columns,size,nNonZeros,By,Y);
				//paste				
				#pragma omp parallel for
				for(int y=0;y<rows;y++)
					for(int x=0;x<cols;x++)				
						_qpath[z].at<Vec2f>(y,x)=Vec2f(X[y*cols+x],Y[y*cols+x]);

				delete[] j_opt;	
				delete[] A;
				delete[] rowindex;
				delete[] columns;
				delete[] Bx;
				delete[] By;
				delete[] X;
				delete[] Y;
	}
				
};

void CQuadraticPath::cudaSolver(float* A, int* rowindex, int* columns,int N,int nz,float*Bx, float*X)
{
	const int max_iter = 10000;     
    const float tol = 1e-12f;
    float r0, r1, alpha, beta;
	int *d_col, *d_row;
    float *d_val, *d_x;
    float *d_r, *d_p, *d_omega;
    const float floatone = 1.0;
    const float floatzero = 0.0;
	float dot, nalpha;

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
    checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_omega, N*sizeof(float)));

    cudaMemcpy(d_col, columns, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row, rowindex, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, A, nz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, X, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, Bx, N*sizeof(float), cudaMemcpyHostToDevice);

    /* Conjugate gradient without preconditioning.
       ------------------------------------------
       Follows the description by Golub & Van Loan, "Matrix Computations 3rd ed.", Section 10.2.6  */

    int k = 0;
    r0 = 0;
    cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);

    while (r1 > tol*tol && k <= max_iter)
    {
        k++;

        if (k == 1)
        {
            cublasScopy(cublasHandle, N, d_r, 1, d_p, 1);
        }
        else
        {
            beta = r1/r0;
            cublasSscal(cublasHandle, N, &beta, d_p, 1);
            cublasSaxpy(cublasHandle, N, &floatone, d_r, 1, d_p, 1) ;
        }

        cusparseScsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, N, N, nz, &floatone, descr, d_val, d_row, d_col, d_p, &floatzero, d_omega);
        cublasSdot(cublasHandle, N, d_p, 1, d_omega, 1, &dot);
        alpha = r1/dot;
        cublasSaxpy(cublasHandle, N, &alpha, d_p, 1, d_x, 1);
        nalpha = -alpha;
        cublasSaxpy(cublasHandle, N, &nalpha, d_omega, 1, d_r, 1);
        r0 = r1;
        cublasSdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
    }

    cudaMemcpy(X, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);
	cudaFree(d_x);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_omega);
	
}

CQuadraticPath::~CQuadraticPath(void)
{
	
}
