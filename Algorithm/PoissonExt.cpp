#include "PoissonExt.h"
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <helper_cuda.h>       // helper for CUDA error checking

CPoissonExt::CPoissonExt(Pyramid &pyramid):_vector(pyramid._vector),_extends1(pyramid._extends1),_extends2(pyramid._extends2)
{
	times=_vector.size();
	cols=_vector[0].cols;
	rows=_vector[0].rows;
	ex=(_extends1[0].cols-cols)/2;
	
	type=new int[(cols+2*ex)*(rows+ex*2)];
	index=new int[(cols+2*ex)*(rows+ex*2)];
}



void CPoissonExt::run()
{
	clock_t start, finisrows;
	start=clock();

	for (int i=0;i<times;i++)
	{
		Rect sourceRect(ex,ex,cols,rows);
		_image1=_extends1[i](sourceRect).clone();
		_image2=_extends2[i](sourceRect).clone();

		int size;
		size=prepare(1,_extends1[i],_vector[i]);	
		poissonExtend(_extends1[i],size);
		size=prepare(2,_extends2[i],_vector[i]);   	
		poissonExtend(_extends2[i],size);
	}


	finisrows=clock();
	_runtime= (float)(finisrows - start)  / CLOCKS_PER_SEC;
	emit sigFinished();
}

CPoissonExt::~CPoissonExt(void)
{
	delete[] type;
	delete[] index;
}

int CPoissonExt:: prepare(int side, cv::Mat &extends,cv::Mat &vector)
{
	int size=0;
	int sign;
	cv::Mat *image;
	if(side==1)
		sign=1,image=&_image2;
	else
		sign=-1,image=&_image1;

	for(int y=0;y<rows+ex*2;y++)
		for (int x=0;x<cols+ex*2;x++)
		{
			int ii=y*(cols+2*ex)+x;
			if((extends.at<Vec4b>(y,x))[3]>0)
			{
				type[ii]=2;
				index[ii]=size++;
			}
			else
			{
				//4ÁÚÓò
				if(y>0&&(extends.at<Vec4b>(y-1,x))[3]>0)
				{
					type[ii]=1;
					index[ii]=size++;
					continue;
				}
				if(y<rows+ex*2-1&&(extends.at<Vec4b>(y+1,x))[3]>0)
				{
					type[ii]=1;
					index[ii]=size++;
					continue;
				}
				if(x>0&&(extends.at<Vec4b>(y,x-1))[3]>0)
				{
					type[ii]=1;
					index[ii]=size++;
					continue;
				}
				if(x<cols+ex*2-1&&(extends.at<Vec4b>(y,x+1))[3]>0)
				{
					type[ii]=1;
					index[ii]=size++;
					continue;
				}

				type[ii]=0;
			}
		}


#pragma omp parallel for
		for(int y=0;y<rows+ex*2;y++)
			for (int x=0;x<cols+ex*2;x++)
			{

				int ii=y*(cols+2*ex)+x;

				if(type[ii]==2)//outside
				{
					Vec2f q,p,v;

					q[0]=x-ex;
					q[1]=y-ex;
					p=q;
					v=BilineaGetColor_clamp<Vec2f,Vec2f>(vector,p[0],p[1]);

					float alprowsa=0.8;

					for(int i=0;i<20;i++)
					{
						p=q+v*sign;
						v=alprowsa*BilineaGetColor_clamp<Vec2f,Vec2f>(vector,p[0],p[1])+(1-alprowsa)*v;
					}

					q=p+v*sign;
					if(q[0]>=0&&q[1]>=0&&q[0]<cols&&q[1]<rows)
					{
						Vec4b rgba=BilineaGetColor_clamp<Vec4b,Vec4f>(*image,q[0],q[1]);
						if(rgba[3]==0)
							extends.at<Vec4b>(y,x)=rgba;
						else
							extends.at<Vec4b>(y,x)=Vec4b(255,0,255,0);
					}
					else
						extends.at<Vec4b>(y,x)=Vec4b(255,0,255,0);
				}
			}


			return size;
}

void CPoissonExt::poissonExtend(cv::Mat &dst,int size)
{

	cv::Mat gx=Mat::zeros(rows+2*ex,cols+2*ex,CV_32FC4);
	cv::Mat gy=Mat::zeros(rows+2*ex,cols+2*ex,CV_32FC4);

#pragma omp parallel for
	for(int y=0;y<rows+ex*2;y++)
		for (int x=0;x<cols+ex*2;x++)
		{
			if(type[y*(cols+2*ex)+x]>1)
			{
				Vec4f RGBA0;
				Vec4f RGBA1;
				RGBA1=dst.at<Vec4b>(y,x);

				if(x>0)
				{
					if(type[y*(cols+2*ex)+x-1]>1)
					{
						RGBA0=dst.at<Vec4b>(y,x-1);

						if(RGBA0!=Vec4f(255,0,255,0)&&RGBA1!=Vec4f(255,0,255,0))
							gx.at<Vec4f>(y,x)=RGBA1-RGBA0;
					}
				}

				if(y>0)
				{
					if(type[(y-1)*(cols+2*ex)+x]>1)
					{
						RGBA0=dst.at<Vec4b>(y-1,x);

						if(RGBA0!=Vec4f(255,0,255,0)&&RGBA1!=Vec4f(255,0,255,0))
							gy.at<Vec4f>(y,x)=RGBA1-RGBA0;
					}
				}
			}

		}
		//matrix
		float *A=new float[5*size];
		_INTEGER_t *columns=new  _INTEGER_t[5*size];
		_INTEGER_t *rowindex=new  _INTEGER_t[size+1];
		float *B0=new float[size];
		float *B1=new float[size];
		float *B2=new float[size];
		float *B3=new float[size];
		float *X0=new float[size];
		float *X1=new float[size];
		float *X2=new float[size];
		//float *X3=new float[size];
		_INTEGER_t  nNonZeros=0;

		memset(A,0,5*size*sizeof(float));
		memset(columns,0,5*size*sizeof(int));
		memset(rowindex,0,(size+1)*sizeof(int));
		memset(B0,0,size*sizeof(float));
		memset(B1,0,size*sizeof(float));
		memset(B2,0,size*sizeof(float));
		memset(B3,0,size*sizeof(float));
		memset(X0,0,size*sizeof(float));
		memset(X1,0,size*sizeof(float));
		memset(X2,0,size*sizeof(float));
		//memset(X3,0,size*sizeof(float));

		Vec4f RGBA;
		for(int y=0;y<rows+ex*2;y++)
			for (int x=0;x<cols+ex*2;x++)
			{
				float a[5];
				a[0]=a[1]=a[2]=a[3]=a[4]=0.0f;
				int ii=y*(cols+2*ex)+x;

				switch(type[ii])
				{
				case 0://inside
					break;
				case 1://boundary
					a[2]+=1.0f;
					RGBA=dst.at<Vec4b>(y,x);
					B0[index[ii]]+=RGBA[0];
					B1[index[ii]]+=RGBA[1];
					B2[index[ii]]+=RGBA[2];
					//B3[index[ii]]+=RGBA[3];

				case 2://outside
					if(y-1>=0&&type[ii-(cols+2*ex)]>0)
					{
						a[2]+=1.0f;
						a[0]-=1.0f;
						RGBA=gy.at<Vec4f>(y,x);
						B0[index[ii]]+=RGBA[0];
						B1[index[ii]]+=RGBA[1];
						B2[index[ii]]+=RGBA[2];
						//B3[index[ii]]+=RGBA[3];
					}
					if(x-1>=0&&type[ii-1]>0)
					{
						a[2]+=1.0f;
						a[1]-=1.0f;
						RGBA=gx.at<Vec4f>(y,x);
						B0[index[ii]]+=RGBA[0];
						B1[index[ii]]+=RGBA[1];
						B2[index[ii]]+=RGBA[2];
						//B3[index[ii]]+=RGBA[3];
					}
					if(x+1<cols+2*ex&&type[ii+1]>0)
					{
						a[2]+=1.0f;
						a[3]-=1.0f;
						RGBA=gx.at<Vec4f>(y,x+1);
						B0[index[ii]]-=RGBA[0];
						B1[index[ii]]-=RGBA[1];
						B2[index[ii]]-=RGBA[2];
						//B3[index[ii]]-=RGBA[3];
					}
					if(y+1<rows+2*ex&&type[ii+(cols+2*ex)]>0)
					{
						a[2]+=1.0f;
						a[4]-=1.0f;
						RGBA=gy.at<Vec4f>(y+1,x);
						B0[index[ii]]-=RGBA[0];
						B1[index[ii]]-=RGBA[1];
						B2[index[ii]]-=RGBA[2];
						//B3[index[ii]]-=RGBA[3];
					}

					//put into A
					if(a[0]!=0)
					{
						A[nNonZeros]=a[0];
						columns[nNonZeros]=index[ii-(cols+2*ex)];
						nNonZeros++;
					}
					if(a[1]!=0)
					{
						A[nNonZeros]=a[1];
						columns[nNonZeros]=index[ii-1];
						nNonZeros++;
					}
					if(a[2]!=0)
					{
						A[nNonZeros]=a[2];
						columns[nNonZeros]=index[ii];
						nNonZeros++;
					}
					if(a[3]!=0)
					{
						A[nNonZeros]=a[3];
						columns[nNonZeros]=index[ii+1];
						nNonZeros++;
					}
					if(a[4]!=0)
					{
						A[nNonZeros]=a[4];
						columns[nNonZeros]=index[ii+(cols+2*ex)];
						nNonZeros++;
					}
					rowindex[index[ii]+1]=nNonZeros;

					break;
				}
			}



				//SOLVER
			_INTEGER_t error;
			_MKL_DSS_HANDLE_t solver;
			_INTEGER_t opt=MKL_DSS_MSG_LVL_WARNING + MKL_DSS_TERM_LVL_ERROR + MKL_DSS_SINGLE_PRECISION + MKL_DSS_ZERO_BASED_INDEXING;
			_INTEGER_t sym=MKL_DSS_NON_SYMMETRIC;
			_INTEGER_t typ=MKL_DSS_POSITIVE_DEFINITE;
			_INTEGER_t ord=MKL_DSS_AUTO_ORDER;
			_INTEGER_t sov=MKL_DSS_DEFAULTS;
			_INTEGER_t nRrowss = 1;
			_INTEGER_t size_l=size;

			error= dss_create(solver, opt);
			error = dss_define_structure(solver,sym, rowindex, size_l, size_l,columns, nNonZeros);
			error = dss_reorder( solver,ord, 0);
			error = dss_factor_real( solver, typ, A );
			error = dss_solve_real( solver, sov, B0, nRrowss, X0 );
			error = dss_solve_real( solver, sov, B1, nRrowss, X1 );
			error = dss_solve_real( solver, sov, B2, nRrowss, X2 );
			//error = dss_solve_real( solver, sov, B3, nRrowss, X3 );
			error = dss_delete( solver, opt );



			//paste
				#pragma omp parallel for
			for(int y=0;y<rows+ex*2;y++)
				for (int x=0;x<cols+ex*2;x++)
				{
					int ii=y*(cols+2*ex)+x;
					if(type[ii]>0)
					{
						int R=MIN(MAX(X0[index[ii]],0),255);
						int G=MIN(MAX(X1[index[ii]],0),255);
						int B=MIN(MAX(X2[index[ii]],0),255);
						int A=0;
						dst.at<Vec4b>(y,x)=Vec4b(R,G,B,A);
					}
				}


				delete[] A;
				delete[] B0;
				delete[] B1;
				delete[] B2;
				delete[] B3;
				delete[] X0;
				delete[] X1;
				delete[] X2;
				//delete[] X3;
				delete[] columns;
				delete[] rowindex;

}


////inline functions////
template<class T_in, class T_out>
T_out CPoissonExt::BilineaGetColor_clamp(cv::Mat& img, float px,float py)//clamp for outside of trowse boundary
{
	int x[2],y[2];
	T_out value[2][2];
	int cols=img.cols;
	int rows=img.rows;
	x[0]=floor(px);
	y[0]=floor(py);
	x[1]=ceil(px);
	y[1]=ceil(py);

	float u=px-x[0];
	float v=py-y[0];

	for (int i=0;i<2;i++)
		for(int j=0;j<2;j++)
		{
			int temp_x,temp_y;
			temp_x=x[i];
			temp_y=y[j];
			temp_x=MAX(0,temp_x);
			temp_x=MIN(cols-1,temp_x);
			temp_y=MAX(0,temp_y);
			temp_y=MIN(rows-1,temp_y);
			value[i][j]=(img.at<T_in>(temp_y,temp_x));
		}


		return
			value[0][0]*(1-u)*(1-v)+value[0][1]*(1-u)*v+value[1][0]*u*(1-v)+value[1][1]*u*v;
}


void CPoissonExt::cudaSolver(float* A, int* rowindex, int* columns,int N,int nz,float*Bx, float*X)
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
