#include "RenderWidget.h"
#include "ExternalThread.h"
#include <util/dimage.h>
#include <Algorithm/imgio.h>

RenderWidget::RenderWidget()
{
	setAttribute(Qt::WA_StaticContents);
	setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	_image = QImage(512, 512, QImage::Format_RGB888);
	_image.fill(QColor(240, 240, 240, 255));
	_real_size=_image.size();
	_image_loaded=false;	

	_frame=0;
	_maxf=MAX_FRAME-1;
	_minf=0;
	_colorfrom=1;
	_add=true;
	_save=false;

	_timer=new QTimer(this);
	connect(_timer,SIGNAL(timeout()), this, SLOT(newframe()) );
	_timer->start(1000);

	_pAction = new QAction(this);
	_pAction->setCheckable(true);
	connect(_pAction, SIGNAL(triggered()), this, SLOT(show()));
	connect(_pAction, SIGNAL(triggered()), this, SLOT(setFocus()));

}

RenderWidget::~RenderWidget()
{
}



void RenderWidget::paintEvent(QPaintEvent *event)
{
	if (!_image_loaded)
		return;

	QPainter painter(this);
	painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
	QPixmap pixmaptoshow;
	pixmaptoshow=QPixmap::fromImage(_image.scaled(this->size(),Qt::KeepAspectRatio));

	painter.drawPixmap(0,0, pixmaptoshow);

	_real_size=pixmaptoshow.size();

}



void RenderWidget::set(QString pro_path,int stage, Parameters& parameters, Pyramid& pyramid)
{
	_pro_path=pro_path;
	_stage=stage;
	_minf=0;
	_maxf=parameters.total_frame-1;
	_parameters=&parameters;
	if(_frame>_maxf-1) _frame=_maxf-1,_add=false;
	_extends0=&(pyramid._extends1);
	_extends1=&(pyramid._extends2);
	_vector=&pyramid._vector;
	_qpath=&pyramid._qpath;
	_f0=pyramid._forw0;
	_f1=pyramid._forw1;
	_video0=pyramid._video0;
	_video1=pyramid._video1;

	w=(*_vector)[0].cols;
	h=(*_vector)[0].rows;
	ex=((*_extends1)[0].cols-w)/2;
	d=(*_vector).size();

	_image_loaded=true;
}



void RenderWidget::newframe()
{		
	if(_stage<2)	
	{
		RenderStage1(_mat,0.5,_frame);
	}
	else
	{
		float fa=float(_frame-_minf)/float(_maxf-_minf);
		float color_fa = SmoothStep(fa, 0.0f, 1.0f);
		float geo_fa =  SmoothStep(fa, 0.0f, 1.0f);
		RenderStage2(_mat,color_fa,geo_fa,_colorfrom,_frame);
	}	
	_image=QImage((uchar*)_mat.data, _mat.cols, _mat.rows, QImage::Format_RGB888);

	if(_save)
	{	QString filename;
		int a=(_frame)/100;
		int b=((_frame)%100)/10;
		int c=(_frame)%10;
	
		QDir dir(_pro_path);
		dir.mkdir("results");

		filename.sprintf("%s\\results\\frame%d%d%d.png",_pro_path.toLatin1().data(),a,b,c);
		_image.save(filename);

			
		//mp4

		if(_frame>=_maxf)
		{					
			_save=false;
			QFile file("all.bat");							
			if (file.open(QFile::WriteOnly | QFile::Truncate))
			{
				QTextStream out(&file);	
				QString line;

				//mp4_1
				line.sprintf("libav\\avconv.exe -r 15 -i %s\\results\\frame%%%%03d.png -y %s\\movie_color%d.mp4\n",_pro_path.toLatin1().data(),_pro_path.toLatin1().data(),_colorfrom);
				out<<line;					
							
				//del
				line.sprintf("rmdir /Q /S %s\\results\n", _pro_path.toLatin1().data());
				out << line;

				//del
				line.sprintf("del all.bat\n");
				out << line;
					
			}
			file.flush(); 
			file.close(); 			 

			CExternalThread* external_thread = new CExternalThread();
			external_thread->start(QThread::HighestPriority);
			external_thread->wait();
			emit sigRecordFinished(); 
					
		}
	
	}


	if(_add)
	{
		_frame++;
		if(_frame>=_maxf)
			_add=false;

	}
	else
	{
		_frame--;
		if(_frame<=_minf)
			_add=true;

	}

	update();
}

void RenderWidget::StatusChange(int status)
{

	switch(status)
	{
	case 0:
		_timer->stop();
		_frame=_minf;
		_save=true;
		_add=true;
		_timer->start(200);
		break;
	case 1:
		_timer->start(1000);
		break;
	case 2:
		_timer->stop();
		break;
	case 3:
		_timer->stop();
		_frame=50;
		break;

	}
	update();
}

void RenderWidget::RangeChange(int range)
{
	_timer->stop();
	_minf=50-range;
	_maxf=50+range;
	_frame=_minf;
	_timer->start(1000);
	update();
}

void RenderWidget::RenderStage1(Mat& img, float fa,int frame)
{

	if(_image_loaded)
	{
		cudaArray *a_vector;
		rod::dvector<uchar3> result;
		int rowstride = (w + 31)/32 * 32;
		result.resize(rowstride*h);

		cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float4>();
		cudaMallocArray(&a_vector, &ccd, w, h);	
		cudaMemcpy2DToArray(a_vector, 0, 0, (*_vector)[frame].data,(*_vector)[frame].step, w*sizeof(float4), h,cudaMemcpyHostToDevice); 

		render_resample_image(result, rowstride,w,h,d,fa,frame, _video0, _video1, a_vector, _f0, _f1);

		img=Mat(h,w,CV_8UC3);
		cudaMemcpy2D(img.data,img.step,&result,rowstride*sizeof(uchar3),w*sizeof(uchar3),h,cudaMemcpyDeviceToHost);

		cudaFreeArray(a_vector);
	}	

}

void RenderWidget::RenderStage2(Mat& img,float color_fa, float geo_fa, int color_from, int frame)
{
	if(_image_loaded)
	{
		cudaArray *a_in0, *a_in1 , *a_vector, *a_qpath;
		rod::dvector<uchar3> result;
		int rowstride = (w + 31)/32 * 32;
		result.resize(rowstride*h);

		cudaChannelFormatDesc ccd = cudaCreateChannelDesc<float4>();
		cudaMallocArray(&a_in0, &ccd, w+ex*2, h+ex*2);
		cudaMallocArray(&a_in1, &ccd, w+ex*2, h+ex*2);

		Mat temp;
		(*_extends0)[frame].convertTo(temp,CV_32FC4);
		cudaMemcpy2DToArray(a_in0, 0, 0, temp.data,temp.step, (w+2*ex)*sizeof(float4), h+2*ex,cudaMemcpyHostToDevice);
		(*_extends1)[frame].convertTo(temp,CV_32FC4);
		cudaMemcpy2DToArray(a_in1, 0, 0, temp.data,temp.step, (w+2*ex)*sizeof(float4), h+2*ex,cudaMemcpyHostToDevice);

		ccd = cudaCreateChannelDesc<float2>();
		cudaMallocArray(&a_vector, &ccd, w, h);
		cudaMallocArray(&a_qpath, &ccd, w, h);
		cudaMemcpy2DToArray(a_vector, 0, 0, (*_vector)[frame].data,(*_vector)[frame].step, w*sizeof(float2), h,cudaMemcpyHostToDevice);
		cudaMemcpy2DToArray(a_qpath, 0, 0, (*_qpath)[frame].data,(*_qpath)[frame].step, w*sizeof(float2), h,cudaMemcpyHostToDevice);


		render_halfway_image(result, rowstride,w,h,ex, color_fa, geo_fa, color_from,  a_in0, a_in1, a_vector, a_qpath);

		img=Mat(h,w,CV_8UC3);

		cudaMemcpy2D(img.data,img.step,&result,rowstride*sizeof(uchar3),w*sizeof(uchar3),h,cudaMemcpyDeviceToHost);

		cudaFreeArray(a_in0);
		cudaFreeArray(a_in1);
		cudaFreeArray(a_vector);
		cudaFreeArray(a_qpath);	
	}	
}

inline float RenderWidget::SmoothStep(float t, float a, float b) {
	if (t < a) return 0;
	if (t > b) return 1;
	t = (t-a)/(b-a);
	return t*t * (3-2*t);
}