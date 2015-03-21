#pragma once
#include "../Header.h"
#include "../Algorithm/Pyramid.h"

class RenderWidget : public QWidget
{
	Q_OBJECT

public:
	RenderWidget();
	~RenderWidget();
	void set(QString pro_path,int stage, Parameters& parameters, Pyramid& pyramid);
	const QImage& getImage() const { return _image; }
	void RenderStage1(Mat& img,float fa,int frame);
	void RenderStage2(Mat& img, float color_fa, float geo_fa, int color_from,int frame);
	float SmoothStep(float t, float a, float b);
public:
signals:
	void sigRecordFinished();

protected:
	void paintEvent(QPaintEvent *event);

public slots:
	void newframe();
	void RangeChange(int range);
	void StatusChange(int status);

public:
	QImage _image;
	Mat _mat;
	bool _image_loaded;
	int _colorfrom;	
	int _frame;
	int _minf,_maxf;
	bool _add;
	bool _save;

private:
	QAction *_pAction;
	QSize _real_size;
	QString _pro_path;
	QTimer *_timer;
	float _runtime;
	int _stage;
	std::vector<Mat> *_extends0,*_extends1,*_vector,*_qpath;
	cudaArray *_f0, *_f1,*_video0,*_video1;
	Parameters *_parameters;
	int w,h,d,ex;
};

void render_halfway_image(rod::dvector<uchar3> &out, int rowstride, int width, int height, int ex,
						  float color_fa, float geo_fa,int color_from,
						  const cudaArray *img0, 
						  const cudaArray *img1,
						  const cudaArray *vector, 
						  const cudaArray *qpath);
		
void render_resample_image(rod::dvector<uchar3> &out, int rowstride, int width, int height, int depth,
						   float fa, int frame,
						   const cudaArray* img0, 
						   const cudaArray* img1,
						   const cudaArray *vector, 
						   const cudaArray* f0,
						   const cudaArray* f1);