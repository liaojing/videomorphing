#include "HalfwayImage.h"

HalfwayImage::HalfwayImage(char name,Parameters& parameters,QImage& imagel,QImage& imager):_parameters(parameters),_imageL(imagel),_imageR(imager)
{
	setAttribute(Qt::WA_StaticContents);
	setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	setMouseTracking(true);
	_image = QImage(512, 512, QImage::Format_RGB888);	
	_image.fill(QColor(240, 240, 240, 255));
	_name=name;
	
	_real_size=_image.size();

	_image_loaded=false;
	_flag_error=false;	
	_pressed=false;
	

	_pAction = new QAction(this);
	_pAction->setCheckable(true);
	connect(_pAction, SIGNAL(triggered()), this, SLOT(show()));
	connect(_pAction, SIGNAL(triggered()), this, SLOT(setFocus()));
		
}


QSize HalfwayImage::sizeHint() const
{
	QSize size =  _image.size();
	return size;
}


void HalfwayImage::setImage2()
{
	 
	_image=QImage((uchar*)_mat.data, _mat.cols, _mat.rows, QImage::Format_RGB888);
	_image_loaded=true;	
		
	update();		
}

void HalfwayImage::setImage1()
{
	if(_imageL.width()==_imageR.width()&&_imageL.height()==_imageR.height())
	{
		Mat matl(_imageL.height(),_imageL.width(),CV_8UC3,(uchar*)_imageL.bits(),_imageL.bytesPerLine());
		Mat matr(_imageR.height(),_imageR.width(),CV_8UC3,(uchar*)_imageR.bits(),_imageL.bytesPerLine());
		_mat=(matl+matr)/2;		
		_image=QImage((uchar*)_mat.data, _mat.cols,_mat.rows, QImage::Format_RGB888);			
		_image_loaded=true;	
		update();	
	}
}

void HalfwayImage::mousePressEvent(QMouseEvent *event)
{
	if (_flag_error||!_image_loaded||event->pos().x()>_real_size.width()-1||event->pos().y()>=_real_size.height()-1)
		return;

	if (event->button() == Qt::LeftButton) 
	{
		_left_pos=QPoint(event->pos().x(),event->pos().y());
		_pressed=true;
	}		

	_mouse_pos=QPointF(((float)event->pos().x()+0.5)/(float)_real_size.width(),((float)event->pos().y()+0.5)/(float)_real_size.height());

	emit sigUpdate();
}



void HalfwayImage::mouseMoveEvent(QMouseEvent *event)
{
	if (_flag_error||!_image_loaded||event->pos().x()>_real_size.width()-1||event->pos().y()>=_real_size.height()-1)
		return;
		
	_mouse_pos=QPointF(((float)event->pos().x()+0.5)/(float)_real_size.width(),((float)event->pos().y()+0.5)/(float)_real_size.height());

	emit sigUpdate();
	
}

void HalfwayImage::mouseReleaseEvent(QMouseEvent *event)
{
	if (_flag_error||!_image_loaded)
		return;

	if (_pressed) 
	{
		_pressed=false;
		
		Conp elem;
		elem.p.x=(_left_pos.x()+0.5)/_real_size.width()*_image.width();
		elem.p.y=(_left_pos.y()+0.5)/_real_size.height()*_image.height();
		elem.p.z=_parameters.frame0;
		elem.p.w=1;
		elem.weight=1.0f;

		std::vector<Conp> listl;
		listl.push_back(elem);
		_parameters.lp.push_back(listl);
		_parameters.ActIndex_l.x=_parameters.lp.size()-1;		
		_parameters.ActIndex_l.y=0;

		elem.p.x=(event->pos().x()+0.5)/_real_size.width()*_image.width();
		elem.p.y=(event->pos().y()+0.5)/_real_size.height()*_image.height();
		elem.p.z=_parameters.frame1;
		elem.p.w=1;
		elem.weight=1.0f;

		std::vector<Conp> listr;
		listr.push_back(elem);
		_parameters.rp.push_back(listr);
		_parameters.ActIndex_r.x=_parameters.rp.size()-1;		
		_parameters.ActIndex_r.y=0;	
		
		emit sigModified(_name,'a',true);	
		
	}

	if (event->button() == Qt::MiddleButton)
	{
		emit sigModified(_name, 'c', true);
	}
	emit sigUpdate();
}



void HalfwayImage::paintEvent(QPaintEvent *event)
{
	if(!_image_loaded)
		return;

	QPainter painter(this);
	QPixmap pixmaptoshow;		
		
	QImage tempImage(_image);
	

	QPoint MouseP(_mouse_pos.x()*_image.width(),_mouse_pos.y()*_image.height());
	int radius;
	if (_pressed)		
		radius=_image.width()/8;
	else
		radius=_image.width()/16;
	QRect rect(MouseP-QPoint(radius,radius),MouseP+QPoint(radius,radius));

	for(int y=rect.top();y<=rect.bottom();y++)
		for(int x=rect.left();x<=rect.right();x++)
		{
			if (tempImage.rect().contains(QPoint(x,y))&&(y-MouseP.y())*(y-MouseP.y())+(x-MouseP.x())*(x-MouseP.x())<radius*radius)
			{
				if (_pressed)					
					tempImage.setPixel(QPoint(x,y),_imageR.pixel(QPoint(x,y)));					
				else					
					tempImage.setPixel(QPoint(x,y),_imageL.pixel(QPoint(x,y)));	
			}
		}

		QPainter img_painter(&tempImage);			
		QPen blackPen(qRgba(0, 0, 0, 255));
		img_painter.setPen(blackPen);
		QBrush EmptyBrush(Qt::NoBrush);
		img_painter.setBrush(EmptyBrush);
		img_painter.drawEllipse(MouseP,radius,radius);
				
	pixmaptoshow=QPixmap::fromImage(tempImage.scaled(this->size(),Qt::KeepAspectRatio));	
		
	painter.drawPixmap(0,0, pixmaptoshow);
	_real_size=pixmaptoshow.size();		
}


