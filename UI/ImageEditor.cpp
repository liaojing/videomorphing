#include "ImageEditor.h"

ImageEditor::ImageEditor(char name,Parameters& parameters):_parameters(parameters)
{
	setAttribute(Qt::WA_StaticContents);
	setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	setMouseTracking(true);

	_image = QImage(512, 512, QImage::Format_RGB888);
	_image.fill(QColor(240, 240, 240, 255));

	_name=name;	
	_action='n';
	_image_loaded=false;
	
	_real_size=_image.size();
	
    _pAction = new QAction(this);
    _pAction->setCheckable(true);
    connect(_pAction, SIGNAL(triggered()), this, SLOT(show()));
    connect(_pAction, SIGNAL(triggered()), this, SLOT(setFocus()));

}

ImageEditor::~ImageEditor()
{
}


QSize ImageEditor::sizeHint() const
{
    QSize size =  _image.size();
    return size;
}

void ImageEditor::setImage(cv::Mat& image)
{
	if (image.rows>0) {  
		_image=QImage((uchar*)image.data, image.cols, image.rows, QImage::Format_RGB888);
		_image_loaded=true;
		update();							
	}
} 


void ImageEditor::mousePressEvent(QMouseEvent *event)
{
	if (!_image_loaded)
		return;
	_action='n';
	if (!_image_loaded||event->pos().x()>_real_size.width()-1||event->pos().y()>=_real_size.height()-1)
		return;

	std::vector<std::vector<Conp>> *points;
	int frame;
	int2 *ActIndex;

	if (_name=='l')
		points=&_parameters.lp,frame=_parameters.frame0,ActIndex=&_parameters.ActIndex_l;
	else
		points=&_parameters.rp,frame=_parameters.frame1,ActIndex=&_parameters.ActIndex_r;

	int x=(event->pos().x()+0.5)/_real_size.width()*_image.width();
	int y=(event->pos().y()+0.5)/_real_size.height()*_image.height();
	
	if (event->button() == Qt::LeftButton) 
	{//select
		bool flag=true;
		for (int i=0;i<(*points).size();i++)
			{	
				if((*points)[i].size()==0)
					continue;
				int j=frame;						
				if (abs((*points)[i][j].p.x-x)<=3&&abs((*points)[i][j].p.y-y)<=3&&abs((*points)[i][j].p.z-frame)<1)
				{
					flag=false;
					ActIndex->x=i;
					ActIndex->y=j;
					_action='m';

					//connected
					for(int k=0;k<_parameters.cnt.size();k++)
						for(int l=0;l<_parameters.cnt[k].size();l++)
					{
						if(_name=='l'&&_parameters.cnt[k][l].li.x==i&&_parameters.cnt[k][l].li.y==j)
						{
							_parameters.ActIndex_r=_parameters.cnt[k][l].ri;	
							_parameters.frame1=_parameters.cnt[k][l].ri.y;
							emit sigFrame('l');
							break;
						}
						if(_name=='r'&&_parameters.cnt[k][l].ri.x==i&&_parameters.cnt[k][l].ri.y==j)
						{
							_parameters.ActIndex_l=_parameters.cnt[k][l].li;
							_parameters.frame0=_parameters.cnt[k][l].li.y;
							emit sigFrame('r');	
							break;
						}
					}

					break;
				}
			}
			//new point
			if(flag)
			{
				Conp elem;
				elem.p.x=x;
				elem.p.y=y;
				elem.p.z=frame;
				elem.p.w=1;
				elem.weight=1.0f;

				std::vector<Conp> list;
				list.push_back(elem);
				(*points).push_back(list);
				ActIndex->x=(*points).size()-1;		
				ActIndex->y=0;
				_action='a';
			}
	}
	
	else if(event->button() == Qt::RightButton)
	{
		for (int i=0;i<(*points).size();i++)
			{	
				if((*points)[i].size()==0)
					continue;
				int j=frame;	
				if (abs((*points)[i][j].p.x-x)<=3&&abs((*points)[i][j].p.y-y)<=3&&abs((*points)[i][j].p.z-frame)<1)
				{
					for(int k=0;k<_parameters.cnt.size();k++)
					{
						if(_name=='l'&&_parameters.cnt[k][0].li.x==i)
						{
							_parameters.cnt[k].clear();
							_parameters.cnt.erase(_parameters.cnt.begin()+k);
							break;
						}
						if(_name=='r'&&_parameters.cnt[k][0].ri.x==i)
						{
							_parameters.cnt[k].clear();
							_parameters.cnt.erase(_parameters.cnt.begin()+k);
							break;
						}
					}
					(*points)[i].clear();
					ActIndex->x=-1;	
					ActIndex->y=-1;		
					_action='d';
					break;
				}
			}		
	}
	else if(event->button() == Qt::MiddleButton)
	{
		_action = 'c';
	}
	
	emit sigUpdate();
}


void ImageEditor::mouseMoveEvent(QMouseEvent *event)
{
	if (!_image_loaded)
		return;
	if (event->buttons() & Qt::LeftButton) 
	{
		std::vector<std::vector<Conp>> *points;
		int frame;
		int2 *ActIndex;

		if (_name=='l')
			points=&_parameters.lp,frame=_parameters.frame0,ActIndex=&_parameters.ActIndex_l;
		else
			points=&_parameters.rp,frame=_parameters.frame1,ActIndex=&_parameters.ActIndex_r;

		if (!_image_loaded||ActIndex->x<0||ActIndex->y<0)
			return;

		int x=MAX(MIN(event->pos().x(),_real_size.width()-1),0);
		int y=MAX(MIN(event->pos().y(),_real_size.height()-1),0);

		x=(x+0.5)/_real_size.width()*_image.width();
		y=(y+0.5)/_real_size.height()*_image.height();
		
		Conp elem;
		elem.p.x=x;
		elem.p.y=y;
		elem.p.z=frame;
		elem.p.w=1;
		elem.weight=1.0f;
					
	  (*points)[ActIndex->x][ActIndex->y]=elem;		
		
	} 

	emit sigUpdate();
   
}

void ImageEditor::mouseReleaseEvent(QMouseEvent *event)
{
	if (!_image_loaded)
		return;
	emit sigModified(_name,_action,true);		
}

void ImageEditor::paintEvent(QPaintEvent *event)
{
	if (!_image_loaded)
		return;

	QPainter painter(this);
	painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
	QPixmap pixmaptoshow;
	pixmaptoshow=QPixmap::fromImage(_image.scaled(this->size(),Qt::KeepAspectRatio));

	painter.drawPixmap(0,0, pixmaptoshow);

	_real_size=pixmaptoshow.size();

	std::vector<std::vector<Conp>> *points;
	int frame;
	int2 *ActIndex;

	if (_name=='l')
		points=&_parameters.lp,frame=_parameters.frame0,ActIndex=&_parameters.ActIndex_l;
	else
		points=&_parameters.rp,frame=_parameters.frame1,ActIndex=&_parameters.ActIndex_r;

	//draw point	
	for(int i=0;i<(*points).size();i++)
		for(int j=0;j<(*points)[i].size();j++)
		{
			if(abs(frame-(*points)[i][j].p.z)<1)
			{
				if(i==ActIndex->x&&j==ActIndex->y)
				{
					painter.setBrush(QColor(255, 0, 0, 255));
					painter.setPen(QColor(0, 0, 0, 255));
				}
				else if((*points)[i][j].p.w)
				{
					painter.setBrush(QColor(0, 255, 0, 255));
					painter.setPen(QColor(0, 0, 0, 255));
				}
				else
				{
					painter.setBrush(QColor(255, 255, 0, (*points)[i][j].weight*255));
					painter.setPen(QColor(0, 0, 0, 255));
				}

				QPoint ConP=QPoint(((*points)[i][j].p.x+0.5f)/_image.width()*_real_size.width(),((*points)[i][j].p.y+0.5f)/_image.height()*_real_size.height());
				painter.drawEllipse(ConP,3,3);
			}
				
		}

		for(int k=0;k<_parameters.cnt.size();k++)
			for(int l=0;l<_parameters.cnt[k].size();l++)
		{
			painter.setBrush(Qt::NoBrush);
			painter.setPen(QColor(0, 0, 0, 255));
			QPoint ConP;
			if(_name=='l'&&abs((*points)[_parameters.cnt[k][l].li.x][_parameters.cnt[k][l].li.y].p.z-_parameters.frame0)<1)
			{
				ConP=QPoint(((*points)[_parameters.cnt[k][l].li.x][_parameters.cnt[k][l].li.y].p.x+0.5f)/_image.width()*_real_size.width(),((*points)[_parameters.cnt[k][l].li.x][_parameters.cnt[k][l].li.y].p.y+0.5f)/_image.height()*_real_size.height());
				painter.drawEllipse(ConP,5,5);
			}
			if(_name=='r'&&abs((*points)[_parameters.cnt[k][l].ri.x][_parameters.cnt[k][l].ri.y].p.z-_parameters.frame1)<1)
			{
				ConP=QPoint(((*points)[_parameters.cnt[k][l].ri.x][_parameters.cnt[k][l].ri.y].p.x+0.5f)/_image.width()*_real_size.width(),((*points)[_parameters.cnt[k][l].ri.x][_parameters.cnt[k][l].ri.y].p.y+0.5f)/_image.height()*_real_size.height());
				painter.drawEllipse(ConP,5,5);
			}
		}

}


 