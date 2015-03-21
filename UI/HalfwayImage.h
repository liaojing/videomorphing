#pragma once
#include "../Header.h"
#include "../Algorithm/Pyramid.h"

class HalfwayImage : public QWidget
{
	 Q_OBJECT

public:
	HalfwayImage(char name,Parameters& parameters,QImage& imagel,QImage& imager);
	void setImage1();
	void setImage2();
	const QImage& getImage() const { return _image; }
	
protected:
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);
	void paintEvent(QPaintEvent *event);
	QSize sizeHint() const;

public:
signals:
	void sigUpdate();
	void sigModified(char name, char action,bool flag);


public:
	QImage _image;
	QImage& _imageL;
	QImage& _imageR;
	bool _image_loaded;
	bool _flag_error;
	char _name;
	Mat _mat;


private:
	QAction *_pAction;
	QSize _real_size;
	bool _button_up;
	QPointF _mouse_pos;
	QPoint _left_pos;
	bool _pressed;

	Parameters& _parameters;

};


