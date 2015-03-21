#pragma  once
#ifndef IMAGEEDITOR_H
#define IMAGEEDITOR_H

#include "../Header.h"
#include "../Algorithm/Pyramid.h"

class ImageEditor : public QWidget
{
     Q_OBJECT

public:
    ImageEditor(char name,Parameters& parameters);
	~ImageEditor();

public:
	QSize sizeHint() const;
    void setImage(cv::Mat& image);
	const QImage& getImage() const { return _image; }

protected:
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
	void paintEvent(QPaintEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);

public:
signals:
	void sigUpdate();
	void sigModified(char name, char action,bool flag);
	void sigFrame(char name);
	
public:
	QImage _image;
	bool _image_loaded;
	char _name;
	char _action;
		
private:
	QAction *_pAction;
	Parameters& _parameters;
	QSize _real_size;

};

#endif


