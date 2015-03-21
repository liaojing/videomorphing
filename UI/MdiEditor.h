#pragma once
#ifndef MdiEditor_H
#define MdiEditor_H


#include "../Header.h"
#include "ImageEditor.h"
#include "RenderWidget.h"
#include "HalfwayImage.h"
#include "DlgPara.h"
#include "../Algorithm/MatchingThread.h"
#include "../Algorithm/PoissonExt.h"
#include "../Algorithm/QuadraticPath.h"
#include "../Algorithm/SyncThread.h"
#include "CtrBar.h"

class MdiEditor : public QMainWindow
{
	Q_OBJECT

public:
	MdiEditor(QApplication* app, QWidget *parent = 0);
	~MdiEditor();

protected:
	bool ReadXmlFile(QString filename);
	bool WriteXmlFile(QString filename);
	void createDockWidget();
	void createStatusBar();
	void createMenuBar();
	void clear();
	void resizeEvent ();
	void DeleteThread();
	void paintEvent(QPaintEvent *event);
	float SSD(int4 &p1, int4 &p2, std::vector<Mat>& video);
	float Histo(int4 &p1, int4 &p2, std::vector<Mat>& video);
	void AddPoint(std::vector<std::vector<Conp>> &points, int2 &ActIndex, std::vector<cv::Mat> &video, std::vector<cv::Mat> &flow_f, std::vector<cv::Mat> &flow_b);
	void MovePoint(std::vector<std::vector<Conp>> &points, int2 &ActIndex, std::vector<cv::Mat> &video, std::vector<cv::Mat> &flow_f,std::vector<cv::Mat> &flow_b);
	void ConnectPoint();
	bool CudaInit();
	void OpticalFlow(std::vector<Mat>& video1, std::vector<Mat>& video2);	
	void Calculate_op(cuda::GpuMat &gray1, cuda::GpuMat& gray2, Mat& flow);
	
	
public slots:
	void NewProject(bool flag = false);
 	void SaveProject();
	void ModifyPara();
	void updateALL();
	void match_start();
	void sync_start();
	void match_finished();
	void sync_finished();
	void poisson_finished();
	void qpath_finished();
	void SetResults();
	void PtModified(char name, char action,bool flag);
	void ModifyFrameL(int frame);
	void ModifyFrameR(int frame);
	void ModifyFrameBoth(int frame);
	void AutoQuit();

	void ShowHalfway();
	void ShowError();
	void ColorFromImage1();
	void ColorFromImage12();
	void ColorFromImage2();
	void NextStage();
	void PreviousStage();
	void Frame(char name);
	
	
	
private:
	QLabel *readyLabel;
	ImageEditor *imageEditorL;
	HalfwayImage *imageEditorM;
	ImageEditor *imageEditorR;
	RenderWidget *imageEditorA;
	QWidget *widgetA,*widgetL,*widgetR;
	QGridLayout *gridLayoutA,*gridLayoutL,*gridLayoutR;
	QSlider *sliderL,*sliderR;
	CCtrBar *ctrbar;
		
	QDockWidget *imageDockEditorL;
	QDockWidget *imageDockEditorM;
	QDockWidget *imageDockEditorR;
	QDockWidget *imageDockEditorA;
	QAction *new_pro,*save_pro,*prev_stage,*next_stage,*mod_para,*show_halfway,*show_error,*show_image1,*show_image12,*show_image2;
	QAction *cancel,*confirm;
	QMenu *pro_menu,*setting_menu,*view_menu;
	QMenu *result_view,*color_view;
	
	//video
	std::vector<cv::Mat> video1,video2;	
	std::vector<cv::Mat> resample1,resample2;	
	std::vector<cv::Mat> f1,f2,b1,b2;
	
	CSyncThread *sync_thread;
	CMatchingThread *match_thread;
	CPoissonExt *poison_thread;
	CQuadraticPath *qpath_thread;
	Parameters parameters;
	Pyramid pyramid;
	int thread_flag;

public:
	bool _auto;
	QString pro_path;
	QApplication* _app;

	
}; // class MdiEditor

#endif
