#include "MdiEditor.h"
#include "ExternalThread.h"

MdiEditor::MdiEditor(QApplication* app, QWidget *parent)
	: QMainWindow(parent)
{
	_app = app;
	readyLabel=NULL;
	new_pro=save_pro=mod_para=prev_stage=next_stage=NULL;
	imageEditorL = new ImageEditor('l',parameters);
	imageEditorR = new ImageEditor('r',parameters);
	imageEditorM = new HalfwayImage('m',parameters,imageEditorL->_image,imageEditorR->_image);
	widgetA = new QWidget();	
	widgetL= new QWidget();
	widgetR = new QWidget();
	imageEditorA = new RenderWidget();
	ctrbar=new CCtrBar();
	
	sync_thread=NULL;
	match_thread=NULL;
 	poison_thread=NULL;
	qpath_thread=NULL;

	omp_set_num_threads(14);
	CudaInit();
	createDockWidget();
	createMenuBar();
	createStatusBar();

	connect(imageEditorL,SIGNAL(sigUpdate()),this,SLOT(updateALL()));
	connect(imageEditorR,SIGNAL(sigUpdate()),this,SLOT(updateALL()));
	connect(imageEditorM,SIGNAL(sigUpdate()),this,SLOT(updateALL()));
	connect(ctrbar,SIGNAL(sigUpdate()),this,SLOT(updateALL()));
	connect(imageEditorL,SIGNAL(sigModified(char,char,bool)),this,SLOT(PtModified(char,char,bool)));
	connect(imageEditorR,SIGNAL(sigModified(char,char,bool)),this,SLOT(PtModified(char,char,bool)));
	connect(imageEditorM,SIGNAL(sigModified(char,char,bool)),this,SLOT(PtModified(char,char,bool)));
	connect(imageEditorL,SIGNAL(sigFrame(char)),this,SLOT(Frame(char)));
	connect(imageEditorR,SIGNAL(sigFrame(char)),this,SLOT(Frame(char)));
	connect(imageEditorL,SIGNAL(sigLayerReorder(char, bool,float, float)),this,SLOT(ReorderLayer(char, bool,float, float)));
	connect(imageEditorR,SIGNAL(sigLayerReorder(char, bool,float, float)),this,SLOT(ReorderLayer(char, bool,float, float)));
	connect(ctrbar,SIGNAL(sigStatusChange(int)),imageEditorA,SLOT(StatusChange(int)));
	connect(ctrbar,SIGNAL(sigRangeChange(int)),imageEditorA,SLOT(RangeChange(int)));
	connect(imageEditorA,SIGNAL(sigRecordFinished()),ctrbar,SLOT(record_finished()));
	connect(sliderL,SIGNAL(valueChanged(int)),this,SLOT(ModifyFrameL(int)));
	connect(sliderR,SIGNAL(valueChanged(int)),this,SLOT(ModifyFrameR(int)));
	connect(sliderL,SIGNAL(valueChanged(int)),this,SLOT(ModifyFrameBoth(int)));
	connect(sliderR,SIGNAL(valueChanged(int)),this,SLOT(ModifyFrameBoth(int)));
	connect(imageEditorA, SIGNAL(sigRecordFinished()), this, SLOT(AutoQuit()));
	
	clear();
    setWindowTitle(tr("Pixel Morph"));
	showMaximized();
}
bool MdiEditor::CudaInit()
{
	int i;
	int device_count;
	if( cudaGetDeviceCount(&device_count) )
		return false;

	for(i=0;i<device_count;i++)
	{
		struct cudaDeviceProp device_prop;
		if(cudaGetDeviceProperties(&device_prop,i)==cudaSuccess)
		{
			if(device_prop.major>=2)
			{
				if(cudaSetDevice(i)==cudaSuccess)
					return true;
			}
		}
	}
	return false;

}

void MdiEditor::DeleteThread()
{
	if (sync_thread)
	{
		disconnect(sync_thread,0,0,0);
		sync_thread->runflag=false;
		sync_thread->wait();
		sync_thread->deleteLater();
		sync_thread=NULL;
	}

	if (match_thread)
	{
		disconnect(match_thread,0,0,0);
		match_thread->runflag=false;
		match_thread->wait();
		match_thread->deleteLater();
		match_thread=NULL;
	}
	 	
  	if(poison_thread)
  	{
  		disconnect(poison_thread,0,0,0);
  		poison_thread->wait();
  		poison_thread->deleteLater();
  		poison_thread=NULL;
  	}
  
  	if(qpath_thread)
  	{
  		disconnect(qpath_thread,0,0,0);
  		qpath_thread->wait();
  		qpath_thread->deleteLater();
  		qpath_thread=NULL;
  	}


}
void MdiEditor::clear()
{
	
	DeleteThread();
	thread_flag=-1;	

	for(int i=0;i<parameters.lp.size();i++)
		parameters.lp[i].clear();
	parameters.lp.clear();
	for(int i=0;i<parameters.rp.size();i++)
		parameters.rp[i].clear();
	parameters.rp.clear();
	parameters.ActIndex_l.x=-1;
	parameters.ActIndex_l.y=-1;
	parameters.ActIndex_r.x=-1;
	parameters.ActIndex_r.y=-1;
	parameters.w_ssim=100.0f;
	parameters.ssim_clamp=0.0f;
	parameters.w_tps=0.05f;
	parameters.w_ui=100000.0f;
	parameters.w_temp=10.0f;
	parameters.max_iter=1000;
	parameters.max_iter_drop_factor=2;
	parameters.eps=0.01f;
	parameters.start_res=8;
	parameters.bcond=BCOND_NONE;
			

	imageEditorL->_image_loaded=false;
	imageEditorR->_image_loaded=false;
	imageEditorM->_image_loaded=false;
	imageEditorM->_flag_error=false;
	
	ctrbar->_status=-1;
}

void MdiEditor::paintEvent(QPaintEvent *event)
{
   (void)event; // ignore argument

	switch(thread_flag)
	{
	case -1://before loaded
		pro_menu->setEnabled(true);
		view_menu->setEnabled(false);
		setting_menu->setEnabled(false);
		save_pro->setEnabled(false);
		prev_stage->setEnabled(false);
		next_stage->setEnabled(false);
		sliderL->setEnabled(false);
		sliderR->setEnabled(false);
		break;
	case 0://before optimizing
	case 1:
		pro_menu->setEnabled(true);
		view_menu->setEnabled(true);
		setting_menu->setEnabled(true);
		save_pro->setEnabled(true);
		prev_stage->setEnabled(false);
		next_stage->setEnabled(true);
		sliderL->setEnabled(true);
		sliderR->setEnabled(true);
		break;	
	default:
		pro_menu->setEnabled(true);
		view_menu->setEnabled(true);
		setting_menu->setEnabled(true);
		save_pro->setEnabled(true);
		prev_stage->setEnabled(true);
		next_stage->setEnabled(false);
		sliderL->setEnabled(true);
		sliderR->setEnabled(true);
		break;	
	}
	
	switch(imageEditorA->_colorfrom)
	{
		case 0:
			show_image1->setChecked(true);
			show_image12->setChecked(false);
			show_image2->setChecked(false);
			break;
		case 1:
			show_image1->setChecked(false);
			show_image12->setChecked(true);
			show_image2->setChecked(false);
			break;
		case 2:
			show_image1->setChecked(false);
			show_image12->setChecked(false);
			show_image2->setChecked(true);
			break;
	}
	if (imageEditorM->_flag_error)
	{
		show_halfway->setChecked(false);
		show_error->setChecked(true);
	}
	else
	{
		show_halfway->setChecked(true);
		show_error->setChecked(false);
	}

}

MdiEditor::~MdiEditor()
{
	clear();

	if(readyLabel)
		delete readyLabel;
	if(imageEditorL)
		delete imageEditorL;
	if(imageEditorM)
		delete imageEditorM;
	if(imageEditorR)
		delete imageEditorR;
	if(imageEditorA)
		delete imageEditorA;
	if(ctrbar)
		delete ctrbar;
	if(sliderL)
		delete sliderL;
	if(sliderR)
		delete sliderR;
	if(gridLayoutA)
		delete gridLayoutA;	
	if(gridLayoutL)
		delete gridLayoutL;	
	if(gridLayoutR)
		delete gridLayoutR;	
	if(widgetA)
		delete widgetA;
	if(widgetL)
		delete widgetL;
	if(widgetR)
		delete widgetR;	
	if(imageDockEditorL)
		delete imageDockEditorL;
	if(imageDockEditorM)
		delete imageDockEditorM;
	if(imageDockEditorR)
		delete imageDockEditorR;
	if(imageDockEditorA)
		delete imageDockEditorA;
	
}

 void MdiEditor::createDockWidget()
 {
	imageDockEditorL = new QDockWidget(tr("Input video0"),this);
	imageDockEditorL->setFeatures(QDockWidget::DockWidgetMovable|
 		QDockWidget::DockWidgetFloatable);
 	imageDockEditorL->setAllowedAreas(Qt::AllDockWidgetAreas);

	gridLayoutL=new QGridLayout();
	gridLayoutL->addWidget(imageEditorL,0,0,1,1);
	sliderL=new QSlider(this);
	sliderL->setOrientation(Qt::Horizontal);
	gridLayoutL->addWidget(sliderL,1,0,1,1);
	widgetL->setLayout(gridLayoutL);
 	imageDockEditorL->setWidget(widgetL);

	imageDockEditorR = new QDockWidget(tr("Input video1"),this);
 	imageDockEditorR->setFeatures(QDockWidget::DockWidgetMovable|
 		QDockWidget::DockWidgetFloatable);
 	imageDockEditorR->setAllowedAreas(Qt::AllDockWidgetAreas);
 
	gridLayoutR=new QGridLayout();
	gridLayoutR->addWidget(imageEditorR,0,0,1,1);
	sliderR=new QSlider(this);
	sliderR->setOrientation(Qt::Horizontal);
	gridLayoutR->addWidget(sliderR,1,0,1,1);
	widgetR->setLayout(gridLayoutR);
	imageDockEditorR->setWidget(widgetR);
 	
	imageDockEditorM = new QDockWidget(tr("Halfway result"),this);
 	imageDockEditorM->setFeatures(QDockWidget::DockWidgetMovable|
 		QDockWidget::DockWidgetFloatable);
 	imageDockEditorM->setAllowedAreas(Qt::AllDockWidgetAreas);
 	imageDockEditorM->setWidget(imageEditorM); 		
		
	imageDockEditorA = new QDockWidget(tr("Morphing result"),this);
 	imageDockEditorA->setFeatures(QDockWidget::DockWidgetMovable|
 		QDockWidget::DockWidgetFloatable);
 	imageDockEditorA->setAllowedAreas(Qt::AllDockWidgetAreas);

	gridLayoutA=new QGridLayout();
	gridLayoutA->addWidget(ctrbar,0,0);
 	gridLayoutA->addWidget(imageEditorA,1,0,10,1);
	widgetA->setLayout(gridLayoutA);
	imageDockEditorA->setWidget(widgetA);

 	addDockWidget(Qt::TopDockWidgetArea,imageDockEditorL);
 	addDockWidget(Qt::RightDockWidgetArea,imageDockEditorR);
 	addDockWidget(Qt::LeftDockWidgetArea,imageDockEditorM);
 	addDockWidget(Qt::BottomDockWidgetArea,imageDockEditorA);

 	setCorner(Qt::TopLeftCorner,Qt::TopDockWidgetArea);
 	setCorner(Qt::TopRightCorner,Qt::RightDockWidgetArea);
 	setCorner(Qt::BottomLeftCorner,Qt::LeftDockWidgetArea);
 	setCorner(Qt::BottomRightCorner,Qt::BottomDockWidgetArea);

 }

 void MdiEditor::createStatusBar()
 {
     readyLabel = new QLabel(tr("Please create/load a project"));
     statusBar()->addWidget(readyLabel, 1);
 }

 void MdiEditor::createMenuBar()
 {
	pro_menu=menuBar()->addMenu("Project");
	new_pro=new QAction("&New Project", this);
	save_pro=new QAction("&Save Project", this);
	prev_stage=new QAction("&Sync stage", this);
	next_stage=new QAction("&Match stage", this);
	pro_menu->addAction(new_pro);
	pro_menu->addAction(save_pro);
	pro_menu->addAction(prev_stage);
	pro_menu->addAction(next_stage);

	setting_menu=menuBar()->addMenu("Settings");
	mod_para=new QAction("&Parameters", this);
	setting_menu->addAction(mod_para);

	view_menu=menuBar()->addMenu("View");
	
	result_view=view_menu->addMenu("Results");
	show_halfway=new QAction("&Halfway Image", this);
	result_view->addAction(show_halfway);
	show_error=new QAction("&Error Image", this);
	result_view->addAction(show_error);

	color_view=view_menu->addMenu("Color from");
	show_image1=new QAction("&Image1", this);
	color_view->addAction(show_image1);
	show_image12=new QAction("&Both image1 & image2", this);
	color_view->addAction(show_image12);
	show_image2=new QAction("&Image2", this);
	color_view->addAction(show_image2);	
	
	show_halfway->setCheckable(true);
	show_error->setCheckable(true);
	show_image1->setCheckable(true);
	show_image12->setCheckable(true);
	show_image2->setCheckable(true);	

	
	//signal
	connect(new_pro,SIGNAL(triggered()),this,SLOT(NewProject()));
	connect(save_pro,SIGNAL(triggered()),this,SLOT(SaveProject()));
	connect(prev_stage,SIGNAL(triggered()),this,SLOT(PreviousStage()));
	connect(next_stage,SIGNAL(triggered()),this,SLOT(NextStage()));
	connect(mod_para,SIGNAL(triggered()),this,SLOT(ModifyPara()));
	connect(show_halfway,SIGNAL(triggered()),this,SLOT(ShowHalfway()));
	connect(show_error,SIGNAL(triggered()),this,SLOT(ShowError()));
	connect(show_image1,SIGNAL(triggered()),this,SLOT(ColorFromImage1()));
	connect(show_image12,SIGNAL(triggered()),this,SLOT(ColorFromImage12()));
	connect(show_image2,SIGNAL(triggered()),this,SLOT(ColorFromImage2()));

 }

 void MdiEditor::resizeEvent ()
 {
 	imageEditorM->update();
 	imageEditorM->updateGeometry();
 	imageEditorL->update();
 	imageEditorL->updateGeometry();
 	imageEditorR->update();
 	imageEditorR->updateGeometry();
 }

 void MdiEditor::updateALL()
 {
 	imageEditorM->update();
 	imageEditorL->update();
 	imageEditorR->update();
	imageEditorA->update();
  }

 void MdiEditor::NewProject(bool flag)
 {
 	clear();
	if (!flag)
		pro_path = QFileDialog::getExistingDirectory(this);
	
	

	if (!pro_path.isNull())
	{
		
		if (!ReadXmlFile(pro_path + "\\settings.xml"))//exist
		{

			//load two videos

			QString ImagePathName1 = QFileDialog::getOpenFileName(
				this,
				"Load Video1",
				QDir::currentPath(),
				"Image files (*.mp4 *.avi *.wmv );All files(*.*)");

			QString ImagePathName2 = QFileDialog::getOpenFileName(
				this,
				"Load Video1",
				QDir::currentPath(),
				"Image files (*.mp4 *.avi *.wmv );All files(*.*)");

		

			VideoCapture cap1(ImagePathName1.toLatin1().data()); // open video1
			VideoCapture cap2(ImagePathName2.toLatin1().data()); // open video2
			if (cap1.isOpened() && cap2.isOpened())  // check if we succeeded
			{
				int w;
				int h;
				for (int i = 0; i<MAX_FRAME; i++)
				{
					Mat frame1;
					Mat frame2;

					bool bSuccess1 = cap1.read(frame1); // read a new frame from video
					bool bSuccess2 = cap2.read(frame2); // read a new frame from video

					if (bSuccess1&&bSuccess2) //if not success, break loop
					{
						//scale to the same size
						if (i == 0)
						{
							w = frame1.cols;
							h = frame1.rows;


							while (w>Max_DIM || h > Max_DIM)
							{

								if (w >= h)
								{
									w = Max_DIM;
									h = (float)Max_DIM / (float)frame1.cols*h;
									if (h % 2 == 1) h++;

								}
								else
								{
									h = Max_DIM;
									w = (float)Max_DIM / (float)frame1.rows*w;
									if (w % 2 == 1) w++;
								}
							}
						}

						cvtColor(frame1, frame1, COLOR_BGR2RGB);
						cv::resize(frame1, frame1, Size(w, h));
						video1.push_back(frame1);
						resample1.push_back(frame1);

						cvtColor(frame2, frame2, COLOR_BGR2RGB);
						cv::resize(frame2, frame2, Size(w, h));
						video2.push_back(frame2);
						resample2.push_back(frame2);


					}
					else
						break;
				}

				parameters.total_frame = min(video1.size(), video2.size());
				if (parameters.total_frame <= 0)
					return;

			}
			else
				return;

		}

		if (thread_flag < 2)
		{

			OpticalFlow(video1, video2);
			pyramid.build(video1, video2, f1, f2, parameters.start_res / 2);
			thread_flag = 0;
			ctrbar->_status = 1;
			sliderL->setRange(0, parameters.total_frame - 1);
			sliderR->setRange(0, parameters.total_frame - 1);
			parameters.frame0 = parameters.frame1 = parameters.total_frame / 2;
			ModifyFrameL(parameters.frame0);
			ModifyFrameR(parameters.frame1);
			sync_start();
		}
		else
		{
			OpticalFlow(resample1, resample2);
			pyramid.build(resample1, resample2, f1, f2, b1, b2, parameters.start_res);
			thread_flag = 2;
			ctrbar->_status = 1;
			sliderL->setRange(0, parameters.total_frame - 1);
			sliderR->setRange(0, parameters.total_frame - 1);
			parameters.frame0 = parameters.frame1 = parameters.total_frame / 2;
			ModifyFrameBoth(parameters.frame0);
			match_start();
		}
		updateALL();
	}
		
 }

 void MdiEditor::SaveProject()
 {

  	WriteXmlFile(pro_path+"\\settings.xml");

	/*int t = video1.size();
	int w = video1[0].cols;
	int h = video1[1].rows;

	for (int i = 0; i <t; i++)
	{
		QString filename;
		int a = i / 100;
		int b = i % 100 / 10;
		int c = i % 10;
		filename.sprintf("%s\\frame%d%d%d.dat", pro_path.toLatin1().data(),a,b,c);
		QFile file(filename.toLatin1().data());
		if (file.open(QIODevice::WriteOnly))
		{
		QDataStream out(&file);
		out.setVersion(QDataStream::Qt_4_3);


		for (int y = 0; y<h; y++)
		for (int x = 0; x<w; x++)
		{
			Vec2f v = pyramid._vector[i].at <Vec2f> (y,x);
			out << v[0] << v[1];
		}

		file.flush();
		file.close();
		}

	}*/
	
 }


 bool MdiEditor::ReadXmlFile(QString filename)
 {
	
	 QDomDocument doc("settings");
  	QFile file(filename);

  	if(file.open(QIODevice::ReadOnly))
  	{
  		doc.setContent(&file);
  		QDomElement root = doc.documentElement();
  		QDomElement child1=root.firstChildElement();

   		while(!child1.isNull())
   		{
			if(child1.tagName()=="stage")
			{
				thread_flag=child1.attribute("stage").toInt();
			}
  			else if (child1.tagName()=="videos")
  			{
				QString ImagePathName1 = pro_path + child1.attribute("video1");
				QString ImagePathName2 = pro_path + child1.attribute("video2");
				QString ImagePathName3 = pro_path + child1.attribute("resample1");
				QString ImagePathName4 = pro_path + child1.attribute("resample2");

				VideoCapture cap1(ImagePathName1.toLatin1().data()); // open video1
				VideoCapture cap2(ImagePathName2.toLatin1().data()); // open video2
				VideoCapture cap3(ImagePathName3.toLatin1().data()); // open video3
				VideoCapture cap4(ImagePathName4.toLatin1().data()); // open video4

				if (cap1.isOpened() && cap2.isOpened() && cap3.isOpened() && cap4.isOpened())  // check if we succeeded
				{
					for (int i = 0; i < MAX_FRAME; i++)
					{
						Mat frame1;
						Mat frame2;
						Mat frame3;
						Mat frame4;

						bool bSuccess1 = cap1.read(frame1); // read a new frame from video
						bool bSuccess2 = cap2.read(frame2); // read a new frame from video
						bool bSuccess3 = cap3.read(frame3); // read a new frame from video
						bool bSuccess4 = cap4.read(frame4); // read a new frame from video

						if (bSuccess1&&bSuccess2&&bSuccess3&&bSuccess4) //if not success, break loop
						{
							//scale to the same size

							cvtColor(frame1, frame1, COLOR_BGR2RGB);
							video1.push_back(frame1);

							cvtColor(frame2, frame2, COLOR_BGR2RGB);
							video2.push_back(frame2);

							cvtColor(frame3, frame3, COLOR_BGR2RGB);
							resample1.push_back(frame3);

							cvtColor(frame4, frame4, COLOR_BGR2RGB);
							resample2.push_back(frame4);
						}
						else
							break;

					}

					parameters.total_frame = video1.size();
					if (parameters.total_frame <= 0)
						return false;
				}
				else
					return false;
		    }
 			else if (child1.tagName()=="parameters")
 			{
 				QDomElement elem=child1.firstChildElement();
 				while(!elem.isNull())
 				{
 					if(elem.tagName()=="weight")
 					{
 						parameters.w_ssim=elem.attribute("ssim").toFloat();
 						parameters.w_tps=elem.attribute("tps").toFloat();
 						parameters.w_ui=elem.attribute("ui").toFloat();
						parameters.w_temp=elem.attribute("temp").toFloat();
 						parameters.ssim_clamp=elem.attribute("ssimclamp").toFloat();
 					}
 					else if(elem.tagName()=="points")
 					{
						QString points=elem.attribute("image1");
						QStringList list=points.split(" ");
						std::vector<Conp> pt_list;
						for (int i=0;i<list.count()-1;i+=5)
						{	
							Conp elem;
							elem.p.x=list[i].toInt();
							elem.p.y=list[i+1].toInt();
							elem.p.z=list[i+2].toInt();
							elem.p.w=list[i+3].toInt();
							elem.weight=list[i+4].toFloat();
							if(elem.p.x!=-1||elem.p.y!=-1||elem.p.z!=-1||elem.p.w!=-1)
								pt_list.push_back(elem);
							else
							{
								parameters.lp.push_back(pt_list);
								pt_list.clear();
							}
						}
						
						
						points=elem.attribute("image2");
						list=points.split(" ");

						for (int i=0;i<list.count()-1;i+=5)
						{	
							Conp elem;
							elem.p.x=list[i].toInt();
							elem.p.y=list[i+1].toInt();
							elem.p.z=list[i+2].toInt();
							elem.p.w=list[i+3].toInt();
							elem.weight=list[i+4].toFloat();
							if(elem.p.x!=-1||elem.p.y!=-1||elem.p.z!=-1||elem.p.w!=-1)
								pt_list.push_back(elem);
							else				
							{
								parameters.rp.push_back(pt_list);
								pt_list.clear();
							}
						}

						points=elem.attribute("connection");
						list=points.split(" ");
						std::vector<Connect> cn_list;
						for (int i=0;i<list.count()-1;i+=4)
						{	
							Connect elem;
							elem.li.x=list[i].toInt();
							elem.li.y=list[i+1].toInt();
							elem.ri.x=list[i+2].toInt();
							elem.ri.y=list[i+3].toInt();
							if(elem.li.x!=-1||elem.li.y!=-1||elem.ri.x!=-1||elem.ri.y!=-1)
								cn_list.push_back(elem);
							else				
							{
								parameters.cnt.push_back(cn_list);
								cn_list.clear();
							}
							
						}
					}
 					else if(elem.tagName()=="boundary")
					{
						int cond=elem.attribute("lock").toInt();
						switch(cond)
						{
						case 0:
							parameters.bcond=BCOND_NONE;
							break;
						case 1:
							parameters.bcond=BCOND_CORNER;
							break;
						case 2:
							parameters.bcond=BCOND_BORDER;
							break;
						}
					}
					
 					else if(elem.tagName()=="debug")
 					{
 						parameters.max_iter=elem.attribute("iternum").toInt();
 						parameters.max_iter_drop_factor=elem.attribute("dropfactor").toFloat();
 						parameters.eps=elem.attribute("eps").toFloat();
 						parameters.start_res=elem.attribute("startres").toInt();
 					}

 					elem=elem.nextSiblingElement();
 				}
		}

   		child1=child1.nextSiblingElement();
 	}
  		file.close();
  		return true;
  }
 	return false;
 }

 bool MdiEditor::WriteXmlFile(QString filename)
 {
 	//getParameters();
  	QFile file(filename);
  	if(file.open(QIODevice::WriteOnly | QIODevice::Truncate |QIODevice::Text))
  	{
  		QString str;
  		QDomDocument doc;
  		QDomText text;
  		QDomElement element;
  		QDomAttr attribute;

  		QDomProcessingInstruction instruction = doc.createProcessingInstruction("xml","version=\'1.0\'");
  		doc.appendChild(instruction);

   		QDomElement root = doc.createElement("project");
   		doc.appendChild(root);
		
		//stage
		QDomElement estage=doc.createElement("stage");
		root.appendChild(estage);
		attribute=doc.createAttribute("stage");
		attribute.setValue(str.sprintf("%d",thread_flag));
		estage.setAttributeNode(attribute);


  		//Images
   		QDomElement eimages=doc.createElement("videos");
   		root.appendChild(eimages);

		QDir dir(pro_path);

		dir.mkdir("video1");
		for(int i=0;i<video1.size();i++)
		{
			QString filename;
			int a=i/100;
			int b=i%100/10;
			int c=i%10;
			filename.sprintf("%s\\video1\\frame%d%d%d.png",pro_path.toLatin1().data(),a,b,c);
			Mat img=video1[i].clone();
			cvtColor(video1[i], img, COLOR_RGB2BGR);
			imwrite(filename.toLatin1().data(),img);
		}
   		attribute=doc.createAttribute("video1");
  		attribute.setValue("\\video1.mp4");
		eimages.setAttributeNode(attribute);

		dir.mkdir("video2");
		for(int i=0;i<video2.size();i++)
		{
			QString filename;
			int a=i/100;
			int b=i%100/10;
			int c=i%10;
			filename.sprintf("%s\\video2\\frame%d%d%d.png",pro_path.toLatin1().data(),a,b,c);
			Mat img=video2[i].clone();
			cvtColor(video2[i], img, COLOR_RGB2BGR);
			imwrite(filename.toLatin1().data(),img);
		}
  		attribute=doc.createAttribute("video2");
  		attribute.setValue("\\video2.mp4");
		eimages.setAttributeNode(attribute);
  		
		dir.mkdir("resample1");
		for(int i=0;i<resample1.size();i++)
		{
			QString filename;
			int a=i/100;
			int b=i%100/10;
			int c=i%10;
			filename.sprintf("%s\\resample1\\frame%d%d%d.png",pro_path.toLatin1().data(),a,b,c);
			Mat img=resample1[i].clone();
			cvtColor(resample1[i], img, COLOR_RGB2BGR);
			imwrite(filename.toLatin1().data(),img);
		}
		attribute=doc.createAttribute("resample1");
		attribute.setValue("\\resample1.mp4");
		eimages.setAttributeNode(attribute);

		dir.mkdir("resample2");
		for(int i=0;i<resample1.size();i++)
		{
			QString filename;
			int a=i/100;
			int b=i%100/10;
			int c=i%10;
			filename.sprintf("%s\\resample2\\frame%d%d%d.png",pro_path.toLatin1().data(),a,b,c);
			Mat img=resample2[i].clone();
			cvtColor(resample2[i], img, COLOR_RGB2BGR);
			imwrite(filename.toLatin1().data(),img);
		}
		attribute=doc.createAttribute("resample2");
		attribute.setValue("\\resample2.mp4");
		eimages.setAttributeNode(attribute);

		QFile batfile("all.bat");
		if (batfile.open(QFile::WriteOnly | QFile::Truncate))
		{
			QTextStream out(&batfile);
			QString line;

			//mp4_1
			line.sprintf("libav\\avconv.exe -r 15 -i %s\\video1\\frame%%%%03d.png -y %s\\video1.mp4\n", pro_path.toLatin1().data(), pro_path.toLatin1().data());
			out << line;
			line.sprintf("libav\\avconv.exe -r 15 -i %s\\video2\\frame%%%%03d.png -y %s\\video2.mp4\n", pro_path.toLatin1().data(), pro_path.toLatin1().data());
			out << line;
			line.sprintf("libav\\avconv.exe -r 15 -i %s\\resample1\\frame%%%%03d.png -y %s\\resample1.mp4\n", pro_path.toLatin1().data(), pro_path.toLatin1().data());
			out << line;
			line.sprintf("libav\\avconv.exe -r 15 -i %s\\resample2\\frame%%%%03d.png -y %s\\resample2.mp4\n", pro_path.toLatin1().data(), pro_path.toLatin1().data());
			out << line;
			//del
			line.sprintf("rmdir /Q /S %s\\video1\n",pro_path.toLatin1().data());
			out<<line;		
			line.sprintf("rmdir /Q /S %s\\video2\n", pro_path.toLatin1().data());
			out << line;
			line.sprintf("rmdir /Q /S %s\\resample1\n", pro_path.toLatin1().data());
			out << line;
			line.sprintf("rmdir /Q /S %s\\resample2\n", pro_path.toLatin1().data());
			out << line;
			line.sprintf("del all.bat\n");
			out << line;
		}
		batfile.flush();
		batfile.close();
		CExternalThread* external_thread = new CExternalThread();
		external_thread->start(QThread::HighestPriority);
		external_thread->wait();

   		//para
 		QDomElement epara=doc.createElement("parameters");
 		root.appendChild(epara);
 				
 		//weight
 		element=doc.createElement("weight");
 		epara.appendChild(element);

 		attribute=doc.createAttribute("ssim");
 		attribute.setValue(str.sprintf("%f",parameters.w_ssim));
 		element.setAttributeNode(attribute);

 		attribute=doc.createAttribute("tps");
 		attribute.setValue(str.sprintf("%f",parameters.w_tps));
 		element.setAttributeNode(attribute);

 		attribute=doc.createAttribute("ui");
 		attribute.setValue(str.sprintf("%f",parameters.w_ui));
 		element.setAttributeNode(attribute);

		attribute=doc.createAttribute("temp");
		attribute.setValue(str.sprintf("%f",parameters.w_temp));
		element.setAttributeNode(attribute);

 		attribute=doc.createAttribute("ssimclamp");
 		attribute.setValue(str.sprintf("%f",parameters.ssim_clamp));
 		element.setAttributeNode(attribute);

 		//control points
 		element=doc.createElement("points");
 		epara.appendChild(element);

		int counter=0;
 		attribute=doc.createAttribute("image1");
 		str="";
 		for(size_t i=0;i<parameters.lp.size();i++)
		{
			QString num;
			for (size_t j=0;j<parameters.lp[i].size();j++)
			{ 				
 				str.append(num.sprintf("%d ",parameters.lp[i][j].p.x));
 				str.append(num.sprintf("%d ",parameters.lp[i][j].p.y));
				str.append(num.sprintf("%d ",parameters.lp[i][j].p.z));	
				str.append(num.sprintf("%d ",parameters.lp[i][j].p.w));	
				str.append(num.sprintf("%f ",parameters.lp[i][j].weight));
				if(parameters.lp[i][j].p.w==1) counter++;
 			}
			
				str.append(num.sprintf("%d ",-1));
				str.append(num.sprintf("%d ",-1));
				str.append(num.sprintf("%d ",-1));
				str.append(num.sprintf("%d ",-1));
				str.append(num.sprintf("%f ",-1));
			
		}
 		attribute.setValue(str);
 		element.setAttributeNode(attribute);


 		attribute=doc.createAttribute("image2");
 		str="";
		for(size_t i=0;i<parameters.rp.size();i++)
		{
			QString num;
			for (size_t j=0;j<parameters.rp[i].size();j++)
			{ 				
 				str.append(num.sprintf("%d ",parameters.rp[i][j].p.x));
 				str.append(num.sprintf("%d ",parameters.rp[i][j].p.y));
				str.append(num.sprintf("%d ",parameters.rp[i][j].p.z));		
				str.append(num.sprintf("%d ",parameters.rp[i][j].p.w));	
				str.append(num.sprintf("%f ",parameters.rp[i][j].weight));
				if(parameters.rp[i][j].p.w==1) counter++;
 			}	
			
				str.append(num.sprintf("%d ",-1));
				str.append(num.sprintf("%d ",-1));
				str.append(num.sprintf("%d ",-1));
				str.append(num.sprintf("%d ",-1));
				str.append(num.sprintf("%f ",-1));
			
		}
 		attribute.setValue(str);
 		element.setAttributeNode(attribute);		

		attribute=doc.createAttribute("connection");
		str="";
		for(size_t i=0;i<parameters.cnt.size();i++)
		{
			QString num;
			for(size_t j=0;j<parameters.cnt[i].size();j++)
			{		
				str.append(num.sprintf("%d ",parameters.cnt[i][j].li.x));
				str.append(num.sprintf("%d ",parameters.cnt[i][j].li.y));
				str.append(num.sprintf("%d ",parameters.cnt[i][j].ri.x));
				str.append(num.sprintf("%d ",parameters.cnt[i][j].ri.y));
			}
			
				str.append(num.sprintf("%d ",-1));
				str.append(num.sprintf("%d ",-1));
				str.append(num.sprintf("%d ",-1));
				str.append(num.sprintf("%d ",-1));
						
		}
		attribute.setValue(str);
		element.setAttributeNode(attribute);	

		attribute=doc.createAttribute("num");		
		attribute.setValue(str.sprintf("%d",counter));
 		element.setAttributeNode(attribute);		
 		//boundary
 		element=doc.createElement("boundary");
 		epara.appendChild(element);

 		attribute=doc.createAttribute("lock");
		int bcond=0;
		switch(parameters.bcond)
		{
		case BCOND_NONE:
			bcond=0;
			break;
		case BCOND_CORNER:
			bcond=1;
			break;
		case BCOND_BORDER:
			bcond=2;
			break;
		}
 		attribute.setValue(str.sprintf("%d",bcond));
 		element.setAttributeNode(attribute);


 		//debug
 		element=doc.createElement("debug");
 		epara.appendChild(element);

 		attribute=doc.createAttribute("iternum");
 		attribute.setValue(str.sprintf("%d",parameters.max_iter));
 		element.setAttributeNode(attribute);

 		attribute=doc.createAttribute("dropfactor");
 		attribute.setValue(str.sprintf("%f",parameters.max_iter_drop_factor));
 		element.setAttributeNode(attribute);

 		attribute=doc.createAttribute("eps");
 		attribute.setValue(str.sprintf("%f",parameters.eps));
 		element.setAttributeNode(attribute);

 		attribute=doc.createAttribute("startres");
 		attribute.setValue(str.sprintf("%d",parameters.start_res));
 		element.setAttributeNode(attribute);

   		QTextStream out(&file);
  		out.setCodec("UTF-8");
  		doc.save(out,4);

  		file.close();
 		return true;
  	}

 	return false;
 }

 void MdiEditor::SetResults()
 {
	 if (thread_flag==-1)
	{
		delete readyLabel;
		readyLabel = new QLabel(tr("Please create/load a project"));
		statusBar()->addWidget(readyLabel, 1);
		imageDockEditorM->setPalette(QPalette(QColor ( 219, 219, 219 )));
		imageDockEditorA->setPalette(QPalette(QColor ( 219, 219, 219 )));
	}
	else if(thread_flag==0)
	{
		delete readyLabel;
		QString str;
		str.sprintf("Synchronizing %f%%",sync_thread->percentage);
		readyLabel = new QLabel(str);
		statusBar()->addWidget(readyLabel, 1);
		imageDockEditorM->setPalette(QPalette(QColor ( 255,0,0 )));
		imageDockEditorA->setPalette(QPalette(QColor ( 255,0,0 )));
	    imageEditorA->set(pro_path,thread_flag,parameters,pyramid);		
	}
	else if(thread_flag==1)
	{
		delete readyLabel;
		readyLabel = new QLabel(tr("Synchronization finished"));
		statusBar()->addWidget(readyLabel, 1);
		imageDockEditorM->setPalette(QPalette(QColor ( 0,255,0 )));
		imageDockEditorA->setPalette(QPalette(QColor ( 0,255,0 )));
		imageEditorA->set(pro_path,thread_flag,parameters,pyramid);		
	}
	else if(thread_flag==2)
	{
		delete readyLabel;
		QString str;
		str.sprintf("Optimizing %f%%",match_thread->percentage);
		readyLabel = new QLabel(str);
		statusBar()->addWidget(readyLabel);
		imageDockEditorM->setPalette(QPalette(QColor ( 255, 0, 0 )));
		imageDockEditorA->setPalette(QPalette(QColor ( 255, 0, 0 )));
		imageEditorA->set(pro_path,thread_flag,parameters,pyramid);	
		imageEditorA->RenderStage2(imageEditorM->_mat,0.5, 0.5, 1, parameters.frame0);
		imageEditorM->setImage2();
	}
	else if(thread_flag<5)
	{
		delete readyLabel;
		readyLabel = new QLabel(tr("optimization finished, post-processing"));
		statusBar()->addWidget(readyLabel, 1);
		imageDockEditorM->setPalette(QPalette(QColor ( 255, 255, 0 )));
		imageDockEditorA->setPalette(QPalette(QColor ( 255, 255, 0 )));
		imageEditorA->set(pro_path,thread_flag,parameters,pyramid);	
		imageEditorA->RenderStage2(imageEditorM->_mat,0.5, 0.5, 1, parameters.frame0);
		imageEditorM->setImage2();		

	}
	else
	{
		delete readyLabel;
		readyLabel = new QLabel(tr("Completed"));
		statusBar()->addWidget(readyLabel, 1);
		imageDockEditorM->setPalette(QPalette(QColor ( 0, 255, 0 )));
		imageDockEditorA->setPalette(QPalette(QColor ( 0, 255, 0 )));		
		imageEditorA->set(pro_path,thread_flag,parameters,pyramid);	
		imageEditorA->RenderStage2(imageEditorM->_mat,0.5, 0.5, 1, parameters.frame0);
		imageEditorM->setImage2();		

		if (_auto)
		{
			ctrbar->_status = 0;
			imageEditorA->StatusChange(0);
		}
	}

	updateALL();
	
 }

 void MdiEditor::ModifyPara()
 {
	 CDlgPara dlg(parameters);
	 connect(&dlg,SIGNAL(sigModified(char,char,bool)),this,SLOT(PtModified(char,char,bool)));
	 dlg.exec();
 }

 void MdiEditor::Frame(char name)
 {
	 	 if(name=='l')
			 ModifyFrameR(parameters.frame1);
		 else
			 ModifyFrameL(parameters.frame0);
	 
 }

 void MdiEditor::ModifyFrameL(int frame)
 {
	 if(thread_flag<2)
	 {
		 parameters.frame0=frame;
		 imageEditorL->setImage(video1[parameters.frame0]);			
		 imageEditorM->setImage1();
		 sliderL->blockSignals(true);
		 sliderL->setValue(frame);
		 sliderL->blockSignals(false);
		
	 }
 };

 void MdiEditor::ModifyFrameR(int frame)
 {
	 if(thread_flag<2)
	 {
		 parameters.frame1=frame;
		 imageEditorR->setImage(video2[parameters.frame1]);	
		 imageEditorM->setImage1();
		 sliderR->blockSignals(true);
		 sliderR->setValue(frame);
		 sliderR->blockSignals(false);

	 }	
 }

 
 void MdiEditor::ModifyFrameBoth(int frame)
 {
	 if(thread_flag>=2)
	 {
		 parameters.frame0=frame;
		 parameters.frame1=frame;
		 sliderL->blockSignals(true);
		 sliderL->setValue(frame);
		 sliderL->blockSignals(false);
		 sliderR->blockSignals(true);
		 sliderR->setValue(frame);
		 sliderR->blockSignals(false);
		 imageEditorR->setImage(resample2[parameters.frame1]);
		 imageEditorL->setImage(resample1[parameters.frame0]);		
		 imageEditorA->RenderStage2(imageEditorM->_mat,0.5, 0.5, 1, parameters.frame0);
		 imageEditorM->setImage2();				
	 }
	 
 }
	
		
	 
 void MdiEditor::PtModified(char name, char action,bool flag)
 {
	 
	 if(name=='l')
	 {
		 if(action=='a')
			 AddPoint(parameters.lp,parameters.ActIndex_l,resample1,f1,b1);
		 else if(action=='m')
			 MovePoint(parameters.lp,parameters.ActIndex_l,resample1,f1,b1);
		 else if(action=='c')
			ConnectPoint();

	 }
	 else if(name=='r')
	 {
		 if(action=='a')
			 AddPoint(parameters.rp,parameters.ActIndex_r,resample2,f2,b2);
		 else if(action=='m')
			 MovePoint(parameters.rp,parameters.ActIndex_r,resample2,f2,b2);
		 else if(action=='c')
			 ConnectPoint();
	 }
	 else if(name=='m')
	 {
		 if(action=='a')
		 {
			 AddPoint(parameters.lp,parameters.ActIndex_l,resample1,f1,b1);		 
			 AddPoint(parameters.rp,parameters.ActIndex_r,resample2,f2,b2);
			 ConnectPoint();
		 }
	}
	 if(flag)
	 {
		if(thread_flag<2)
			 sync_start();
		 else
			 match_start();
	 }
	
	 updateALL();

 }


 void MdiEditor::AddPoint(std::vector<std::vector<Conp>> &points, int2 &ActIndex, std::vector<cv::Mat> &video, std::vector<cv::Mat> &flow_f,std::vector<cv::Mat> &flow_b)
 {
	  int rows=video[0].rows;
	  int cols=video[0].cols;

	 //backward
	 Conp pt0=points[ActIndex.x][ActIndex.y];
	 Conp pt=pt0;
	 pt.p.w=0;
	 for (int t=pt.p.z;t>0;t--)
	 {
		 int x=pt.p.x;
		 int y=pt.p.y;
		 x=MIN(MAX(0,x),cols-1);
		 y=MIN(MAX(0,y),rows-1);
		 		
		 Vec2f b=flow_b[t].at<Vec2f>(y,x);
		 pt.p.x+=b[0]+0.5;
		 pt.p.y+=b[1]+0.5;
		 pt.p.z+=-1;
		 pt.weight=Histo(pt.p,pt0.p,video);

		points[ActIndex.x].insert(points[ActIndex.x].begin(),pt);
	 }

	 //forward
	 pt=pt0;
	 pt.p.w=0;
	 for(int t=pt.p.z;t<parameters.total_frame-1;t++)
	 {
		 int x=pt.p.x;
		 int y=pt.p.y;	

		 x=MIN(MAX(0,x),cols-1);
		 y=MIN(MAX(0,y),rows-1);

		 Vec2f f=flow_f[t].at<Vec2f>(y,x);
		 pt.p.x+=f[0]+0.5;
		 pt.p.y+=f[1]+0.5;
		 pt.p.z+=1;

		 pt.weight=Histo(pt.p,pt0.p,video);

		 points[ActIndex.x].push_back(pt);
	 }
	ActIndex.y=pt0.p.z;
 }


void MdiEditor::MovePoint(std::vector<std::vector<Conp>> &points, int2 &ActIndex, std::vector<cv::Mat> &video, std::vector<cv::Mat> &flow_f,std::vector<cv::Mat> &flow_b)
{
	int cols=video[0].cols;
	int rows=video[0].rows;
	int beg,mid,end;
	beg=-1;
	mid=ActIndex.y;
	end=points[ActIndex.x].size();

	//beg
	int i=ActIndex.x;
	for(int j=mid-1;j>=0;j--)
		if (points[i][j].p.w)
		{
			beg=j;
			break;
		}
		//backward
		Conp pt=points[i][mid];
		pt.p.w=0;
		for(int t=mid;t>beg+1;t--)
		{
			int x=pt.p.x;
			int y=pt.p.y;		

			x=MIN(MAX(0,x),cols-1);
			y=MIN(MAX(0,y),rows-1);

			Vec2f b=flow_b[t].at<Vec2f>(y,x);
			pt.p.x+=b[0]+0.5;
			pt.p.y+=b[1]+0.5;
			pt.p.z+=-1;
			pt.weight=Histo(pt.p,points[i][mid].p,video);

			points[i][t-1]=pt;			 
		}
		if(beg>=0)
		{
			pt=points[i][beg];
			pt.p.w=0;
			for(int t=beg;t<mid-1;t++)
			{
				float fa=(float)((t+1)-beg)/(float)(mid-beg);

				int x=pt.p.x;
				int y=pt.p.y;		

				x=MIN(MAX(0,x),cols-1);
				y=MIN(MAX(0,y),rows-1);
				
				Vec2f f=flow_f[t].at<Vec2f>(y,x);
				pt.p.x+=f[0]+0.5;
				pt.p.y+=f[1]+0.5;
				pt.p.z+=1;

				points[i][t+1].p.x=points[i][t+1].p.x*fa+pt.p.x*(1-fa);
				points[i][t+1].p.y=points[i][t+1].p.y*fa+pt.p.y*(1-fa);
				points[i][t+1].weight=Histo(points[i][t+1].p,points[i][mid].p,video)*fa+Histo(points[i][t+1].p,points[i][beg].p,video)*(1-fa);			

			}
		}

		//end
		for(int j=mid+1;j<points[i].size();j++)
			if (points[i][j].p.w)
			{
				end=j;
				break;
			}
	
			pt=points[i][mid];
			pt.p.w=0;
			for(int t=mid;t<end-1;t++)
			{
				int x=pt.p.x;
				int y=pt.p.y;		

				x=MIN(MAX(0,x),cols-1);
				y=MIN(MAX(0,y),rows-1);
				
				Vec2f f=flow_f[t].at<Vec2f>(y,x);
				pt.p.x+=f[0]+0.5;
				pt.p.y+=f[1]+0.5;
				pt.p.z+=1;		
				pt.weight=Histo(pt.p,points[i][mid].p,video);

				points[i][t+1]=pt;			 
			}

			if(end<points[i].size())
			{
				pt=points[i][end];
				pt.p.w=0;
				for(int t=end;t>mid+1;t--)
				{
					float fa=(float)(end-(t-1))/(float)(end-mid);

					int x=pt.p.x;
					int y=pt.p.y;		

					x=MIN(MAX(0,x),cols-1);
					y=MIN(MAX(0,y),rows-1);

					Vec2f b=flow_b[t].at<Vec2f>(y,x);
					pt.p.x+=b[0]+0.5;
					pt.p.y+=b[1]+0.5;
					pt.p.z+=-1;

					points[i][t-1].p.x=points[i][t-1].p.x*fa+pt.p.x*(1-fa);
					points[i][t-1].p.y=points[i][t-1].p.y*fa+pt.p.y*(1-fa);			
					points[i][t-1].weight=Histo(points[i][t-1].p,points[i][mid].p,video)*fa+Histo(points[i][t-1].p,points[i][end].p,video)*(1-fa);								
				
				}
			}
}

void MdiEditor::ConnectPoint()
{
	//check already connect or not
	if (parameters.ActIndex_l.x<0 || parameters.ActIndex_l.y<0 || parameters.ActIndex_r.x<0 || parameters.ActIndex_r.y<0)
		return;
	int flag = 0;
	
	int k, l;
	
	for (k = 0; k<parameters.cnt.size(); k++)
	{	
		for (l = 0; l < parameters.cnt[k].size(); l++)
		{
			if (parameters.cnt[k][l].li.x == parameters.ActIndex_l.x&&parameters.cnt[k][l].li.y == parameters.ActIndex_l.y || parameters.cnt[k][l].ri.x == parameters.ActIndex_r.x&&parameters.cnt[k][l].ri.y == parameters.ActIndex_r.y)
			{
				flag = 1;

				if (parameters.cnt[k][l].li.x == parameters.ActIndex_l.x&&parameters.cnt[k][l].li.y == parameters.ActIndex_l.y && parameters.cnt[k][l].ri.x == parameters.ActIndex_r.x&&parameters.cnt[k][l].ri.y == parameters.ActIndex_r.y)
				flag = 2;
					
				
				break;
			}		

		}
		if (flag)
			break;
	}


	
	switch (flag)
	{
		case 0: //add
			if (thread_flag < 2)
			{
				Connect c;
				c.li = parameters.ActIndex_l;
				c.ri = parameters.ActIndex_r;

				std::vector<Connect> list;
				list.push_back(c);
				parameters.cnt.push_back(list);
			}
			else
			{
				std::vector<Connect> list;				
				
				for (int i = 0; i < parameters.total_frame;i++)
				{
					Connect c;
					c.li.x = parameters.ActIndex_l.x;
					c.li.y = i;
					c.ri.x = parameters.ActIndex_r.x;
					c.ri.y = i;
					list.push_back(c);
				}
				
				parameters.cnt.push_back(list);
			}
			break;
		case 1:
			break;
		case 2: //delete
			if (thread_flag < 2)
			{
				parameters.cnt[k].erase(parameters.cnt[k].begin() + l);
				if (parameters.cnt[k].size()==0)
					parameters.cnt.erase(parameters.cnt.begin() + k);
			}
			else
			{
				parameters.cnt[k].clear();
				parameters.cnt.erase(parameters.cnt.begin() + k);
			}
			break;
		default:
			break;
	}			
	
}

 float MdiEditor::SSD(int4 &p1, int4 &p2, std::vector<Mat>& video)
 {
	 int rows=video[0].rows;
	 int cols=video[0].cols;
	 float ssd=0.0f;
#pragma omp parallel for
	 for (int i=0;i<5;i++)
		for(int j=0;j<5;j++)
	{
		int x=p1.x+j-2;
		int y=p1.y+i-2;
		int t=p1.z;
		if (x<0) x=0;
		if (x>cols-1) x=cols-1;
		if (y<0) y=0;
		if (y>rows-1) y=rows-1;

		Vec3b color1=video[t].at<Vec3b>(y,x);

		x=p2.x+j-2;
		y=p2.y+i-2;
		t=p2.z;
		if (x<0) x=0;
		if (x>cols-1) x=cols-1;
		if (y<0) y=0;
		if (y>rows-1) y=rows-1;

		Vec3b color2=video[t].at<Vec3b>(y,x);

		float dr=color1[0]-color2[0];
		float dg=color1[1]-color2[1];
		float db=color1[2]-color2[2];
		ssd+=dr*dr+dg*dg+db*db;		
	}

		return ssd;
 }

 
 float MdiEditor::Histo(int4 &p1, int4 &p2, std::vector<Mat>& video)
 {
	 int rows=video[0].rows;
	 int cols=video[0].cols;
	 int lx,ly,rx,ry,t;

	 t=p1.z;
	 lx=p1.x-3;
	 ly=p1.y-3;	
	 if (lx<0) lx=0;
	 if (lx>cols-1) lx=cols-1;
	 if (ly<0) ly=0;
	 if (ly>rows-1) ly=rows-1;

	 rx=p1.x+3;
	 ry=p1.y+3;	
	 if (rx<0) rx=0;
	 if (rx>cols-1) rx=cols-1;
	 if (ry<0) ry=0;
	 if (ry>rows-1) ry=rows-1;

	 Mat sub1 = video[t](Range(ly,ry),Range(lx,rx));

	 t=p2.z;
	 lx=p2.x-3;
	 ly=p2.y-3;	
	 if (lx<0) lx=0;
	 if (lx>cols-1) lx=cols-1;
	 if (ly<0) ly=0;
	 if (ly>rows-1) ly=rows-1;

	 rx=p2.x+3;
	 ry=p2.y+3;	
	 if (rx<0) rx=0;
	 if (rx>cols-1) rx=cols-1;
	 if (ry<0) ry=0;
	 if (ry>rows-1) ry=rows-1;

	Mat sub2 = video[t](Range(ly,ry),Range(lx,rx));


	/// Using 30 bins for hue and 32 for saturation
	int r_bins = 10; int g_bins = 10; int b_bins=10;
	int histSize[] = { r_bins, g_bins, b_bins };

	// hue varies from 0 to 256, saturation from 0 to 180
	float r_ranges[] = { 0, 255 };
	float g_ranges[] = { 0, 255 };
	float b_ranges[] = { 0, 255 };

	const float* ranges[] = { r_ranges, g_ranges, b_ranges };

	// Use the o-th and 1-st channels
	int channels[] = { 0, 1,2 };

	/// Histograms	
	MatND hist_test1;
	MatND hist_test2;

	calcHist( &sub1, 1, channels, Mat(), hist_test1, 3, histSize, ranges, true, false );
	//normalize( hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat() );

	calcHist( &sub2, 1, channels, Mat(), hist_test2, 3, histSize, ranges, true, false );
	//normalize( hist_test2, hist_test2, 0, 1, NORM_MINMAX, -1, Mat() );
	
	return fabs(compareHist( hist_test1, hist_test2, 0 ));		
 }

  void MdiEditor::OpticalFlow(std::vector<Mat>& v1, std::vector<Mat>& v2)
  {
 	 std::vector<cuda::GpuMat> dgray1,dgray2;	
 		
 	 int rows=v1[0].rows;
 	 int cols=v1[0].cols;
 	 int times=v1.size();
 
 	 f1.clear();
 	 f2.clear();
 	 b1.clear();
 	 b2.clear();
 	 for(int i=0;i<times;i++)
 	 {
		 Mat gray1,gray2;
		cvtColor(v1[i], gray1, COLOR_RGB2GRAY);
		cvtColor(v2[i], gray2, COLOR_RGB2GRAY);
  		cuda::GpuMat dg1(gray1);
 		cuda::GpuMat dg2(gray2);
 		dgray1.push_back(dg1);
 		dgray2.push_back(dg2);	 	
 	 }
 	 
	cuda::FarnebackOpticalFlow op;
 	cuda::GpuMat dflow1x(rows,cols,CV_32FC1);
 	cuda::GpuMat dflow1y(rows,cols,CV_32FC1);
 	cuda::GpuMat dflow2x(rows,cols,CV_32FC1);
 	cuda::GpuMat dflow2y(rows,cols,CV_32FC1);
 	
 	 for(int t=0;t<times;t++)
 	 {	
 		 if (t<times-1)
 		 {
 			
 		    op(dgray1[t], dgray1[t+1], dflow1x,dflow1y);
			op(dgray2[t], dgray2[t + 1], dflow2x, dflow2y);
 			
 			Mat flow1x=Mat(dflow1x);			
 			Mat flow1y=Mat(dflow1y);	
 			Mat flow1(rows,cols,CV_32FC2);
 			Mat flow2x=Mat(dflow2x);			
 			Mat flow2y=Mat(dflow2y);	
 			Mat flow2(rows,cols,CV_32FC2);
 		
 			cv::Mat src1[2]={flow1x,flow1y};
 			cv::Mat src2[2]={flow2x,flow2y};
 			int from_to[] = { 0,0,1,1};
 			cv::mixChannels(src1, 2, &flow1, 1, from_to, 2 );
 			cv::mixChannels(src2, 2, &flow2, 1, from_to, 2 );			
 			
 			f1.push_back(flow1);
 			f2.push_back(flow2);
 		 }	 			 
 		 else
 		 {
 			 f1.push_back(Mat::zeros(rows,cols,CV_32FC2));
 			 f2.push_back(Mat::zeros(rows,cols,CV_32FC2));
 		 }	 	
 			
 	 }
 	
 	for(int t=0;t<times;t++)
 	 	{	
 			if (t>0)
 			{
				op(dgray1[t], dgray1[t - 1], dflow1x, dflow1y);
				op(dgray2[t], dgray2[t - 1], dflow2x, dflow2y);
 
 				Mat flow1x=Mat(dflow1x);			
 				Mat flow1y=Mat(dflow1y);	
 				Mat flow1(rows,cols,CV_32FC2);
 				Mat flow2x=Mat(dflow2x);			
 				Mat flow2y=Mat(dflow2y);	
 				Mat flow2(rows,cols,CV_32FC2);
 
 				cv::Mat src1[2]={flow1x,flow1y};
 				cv::Mat src2[2]={flow2x,flow2y};
 				int from_to[] = { 0,0,1,1};
 				cv::mixChannels(src1, 2, &flow1, 1, from_to, 2 );
 				cv::mixChannels(src2, 2, &flow2, 1, from_to, 2 );			
 
 				b1.push_back(flow1);
 				b2.push_back(flow2);
 			}		
 			else
 			{
 				b1.push_back(Mat::zeros(rows,cols,CV_32FC2));
 				b2.push_back(Mat::zeros(rows,cols,CV_32FC2));
 			}
 	}   
 
	op.releaseMemory();
 	dflow1x.release();
 	dflow1y.release();
 	dflow2x.release();
 	dflow2y.release();
 	for(int t=0;t<times;t++)
 	{
 		dgray1[t].release();
 		dgray2[t].release();
 	}
 	
 	dgray1.clear();
 	dgray2.clear();	
 
  }
 

 


 void MdiEditor::NextStage()
 {
	 DeleteThread();
		 //resample
	 if(parameters.cnt.size()>0)
	 {
		 for (int i=0;i<parameters.total_frame;i++)
		 {
			imageEditorA->RenderStage1(resample1[i],0,i);
			imageEditorA->RenderStage1(resample2[i],1,i);
		 }
		 
		//optical flow
		OpticalFlow(resample1,resample2);

	 }
 	


	 //resample points
	 std::vector<std::vector<Conp>>lp=parameters.lp;
	 std::vector<std::vector<Conp>>rp=parameters.rp;
	 std::vector<std::vector<Connect>> cnt=parameters.cnt;

	 parameters.lp.clear();
	 parameters.rp.clear();
	 parameters.cnt.clear();

	 for (int i=0;i<cnt.size();i++)
		 for (int j=0;j<cnt[i].size();j++)
		 {
			 int2 li=cnt[i][j].li;
			 int2 ri=cnt[i][j].ri;

			 Conp elem;
			 elem.p.x=lp[li.x][li.y].p.x;
			 elem.p.y=lp[li.x][li.y].p.y;
			 elem.p.z=(lp[li.x][li.y].p.z+rp[ri.x][ri.y].p.z)/2+0.5;
			 elem.p.w=1;
			 elem.weight=1.0f;
			 parameters.frame0=parameters.frame1=elem.p.z;

			 if (j==0)
			 {
				 std::vector<Conp> list;
				 list.push_back(elem);
				 parameters.lp.push_back(list);
				 parameters.ActIndex_l.x=i;		
				 parameters.ActIndex_l.y=0;	
				 PtModified('l','a',false);
			 }
			 else
			 {
				 parameters.lp[i][elem.p.z]=elem;
				 parameters.ActIndex_l.x=i;		
				 parameters.ActIndex_l.y=elem.p.z;			
				 PtModified('l','m',false);
			 }

			 elem.p.x=rp[ri.x][ri.y].p.x;
			 elem.p.y=rp[ri.x][ri.y].p.y;
			 elem.p.z=(lp[li.x][li.y].p.z+rp[ri.x][ri.y].p.z)/2+0.5;
			 elem.p.w=1;
			 elem.weight=1.0f;

			 if (j==0)
			 {
				 std::vector<Conp> list;
				 list.push_back(elem);
				 parameters.rp.push_back(list);
				 parameters.ActIndex_r.x=i;		
				 parameters.ActIndex_r.y=0;	
				 PtModified('r','a',false);
			 }
			 else
			 {
			 parameters.rp[i][elem.p.z]=elem;
			 parameters.ActIndex_r.x=i;		
			 parameters.ActIndex_r.y=elem.p.z;			
			 PtModified('r','m',false);
			 }

			 if (j==0)
			 {
				 std::vector<Connect> list;				
				 for(int k=0;k<parameters.total_frame;k++)
				 {
					 Connect c;
					 c.li.x=i;
					 c.li.y=k;
					 c.ri.x=i;
					 c.ri.y=k;	
					 list.push_back(c);					
				 }			
				 parameters.cnt.push_back(list);	
			 }		
		 }



		 for(int i=0;i<lp.size();i++)
			 lp[i].clear();
		 lp.clear();
		 for(int i=0;i<rp.size();i++)
			 rp[i].clear();
		 rp.clear();
		 for(int i=0;i<cnt.size();i++)
			 cnt[i].clear();
		 cnt.clear();

	//pyramids
	 pyramid.build(resample1,resample2,f1,f2,b1,b2,parameters.start_res);	 
	 //start
	 match_start();

	 //show
	 parameters.frame0=parameters.frame1=parameters.total_frame/2;
	 ModifyFrameBoth(parameters.frame0);
	 updateALL();
	 
 }

 void MdiEditor::PreviousStage()
 {

	 DeleteThread();
	 for (int i = 0; i<parameters.cnt.size(); i++)
		 parameters.cnt[i].clear();
	 parameters.cnt.clear();
	 for (int i = 0; i<parameters.lp.size(); i++)
		 parameters.lp[i].clear();
	 parameters.lp.clear();
	 for (int i = 0; i<parameters.rp.size(); i++)
		 parameters.rp[i].clear();
	 parameters.rp.clear();
	
	 parameters.ActIndex_l.x = -1;
	 parameters.ActIndex_l.y = -1;
	 parameters.ActIndex_r.x = -1;
	 parameters.ActIndex_r.y = -1;

	 OpticalFlow(video1, video2);
	 pyramid.build(video1, video2, f1, f2, parameters.start_res / 2);
	 thread_flag = 0;

	
	 sync_start();


 }
	
 void MdiEditor::sync_start()
 {
	 if(parameters.cnt.size()>0)
	 {
		 DeleteThread();

		 sync_thread = new CSyncThread(parameters,pyramid);
		 connect(sync_thread,SIGNAL(sigFinished()),this,SLOT(sync_finished()));
		 connect(sync_thread,SIGNAL(sigUpdate()),this,SLOT(SetResults()));
		 sync_thread->start(QThread::HighestPriority);

		 thread_flag=0;
		 SetResults(); 
	 }
	 else
		 sync_finished();
	
 }

 void MdiEditor::sync_finished()
 {
	
	  thread_flag=1;	 	 
	  SetResults();
	  if (_auto)
		  NextStage();
 }



  void MdiEditor::match_start()
  {
	DeleteThread();

	match_thread = new CMatchingThread(parameters,pyramid);
	connect(match_thread,SIGNAL(sigFinished()),this,SLOT(match_finished()));
	connect(match_thread,SIGNAL(sigUpdate()),this,SLOT(SetResults()));
	match_thread->start(QThread::HighestPriority);

	thread_flag=2;
	SetResults();
  }

 void MdiEditor::match_finished()
 {

	thread_flag=3;
 	 
	poison_thread=new CPoissonExt(pyramid);
	connect(poison_thread,SIGNAL(sigFinished()),this,SLOT(poisson_finished()));
	poison_thread->start(QThread::HighestPriority);
	 	
  /*	qpath_thread=new CQuadraticPath(pyramid);
  	connect(qpath_thread,SIGNAL(sigFinished()),this,SLOT(qpath_finished()));
 	qpath_thread->start(QThread::HighestPriority);*/

	//poisson_finished();
	qpath_finished();

	SetResults();

 }
 
void MdiEditor::poisson_finished()
{
	thread_flag++;
	
	SetResults();
}
 void MdiEditor::qpath_finished()
 {
	 thread_flag++;
	
	 SetResults();
 }
 
 void MdiEditor::ShowHalfway()
 {
	 imageEditorM->_flag_error=false;

	 updateALL();
 }
 void MdiEditor::ShowError()
 {
	 imageEditorM->_flag_error=true;

	 updateALL();
 }
 void MdiEditor::ColorFromImage1()
 {
	 imageEditorA->_colorfrom=0;

	 updateALL();

 }
 void MdiEditor::ColorFromImage12()
 {
	 imageEditorA->_colorfrom=1;

	 updateALL();
 }
 void MdiEditor::ColorFromImage2()
 {
	 imageEditorA->_colorfrom=2;
	 updateALL();
 }

 void MdiEditor::AutoQuit()
 {
	 if (_auto)
	 {		 
		 QString filename;

		 filename.sprintf("%s\\time.txt", pro_path.toLatin1().data());

		 QFile file(filename.toLatin1().data());

		 if (file.open(QFile::WriteOnly | QFile::Truncate))
		 {
			 QTextStream out(&file);
			 QString line;
			 if (sync_thread)
			 {
				 line.sprintf("Synch time: %f s \n", sync_thread->run_time);
				 out << line;
			 }
			 if (match_thread)
			 {
				 line.sprintf("Optimizing time: %f s \n", match_thread->run_time);
				 out << line;
			 }
			 if (poison_thread)
			 {
				 line.sprintf("Poisson time: %f s \n", poison_thread->_runtime);
				 out << line;
			 }
			/* line.sprintf("Quadratic path time: %f ms \n", qpath_thread->_runtime);
			 out << line;*/
			
		 }


		 file.flush();
		 file.close();
		 clear();
		 _app->quit();

	 }
 }
