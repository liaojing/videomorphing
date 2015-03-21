#pragma once

//QT
#include <QtWidgets>
#include <QApplication>
#include <QFile>
#include <QTextStream>
#include <QString>
#include <QTimer>
#include <QPoint>
#include <QThread>
#include <QDockWidget>
#include <QMdiArea>
#include <QToolBar>
#include <QAction>
#include <QStatusBar>
#include <QMouseEvent>
#include <QMessageBox>
#include <QDir>
#include <QFileDialog>
#include <QtXml>
#include <QSlider>


//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
using namespace cv;

//openmp
#include <omp.h>


//mkl
#include "mkl.h"

#include "time.h"

#define MAX_FRAME 100
#define Max_DIM 600

//para
#include "..\Algorithm\parameters.h"