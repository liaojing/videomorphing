TEMPLATE      = app
HEADERS       = Header.h \
		UI/ImageEditor.h \
        UI/MdiEditor.h \
		UI/HalfwayImage.h \
   		UI/ExternalThread.h \
		UI/RenderWidget.h \		
		UI/DlgPara.h \
		UI/CtrBar.h \
    	Algorithm/MatchingThread.h \  
		Algorithm/QuadraticPath.h \   
		Algorithm/Pyramid.h \    
		Algorithm/PoissonExt.h \   
		Algorithm/parameters.h \    
		Algorithm/SyncThread.h \    				
		
		
SOURCES       =  main.cpp \
		UI/MdiEditor.cpp \               
        UI/ImageEditor.cpp \
		UI/HalfwayImage.cpp \
   		UI/ExternalThread.cpp \
		UI/RenderWidget.cpp \
		UI/DlgPara.cpp \
		UI/CtrBar.cpp \
		Algorithm/MatchingThread.cpp \
		Algorithm/QuadraticPath.cpp \  
		Algorithm/PoissonExt.cpp \    
		Algorithm/SyncThread.cpp \    		 		
		
FORMS		  = UI/DlgPara.ui \
		    UI/CtrBar.ui

RESOURCES      =  UI/CtrBar.qrc

QT += xml
QT += opengl
QT += widgets
CONFIG += 64bit