#include <cstring> 
#include <cctype> 
#include "extension.h"

#include "image.h"
#include "error.h"
static extension::clamp clamp;

namespace image {
	int load(image::rgba<float> *rgba, float *data, int w, int h) {
		if (h!=0&&w!=0)
		{
			rgba->resize(h, w);
			const float tof = (1.f/255.f);
			#pragma omp parallel for
			for (int i = h-1; i >= 0; i--) {
				for (int j = 0; j < w; j++) {
					int p = i*w+j; // flip image so y is up
					rgba->r[p] = color::srgbuncurve(data[p*3+0]*tof);
					rgba->g[p] = color::srgbuncurve(data[p*3+1]*tof);
					rgba->b[p] = color::srgbuncurve(data[p*3+2]*tof);
					rgba->a[p] = 1.0f;
				}
			}
				return 1;
		}
		else
		{
			return 0;
		}       
    }

	int load(image::rgba<float> *rgba, float *data, int w, int h, int rowstride,float min,float max) {
		if (h!=0&&w!=0)
		{
			rgba->resize(h, w);
			const float tof = (1.f/(max-min));
			#pragma omp parallel for
			for (int i = h-1; i >= 0; i--) {
				for (int j = 0; j < w; j++) {
					int p = i*w+j; // flip image so y is up
					rgba->r[p] = color::srgbuncurve((data[(i*rowstride+j)*2+0]-min)*tof);
					rgba->g[p] = color::srgbuncurve((data[(i*rowstride+j)*2+1]-min)*tof);
					rgba->b[p] = 1.0f;
					rgba->a[p] = 1.0f;
				}
			}
			return 1;
		}
		else
		{
			return 0;
		}       
	}

  	int store(float *data, const image::rgba<float> &rgba) {
        
		int height=rgba.height();
		int width=rgba.width(); 
		#pragma omp parallel for
		for (int i = height-1; i >= 0; i--) {
			for (int j = 0; j < width; j++) {
				int p = i*width+j; // flip image so y is up
				data[p*3+0] = color::srgbcurve(clamp(rgba.r[p]))*255;
				data[p*3+1] = color::srgbcurve(clamp(rgba.g[p]))*255;
				data[p*3+2] = color::srgbcurve(clamp(rgba.b[p]))*255;						
			}
		}		
        return 1;
	}

	int store(float *data, const image::rgba<float> &rgba,int rowstride,float min,float max) {

		int height=rgba.height();
		int width=rgba.width(); 
		#pragma omp parallel for
		for (int i = height-1; i >= 0; i--) {
			for (int j = 0; j < width; j++) {
				int p = i*rowstride+j; // flip image so y is up
				data[(i*rowstride+j)*2+0] = color::srgbcurve(clamp(rgba.r[p]))*(max-min)+min;
				data[(i*rowstride+j)*2+1] = color::srgbcurve(clamp(rgba.g[p]))*(max-min)+min;									
			}
		}		
		return 1;
	}

	int store_gray(float *data, const image::rgba<float> &rgba) {

		int height=rgba.height();
		int width=rgba.width(); 
		#pragma omp parallel for
		for (int i = height-1; i >= 0; i--) {
			for (int j = 0; j < width; j++) {
				int p = i*width+j; // flip image so y is up
				float r = color::srgbcurve(clamp(rgba.r[p]))*255;
				float g = color::srgbcurve(clamp(rgba.g[p]))*255;
				float b = color::srgbcurve(clamp(rgba.b[p]))*255;		

				data[p]= r*0.299 + g*0.587 + b*0.114;
			}
		}		
		return 1;
	}
}
