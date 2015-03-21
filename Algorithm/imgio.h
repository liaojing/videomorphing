#ifndef GPUMORPH_IMGIO_H
#define GPUMORPH_IMGIO_H

#include <string>
#include <util/dimage_fwd.h>
#include <util/image_ops.h>
#include <opencv2/core/wimage.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

template <class T>
void save(const std::string &fname, const rod::dimage<T> &img);

template <class T>
void load(rod::dimage<T> &img, const std::string &fname);

#endif
