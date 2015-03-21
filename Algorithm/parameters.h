#ifndef GPUMORPH_PARAMETERS_H
#define GPUMORPH_PARAMETERS_H

#include <string>
#include <vector>
#include <vector_types.h>
#include <opencv2/core/core.hpp>

enum BoundaryCondition
{
    BCOND_NONE,
    BCOND_CORNER,
    BCOND_BORDER
};

struct Connect
{
	int2 li;
	int2 ri;
};

struct Conp
{
	int4 p;
	float weight;
};


struct Parameters
{
	int frame0,frame1;
	int2 range0,range1;
	int total_frame;

    float w_ui, w_tps, w_ssim,w_temp;
    float ssim_clamp;
    float eps;

    int max_iter;
    int start_res;
    float max_iter_drop_factor;
	 
    BoundaryCondition bcond;

    std::vector<std::vector<Conp>>lp;
	std::vector<std::vector<Conp>>rp;
	std::vector<std::vector<Connect>> cnt;

    int2 ActIndex_l,ActIndex_r;

	bool verbose;
};

struct KernParameters/*{{{*/
{
	KernParameters() {}
	KernParameters(const Parameters &p)
		: w_temp(p.w_temp)
		, w_ui(p.w_ui)
		, w_tps(p.w_tps)
		, w_ssim(p.w_ssim)
		, ssim_clamp(p.ssim_clamp)
		, eps(p.eps)
		, bcond(p.bcond)
	{
	}

	float w_temp,w_ui, w_tps, w_ssim;
	float ssim_clamp;
	float eps;
	BoundaryCondition bcond;
};/*}}}*/


#endif

