#ifndef GPUMORPH_MORPH_H
#define GPUMORPH_MORPH_H
#include "parameters.h"
#include "pyramid.h"

class Pyramid;
struct PyramidLevel;


class Morph
{
public:
	Morph(Parameters &params, Pyramid &pyramid, bool& run_flag);
	~Morph();

	const Parameters &params();	
	bool calculate_halfway_parametrization();

	int _total_l,_current_l;
	float _total_iter,_current_iter,_max_iter;

private:	
	bool& m_cb;	
	Pyramid &m_pyramid;
	Parameters &m_params;

	void cpu_optimize_level(PyramidLevel &lvl,PyramidLevel &lv0);	
	void initialize_level(PyramidLevel &lvl,PyramidLevel &lv0);
	void optimize_level(PyramidLevel &lvl) ;
	void clear_level(PyramidLevel &lvl);
};


void downsample(rod::dimage<float> &dest, const rod::dimage<float> &orig);
void upsample(PyramidLevel &dest, PyramidLevel &orig);


void render_halfway_image(rod::dimage<float3> &out, PyramidLevel &lvl,
						  const rod::dimage<float3> &in0,
						  const rod::dimage<float3> &in1);

void render_halfway_image(rod::dimage<float3> &out,
						  const rod::dimage<float2> &hwpar,
						  const rod::dimage<float3> &in0,
						  const rod::dimage<float3> &in1);



#endif
