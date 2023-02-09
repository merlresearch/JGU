// Copyright (C) 2013,2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "MGeodesicDT.h"

void MGeodesicDT::Process(cv::Mat1f &out, cv::Mat1i &label, const float *dists, const unsigned int maxIter)
{
	for(int y=0;y<label.rows;y++) {
		for(int x=0;x<label.cols;x++) {
			out(y, x) = (label(y, x)>0) ? 0 : FLT_MAX;
		}
	}
	unsigned int count=0;
	int change=1;
	while(change && count < maxIter) {
		Pass(out,label,change,dists);
		count++;
	}
	//std::cout<<"iteration count = # "<<count<<"\n";
};

void MGeodesicDT::Pass(cv::Mat1f &out, cv::Mat1i &label, int &change, const float *dists)
{
	int imgWidth = out.cols-2;
	int ref;
	int minIdx;
	float curVal;
	float newVal;
	const int xShifts[] = {-1, 0, 1,-1,1,-1,0,1};
	const int yShifts[] = {-1,-1,-1, 0,0, 1,1,1};
	change = 0;
	// forward pass
	for(int y=1;y<out.rows-1;y++) {
		for(int x=1;x<out.cols-1;x++) {
			ref = 8*((x-1) + (y-1)*imgWidth);
			curVal = out(y,x);
			minIdx = -1;
			for(int i=0;i<4;i++) {
				//if(dists[i+ref]<0) {
				//	std::cout<<"ooop";
				//}
				newVal = out(y+yShifts[i],x+xShifts[i]) + dists[i+ref];
				if( newVal < curVal ) {
					curVal = newVal;
					minIdx = i;
				}
			}
			// If we find a smaller value.
			if( minIdx!=-1 ) {
				label(y,x)  = label(y+yShifts[minIdx],x+xShifts[minIdx]);
				out(y,x) = curVal;
				change++;
			}
		}
	}

	// backward pass
	for(int y=out.rows-2;y>=1;y--) {
		for(int x=out.cols-2;x>=1;x--) {
			ref = 8*((x-1) + (y-1)*imgWidth);
			curVal = out(y,x);
			minIdx = -1;
			for(int i=4;i<8;i++) {
				newVal = out(y+yShifts[i],x+xShifts[i]) + dists[i+ref];
				if( newVal < curVal ) {
					curVal = newVal;
					minIdx = i;
				}
			}
			// If we find a smaller value.
			if( minIdx!=-1 ) {
				label(y,x)  = label(y+yShifts[minIdx],x+xShifts[minIdx]);
				out(y,x) = curVal;
				change++;
			}
		}
	}
};
