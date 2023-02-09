// Copyright (C) 2013,2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later
#include "MGeodesicUpsampler.h"

void MGeodesicUpsampler::MultiDimSmooth(std::vector<cv::Mat1f> &out, const cv::Mat3b &img, std::vector<cv::Mat1f> &inm, const float sigma, const float lambda1, const float lambda2, const int interval, const unsigned int algIter/*=1*/)
{
	// Number of internal maps
	const int nMaps = interval*interval;
	const float tSigma = sigma;
	// Distance between two neighboring pixels
	float *dists = new float[img.rows*img.cols * 8];
	// For geodesic distance transform
	std::vector< cv::Mat1i > internalLabels;
	std::vector< cv::Mat1f > gDT;
	internalLabels.resize(nMaps);
	gDT.resize(nMaps);
	for (int i = 0; i < nMaps; i++) {
		internalLabels[i] = cv::Mat1i::zeros(img.rows + 2, img.cols + 2);
		gDT[i] = cv::Mat1f::zeros(img.rows + 2, img.cols + 2);
	}
	ComputeWeight(dists, img, lambda1 / (255.0f), (lambda2));
	cv::Mat1b mask = cv::Mat1b::ones(img.rows, img.cols);
	Multiplex(internalLabels, mask, mask , interval);
#pragma omp parallel for schedule (dynamic)
	for (int i = 0; i < nMaps; i++) {	//Geodesic distance transform
		MGeodesicDT::Process(gDT[i], internalLabels[i], dists);
	}
	out.resize(inm.size());
	for (int f = 0; f < inm.size(); f++)
		out[f] = cv::Mat1f::zeros(img.rows, img.cols);
#pragma omp parallel for schedule (dynamic)
		for (int f = 0; f < inm.size(); f++) {
			for (int y = 0; y < img.rows; y++) {
				for (int x = 0; x < img.cols; x++) {
					std::vector<float> vals;
					std::vector<float> dds;
					float minDist = FLT_MAX;
					for (int i = 0; i<nMaps; i++) {
						// Index the pixel ID in the original depth map
						int idx = internalLabels[i](y + 1, x + 1) - 1;
						int cx = idx%img.cols;
						int cy = idx/img.cols;
						float curDist = gDT[i](y + 1, x + 1);
						if (curDist > 0)
						{
							vals.push_back(inm[f](cy, cx));
							dds.push_back(curDist / tSigma);
							// update the minimal distance
							if (curDist / tSigma < minDist)
								minDist = curDist / tSigma;
						}
					}
					float weiTmp = 0;
					float valTmp = 0;
					for (int i = 0; i < vals.size(); i++) {
						dds[i] = dds[i] - minDist;
						weiTmp += exp(-dds[i] * dds[i] / 2.0f);
						valTmp += exp(-dds[i] * dds[i] / 2.0f)*vals[i];
					}
					out[f](y, x) = std::exp(std::log(std::max(valTmp,FLT_MIN)) - std::log(weiTmp));
				}
			}
		}
	//}
	delete[] dists;
}


/*!
 * Main filtering function.
 * Note that we adapt the sigma value based on the upsampling ratio.
 * 1. computer edge weights
 * 2. create inteval*interval sub-sampled depth maps.
 * 3. computer geodesic distance transforms
 * 4. geodesic filtering
 *
 * Note that all the bad values should be marked as zero in the mask.
 */
void MGeodesicUpsampler::Process(cv::Mat1f &out, const cv::Mat3b &img, const cv::Mat1f &dsDepth, const cv::Mat1b &mask, const int upsamplingRatio, const float sigma, const float lambda1, const float lambda2, const int interval, const unsigned int maxIter) {
	// Number of internal maps
	const int nMaps = interval*interval;
	const float tSigma = sigma;
	// Distance between two neighboring pixels
	float *dists = new float [dsDepth.rows*dsDepth.cols*8];
	// For geodesic distance transform
	std::vector< cv::Mat1i > internalLabels;
	std::vector< cv::Mat1f > gDT;
	internalLabels.resize( nMaps);
	gDT.resize( nMaps);
	for(int i=0;i<nMaps;i++) {
		internalLabels[i] = cv::Mat1i::zeros(img.rows+2,img.cols+2);
		gDT[i] = cv::Mat1f::zeros(img.rows+2,img.cols+2);
	}
	ComputeWeight(dists,img,lambda1/(255.0f),(lambda2/upsamplingRatio));
	Multiplex(internalLabels,dsDepth,mask,interval);
#pragma omp parallel for schedule (dynamic)
	for(int i=0;i<nMaps;i++) {	//Geodesic distance transform
		MGeodesicDT::Process(gDT[i],internalLabels[i],dists,maxIter);
	}
#pragma omp parallel for schedule (dynamic)
	for(int y=0;y<img.rows;y++) {
		for(int x=0;x<img.cols;x++) {
			//std::vector<float> vals(nMaps);
			//std::vector<float> dds(nMaps);
			std::vector<float> vals;
			std::vector<float> dds;
			float minDist = FLT_MAX;
			for(int i=0;i<nMaps;i++) {
				// Index the pixel ID in the original depth map
				int idx = internalLabels[i](y+1,x+1)-1;
				int cx = idx%img.cols;
				int cy = idx/img.cols;
				float curDist = gDT[i](y + 1, x + 1);
				if (curDist > 0) {
					vals.push_back(dsDepth(cy, cx));
					dds.push_back(curDist / tSigma);
					// update the minimal distance
					if (curDist / tSigma < minDist)
						minDist = curDist / tSigma;
				}
				//vals[i] = dsDepth(cy,cx);
				//dds[i] = curDist/tSigma;
				//// update the minimal distance
				//if(dds[i] < minDist)
				//	minDist = dds[i];
			}
			double weiTmp = 0;
			double valTmp = 0;
			for (int i = 0; i<vals.size(); i++) {
				dds[i] = dds[i] - minDist;
				weiTmp += exp(-dds[i]*dds[i]/2.0);
				valTmp += exp(-dds[i]*dds[i]/2.0)*vals[i];
			}
			out(y,x) = (float)std::exp( std::log(valTmp) - std::log(weiTmp) );
		}
	}
	delete [] dists;
}

/*!
 * Decimate the original full resolution depth image.
 * This function creates a decimated depth map from the original full
 * resolution depth map. Basically, it first smoothes the orignal full
 * resolution depth map and then subsampled it by taking every
 * $(upsamplingRatio)-apart pixels both in the vertical and horizontal
 * directions. Though the downsampled depth map has a lower resolution, it
 * has the same dimension as the original image since we keep the subsampled
 * depth values in the original pixel locations. The mask image simply denotes
 * which pixels in the downsampledDepth image have depth values.
 */
void MGeodesicUpsampler::Decimate(cv::Mat1f &decimatedDisp, cv::Mat1b &mask, const cv::Mat1f &inputDepth, const int upsamplingRatio, const float badVal) {
	const int width  = inputDepth.cols;
	const int height = inputDepth.rows;
	const int lowResWidth = (int)ceil(width*1.0 / upsamplingRatio);
	const int lowResHeight = (int)ceil(height*1.0 / upsamplingRatio);
	const int usr = upsamplingRatio;
	mask.release();
	mask = cv::Mat1b::zeros(height,width);
	decimatedDisp.release();
	decimatedDisp.create(height,width);
	for(int y=0;y<lowResHeight;y++) {
		for(int x=0;x<lowResWidth;x++)	{
			if( inputDepth(y*usr,x*usr) != badVal) {
				mask(y*usr,x*usr) = 255;
				decimatedDisp(y*usr,x*usr) = inputDepth(y*usr,x*usr);
			}
		}
	}
};

void MGeodesicUpsampler::PrepareDecimatedMap(cv::Mat1f &decimatedDisp, cv::Mat1b &mask, cv::Mat3b &img, const cv::Mat1f &inputDepth, const int upsamplingRatio, const float badVal) {
	const int width = inputDepth.cols*upsamplingRatio;
	const int height = inputDepth.rows*upsamplingRatio;
	const int lowResWidth = inputDepth.cols;
	const int lowResHeight = inputDepth.rows;
	const int usr = upsamplingRatio;
	mask.release();
	mask = cv::Mat1b::zeros(height, width);
	decimatedDisp.release();
	decimatedDisp.create(height, width);

	for (int y = 0; y<lowResHeight; y++) {
		for (int x = 0; x<lowResWidth; x++)	{
			if (inputDepth(y, x) != badVal) {
				mask(y*usr, x*usr) = 255;
				decimatedDisp(y*usr, x*usr) = inputDepth(y, x);
			}
		}
	}

}

/*!
 * ComputeWeight
 * Compute the distance between two neighboring pixels. Note that we use
 * an 8-connected graph. We allocate twice the amount of memory necessary for
 * storing the distance. This is done this way for easing the memory indexing
 * during computing geodesic distance transform.
 */
void MGeodesicUpsampler::ComputeWeight(float *dists,const cv::Mat3b &img,const float lambda1,const float lambda2){
	//  0  1  2
	//  3  *  4
	//  5  6  7
	const int xShifts[] = {-1, 0, 1,-1,1,-1,0,1};
	const int yShifts[] = {-1,-1,-1, 0,0, 1,1,1};
	const int dd[] = {2,1,2,1,1,2,1,2};
	const int w = img.cols;
	const int h = img.rows;
	const float lambda1SQ=lambda1*lambda1;
	const float lambda2SQ=lambda2*lambda2;
	memset(dists,0,sizeof(float)*h*w*8);
#pragma omp parallel for schedule (dynamic)
	for(int y=0;y<h;y++) {
		for(int x=0;x<w;x++) {
			int ref=8*(x+y*w);
			const cv::Vec3f cur = (cv::Vec3d)img(y,x);
			for(int i=4;i<8;i++) {
				// if it is within the image boundary
				if( (x+xShifts[i])>=0 && (x+xShifts[i])<w && (y+yShifts[i])>=0 && (y+yShifts[i])<h )
				{
					const cv::Vec3f diff = ((cv::Vec3f)img(y+yShifts[i],x+xShifts[i])) - cur;
					const float dprod = (float)diff.ddot(diff);
					dists[i+ref] = cv::sqrt( lambda2SQ*dd[i] + lambda1SQ*dprod );
					const int opref = 8*( (x+xShifts[i]) + (y+yShifts[i])*w );
					dists[(7-i)+opref] = dists[i+ref];
				}
			}
		}
	}
}

/*!
 * Multiplexing function
 * This function decomposes the low resolution depth map into
 * $(interval*interval) channels. This is achieved by creating
 * $(interval*interval) masks whose pixels are either 0 or some positive
 * numbers, referring to the pixel IDs in the original low resolution depth map.
 */
void MGeodesicUpsampler::Multiplex(std::vector<cv::Mat1i> internalLabels,const cv::Mat1b &dsDepth,const cv::Mat1b &mask,int interval) {
	int idx;
	int xIdx=0,yIdx=0,hit=0; // Use to locate the right channel.
	for(int y=0;y<dsDepth.rows;y++) {
		hit = xIdx = 0;
		for(int x=0;x<dsDepth.cols;x++) {// whenever encountering a depth pixel
			if(mask(y,x)>0) {
				hit=1;
				xIdx++;
				// find out which channel to write
				idx = xIdx%interval + (yIdx%interval)*interval;
				internalLabels[idx](y+1,x+1) = 1+x+y*dsDepth.cols;
			}
		}
		// A row can have no depth pixels. We only increase the counter when it have some.
		if(hit==1) yIdx++;
	}
}
