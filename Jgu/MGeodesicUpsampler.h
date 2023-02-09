// Copyright (C) 2013,2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "MGeodesicDT.h"

class MGeodesicUpsampler {
public:
	static void MultiDimSmooth(
		std::vector<cv::Mat1f> &out, const cv::Mat3b &img, std::vector<cv::Mat1f> &inm,
		const float sigma, const float lambda1, const float lambda2, const int interval, const unsigned int algIter=1);

	//! Main filtering step
	static void Process(cv::Mat1f &out, const cv::Mat3b &img, const cv::Mat1f &dsDepth, const cv::Mat1b &mask,
		const int upsamplingRatio,const float sigma,const float lambda1,const float lambda2,const int interval,const unsigned int maxIter=10);
	//! Preprocessing step for demonstration purpose only
	static void Decimate(cv::Mat1f &decimatedDisp, cv::Mat1b &mask, const cv::Mat1f &inputDepth,
		const int upsamplingRatio, const float badVal=0);
	static void PrepareDecimatedMap(cv::Mat1f &decimatedDisp, cv::Mat1b &mask, cv::Mat3b &img, const cv::Mat1f &inputDepth,
		const int upsamplingRatio, const float badVal = 0);
private:
	//! Multiplex the low resolution depth image into several channels.
	static void Multiplex(std::vector<cv::Mat1i> internalLabels,const cv::Mat1b &dsDepth,const cv::Mat1b &mask,int interval);
	//! Compute the edge weights between neighboring pixels
	static void ComputeWeight(float *dists,const cv::Mat3b &img,const float lambda1,const float lambda2);
};
