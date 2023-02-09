// Copyright (C) 2013,2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
/*!
* Perform geodesic distance transform
* This function does the followings. It first initializes all the pixels to
* have infinite geodesic distances. It then performs dynamic programming
* update until convergence.
*/
class MGeodesicDT
{
public:
	static void Process(cv::Mat1f &out, cv::Mat1i &label, const float *dists, const unsigned int maxIter = 100);
private:
	static void Pass(cv::Mat1f &out, cv::Mat1i &label, int &change, const float *dists);
};
