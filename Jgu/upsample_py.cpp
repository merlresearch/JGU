// Copyright (C) 2013,2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <string>
#include "MGeodesicUpsampler.h"

/*!
 * USAGE: upsample(outputName,imgName,depthName,upsampleRate,sigma,lambda1,lambda2,interval)
 *
 * outputName   : output superresoluted depth image name
 * imgName      : input color image name
 * depthName    : input depth image name
				: Note that the input image should have the sample resolution as the color image.
				: This can be simply achieved by using bilinear interporlation.
 * upsampleRate : upsampleing rate
 * sigma        : the bandwidth parameter for the Gaussian kernel
 * lambda1      : the weighting factor on the color difference.
 * lambda2      : the weighting factor on the spatial distance.
 * interval     : approximate optimization parameter.
 *
 */
int main(int argc,const char **argv)
{
	int upsampleRate = 4;
	int interval     = 3;
	double sigma     = 0.5;
	double lambda1   = 10.0;
	double lambda2   =  1.0;
	std::string outputName;
	std::string imgName;
	std::string depthName;

	if(argc!=9)	{
		std::cout << "* upsample(outputName,imgName,depthName,upsampleRate,sigma,lambda1,lambda2,interval)\n";
		std::cout << "* \n";
		std::cout << "* outputName\t: output superresoluted depth image name.\n";
		std::cout << "* imgName\t: input color image name.\n";
		std::cout << "* depthName\t: input depth image name.\n";
		std::cout << "* upsampleRate\t: upsampleing rate.\n";
		std::cout << "* sigma\t\t: the bandwidth parameter for the Gaussian kernel.\n";
		std::cout << "* lambda1\t: the weighting factor on the color difference.\n";
		std::cout << "* lambda2\t: the weighting factor on the spatial distance.\n";
		std::cout << "* interval\t: approximate optimization parameter.\n\n";
		//return -1;

		outputName = std::string("out.png");
		imgName = std::string("view5.png");
		depthName = std::string("disp5.png");
		//depthName = std::string("disp5_4ds.png");
	}
	else {
		outputName   = std::string( argv[1]);
		imgName      = std::string( argv[2]);
		depthName    = std::string( argv[3]);
		upsampleRate = atoi(argv[4]);
		sigma        = atof(argv[5]);
		lambda1      = atof(argv[6]);
		lambda2      = atof(argv[7]);
		interval     = atoi(argv[8]);
	}
	std::cout<<"output depth image name = "<<outputName<<std::endl;
	std::cout<<"input RGB image name = "<<imgName<<std::endl;
	std::cout<<"input depth image name = "<<depthName<<std::endl;
	std::cout<<"upsampling rate = "<<upsampleRate<<std::endl;
	std::cout<<"sigma = "<<sigma<<std::endl;
	std::cout<<"lambda1 = "<<lambda1<<std::endl;
	std::cout<<"lambda2 = "<<lambda2<<std::endl;
	std::cout<<"interval = "<<interval<<std::endl;
	cv::Mat3b img   = cv::imread(imgName,1);
	cv::Mat1b depth = cv::imread(depthName,0);
	cv::Mat1f depthF;
	depth.convertTo(depthF, CV_32F);
	cv::Mat1f upsampledDepthF = cv::Mat1f::zeros(img.rows, img.cols);
	cv::Mat1f downsampledDepth;
	cv::Mat1b mask;
	if (depth.rows == img.rows&&depth.cols == img.cols) {
		std::cout << "Perform decimation\n";
		MGeodesicUpsampler::Decimate(downsampledDepth, mask, depthF, upsampleRate);
	}
	else if (depth.rows*upsampleRate == img.rows&&depth.cols*upsampleRate == img.cols) {
		MGeodesicUpsampler::PrepareDecimatedMap(downsampledDepth, mask, img, depthF, upsampleRate);
	}
	else {
		std::cout << "Incorrect input image size.\n";
		return 0;
	}
	{
		boost::timer::auto_cpu_timer t;
		MGeodesicUpsampler::Process(upsampledDepthF, img, downsampledDepth, mask, upsampleRate, sigma, lambda1, lambda2, interval);
		std::cout << "Time:";
	}
	std::cout<<"Write output image to "<<outputName<<std::endl;
	cv::Mat1b upsampledDepth;
	upsampledDepthF.convertTo(upsampledDepth, CV_8U);
	cv::imwrite(outputName,upsampledDepth);
	cv::imshow("output",upsampledDepth);
	cv::waitKey(0);
	return 0;
}
