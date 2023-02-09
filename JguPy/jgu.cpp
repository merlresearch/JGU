// Copyright (C) 2013,2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <boost/python.hpp>
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"
#include <numpy/ndarrayobject.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <stdlib.h>
#include "MGeodesicUpsampler.h"

using namespace boost::python;
using namespace std;
class JguPy {
public:
	JguPy() { }
	/*
	* upsampleRate : upsampleing rate
	* sigma : the bandwidth parameter for the Gaussian kernel
	* lambda1 : the weighting factor on the color difference.
	* lambda2 : the weighting factor on the spatial distance.
	* interval : approximate optimization parameter.
	*/
	boost::python::object  Upsample(
		boost::python::numeric::array &rgbImg,
		boost::python::numeric::array &lowResTargetImg,
		int upsampleRate,
		int interval,
		double sigma,
		double lambda1)
	{
		double lambda2 = 1.0;
		PyArrayObject* pyRGBImg = (PyArrayObject*)PyArray_FROM_O(rgbImg.ptr());
		int tHeight = *(pyRGBImg->dimensions);
		int tWidth = *(pyRGBImg->dimensions + 1);
		cv::Mat cvRGBImg(cv::Size(tWidth, tHeight), CV_8UC3, (uchar *)pyRGBImg->data);

		PyArrayObject* pyLowResTargetImg = (PyArrayObject*)PyArray_FROM_O(lowResTargetImg.ptr());
		int height = *(pyLowResTargetImg->dimensions);
		int width = *(pyLowResTargetImg->dimensions + 1);
		cv::Mat cvDepth(cv::Size(width, height), CV_8UC1, (uchar *)pyLowResTargetImg->data);

		cv::Mat1f depthF;
		cvDepth.convertTo(depthF, CV_32F);
		cv::Mat1f upsampledDepthF = cv::Mat1f::zeros(tHeight, tWidth);
		cv::Mat1f downsampledDepth;
		cv::Mat1b mask;
		if (height == tHeight&& width == tWidth) {
			//std::cout << "Perform decimation\n";
			MGeodesicUpsampler::Decimate(downsampledDepth, mask, depthF, upsampleRate);
		}
		else if (height*upsampleRate == tHeight&&width*upsampleRate == tWidth) {
			//std::cout << "Donw-sampled input\n";
			MGeodesicUpsampler::PrepareDecimatedMap(downsampledDepth, mask, (cv::Mat3b&)cvRGBImg, depthF, upsampleRate);
		}
		else {
			std::cout << "Incorrect input image size.\n";
			return boost::python::make_tuple(-1);
		}
		MGeodesicUpsampler::Process(upsampledDepthF, (cv::Mat3b&)cvRGBImg, downsampledDepth, mask, upsampleRate, sigma, lambda1, lambda2, interval);
		cv::Mat1b upsampledDepth;
		upsampledDepthF.convertTo(upsampledDepth, CV_8U);
		//cv::imshow("output", upsampledDepth);
		//cv::waitKey(0);
		//cv::imshow("input", cvRGBImg);
		//cv::imshow("depth", cvDepth);
		//cv::waitKey();
		int ndims = 2;
		npy_intp dims[2];
		dims[0] = tHeight; dims[1] = tWidth;
		PyObject * pyObj = PyArray_SimpleNewFromData(ndims, dims, NPY_UBYTE, upsampledDepth.data);
		boost::python::handle<> handle(pyObj);
		boost::python::numeric::array arr(handle);
		return boost::python::make_tuple(arr.copy());
	}
};



BOOST_PYTHON_MODULE(JguPy)
{
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
	import_array();
	class_<JguPy>("jgu")
		.def("Upsample", &JguPy::Upsample)
		;
}
