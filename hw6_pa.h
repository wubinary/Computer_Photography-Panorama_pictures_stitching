#ifndef HW6_PA_H
#define HW6_PA_H

class CylindricalPanorama 
{
public:
	virtual bool makePanorama(
	std::vector<cv::Mat>& img_vec,cv::Mat& img_out,double f
	) = 0;
};

#endif
