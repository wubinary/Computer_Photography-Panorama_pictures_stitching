#include<stdio.h>
#include<string>
#include<bits/stdc++.h>
#include<opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include"hw6_pa.h"

using namespace std;
using namespace cv;

void readFiles(double&,vector<Mat>&);

class Panorama0084:public CylindricalPanorama {
	private:

	public:
		//圖片轉成柱狀座標
		Mat cylinder(Mat &img, double f);
		
		//計算homography matrix (單應矩陣) 
		Mat homography(Mat &image_src,Mat &image_dst);
		
		//兩張圖片拼接
		Mat concat(Mat &image_src,Mat  &image_dst, Mat &H_transfer);
	
		//全景圖片拼接
		bool makePanorama(std::vector<cv::Mat>& img_vec,cv::Mat& img_out,double f);
};

int main(){
	cout<<CV_VERSION<<endl;

	double focal;
	vector<Mat> images; Mat image_out;
	Panorama0084 panorama;

	readFiles(focal, images);
	panorama.makePanorama(images,image_out,focal);
	
	imwrite("/home/wubinray/Desktop/hw/computerPhotography/lab6/out.JPG",image_out);

	return 0;
}

bool Panorama0084::makePanorama(std::vector<cv::Mat>& img_vec,cv::Mat& img_out,double f){

	
	//把圖片轉乘柱狀座標
	vector<Mat>img_cylinder;
	for(int i=0;i<(int)img_vec.size();i++){
		img_cylinder.push_back(this->cylinder(img_vec[i],f));
	}
	
	//圖片拼接	
	Mat img_concat=img_cylinder[img_cylinder.size()-1];
	for(int i=(int)img_cylinder.size()-2;i>=0;i--){
		Mat h = this->homography(img_concat,img_cylinder[i]);
		img_concat = this->concat(img_concat,img_cylinder[i],h);
		cout<<i<<"張圖片完成"<<endl;
	}
	
	img_out = img_concat;
		
	return true;
}
Mat Panorama0084::cylinder(Mat &img, double f){
	Mat output;
	int cols = (int)2 * f * atan(0.5*img.cols / f);
	int rows = (int)img.rows;
	output.create(rows, cols, CV_8UC3);
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){	
			int x = (int)(f * tan((float)(j - cols * 0.5) / f) + img.cols*0.5);
			int y = (int)((i - 0.5*rows)*sqrt(pow(x - img.cols*0.5, 2) + f*f) / f + 0.5*img.rows);
			
			if (0 <= x && x < img.cols && 0 <= y && y < img.rows)
				output.at<Vec3b>(i, j) = img.at<Vec3b>(y, x);
			else
				output.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
		}
	}
	return output;
}
Mat Panorama0084::homography(Mat &image_src, Mat &image_dst){
	Mat image_1,image_2;
	//cvtColor(img_vec[0], image_1, CV_BGR2GRAY);
	//cvtColor(img_vec[2], image_2, CV_BGR2GRAY);
	image_1 = image_src;
	image_2 = image_dst;

	// [特徵辨識器]
	int minHessian = 400;
	SurfFeatureDetector detector( minHessian );
	SurfDescriptorExtractor extractor;	
	//minHessian = 100000;
	//SiftFeatureDetector detector( minHessian ); //(feature threshold, threshold to reduce)
	//SiftDescriptorExtractor extractor;

	// [找出圖片上的所有的features]
	Mat descriptor_src,descriptor_dst;
	vector<cv::KeyPoint> keypoints_src,keypoints_dst;

	detector.detect(image_src, keypoints_src);
	detector.detect(image_dst, keypoints_dst);
	extractor.compute(image_src,keypoints_src,descriptor_src);
	extractor.compute(image_dst,keypoints_dst,descriptor_dst);

	// [兩張圖片做feature匹配]
	vector<DMatch> matches,good_matches;

	FlannBasedMatcher matcher;
	matcher.match(descriptor_src,descriptor_dst,matches);

	// []
	double min_dist=100;
	for( int i = 0; i < (int)matches.size(); i++ ){
		double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
	}
	min_dist = max(6*min_dist, 0.15);

	// [清理匹配，找出相近的匹配]
	for( int i=0; i<(int)matches.size(); i++){
		if(matches[i].distance<min_dist)
			good_matches.push_back(matches[i]);
	}
	/* 
	// [如果清理出來的匹配少於10個]
	int tmp=(int)matches.size();
	for( int i=0; i<min(tmp,300); i++){
		good_matches.push_back(matches[i]);
	}
	*/

	// [把這些好的匹配點抓出來]
	vector<Point2f> srcPoints,dstPoints;
	for( int i=0; i<(int)good_matches.size(); i++){
		srcPoints.push_back(keypoints_src[good_matches[i].queryIdx].pt);
		dstPoints.push_back(keypoints_dst[good_matches[i].trainIdx].pt);
	}

	// [Homography單應矩陣求解]
	Mat H_transfer = findHomography(srcPoints,dstPoints,CV_RANSAC);

	return H_transfer;
}

Mat Panorama0084::concat(Mat &image_src, Mat &image_dst, Mat &H_transfer){
	
	// [image_transform = H_transfer X image_src] [旋轉平移srcImage讓他的齊次坐標系跟dstImage一樣]
	int biasX=H_transfer.at<double>(0,2), biasY=H_transfer.at<double>(1,2);
	double cosXtoY = H_transfer.at<double>(1,0);
	H_transfer.at<double>(0,2) = 0;
	H_transfer.at<double>(1,2) = 0;

	Mat image_transform;
	cv::warpPerspective(image_src, image_transform, H_transfer, Size(image_src.cols,image_dst.rows));

	// [panoroma 圖片拼接 ]
	Mat dst(max(image_dst.rows,image_transform.rows)+abs(biasY)+abs(int(image_src.rows*cosXtoY)), max(image_dst.cols,image_transform.cols)+abs(biasX), CV_8UC3);

	if(biasX<0){ //image_src 在 image_dst 左邊
		if(biasY<0){
			image_transform.copyTo(dst(Rect(0, 30, image_transform.cols, image_transform.rows)));
			image_dst.copyTo(dst(Rect(-biasX, -biasY+30, image_dst.cols, image_dst.rows)));
		}else{
			image_transform.copyTo(dst(Rect(0, biasY+30, image_transform.cols, image_transform.rows)));
			image_dst.copyTo(dst(Rect(-biasX, 30, image_dst.cols, image_dst.rows)));
		}

	}else{ //image_src 在 image_dst 右邊
		if(biasY<0){
			image_transform.copyTo(dst(Rect(biasX, 0, image_transform.cols, image_transform.rows)));
			image_dst.copyTo(dst(Rect(0, -biasY, image_dst.cols, image_dst.rows)));
			//image_transform.copyTo(dst(Rect(biasX, 0, image_transform.cols, image_transform.rows)));
		}else{
			image_transform.copyTo(dst(Rect(biasX, biasY, image_transform.cols, image_transform.rows)));
			image_dst.copyTo(dst(Rect(0, 0, image_dst.cols, image_dst.rows)));
			//image_transform.copyTo(dst(Rect(biasX, biasY, image_transform.cols, image_transform.rows)));
		}
	}

	return dst;

}


void readFiles(double &focal, vector<Mat> &images){
	
	String path="/home/wubinray/Desktop/hw/computerPhotography/lab6/data1/";
	
	// readfile get camera's 焦距
	FILE *file=fopen((path+"K.txt").c_str(),"r");
	fscanf(file,"%lf",&focal);
	fclose(file);

	// scan all .JPG file , and read images
	vector<cv::String> fn;
	glob(path+"*.JPG", fn, false);

	size_t count = fn.size(); //number of png files in images folder
	for (size_t i=0; i<count; i++){
		images.push_back(imread(fn[i]));
		//cout<<fn[i]<<endl;
	}
}
