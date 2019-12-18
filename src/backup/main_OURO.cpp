//#include <opencv2/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <x86intrin.h>
#include <iostream>
#include <cmath>
#include <queue>
#include <string>
#include <sstream>
#include <sys/time.h>
//#include "src/libviso2/src/viso.h"
#include "src/libviso2/src/viso_stereo.h"
//#include "src/libviso2/src/viso.h"
//#include "src/libviso2/src/matcher.h"*/
#include "src/DisparityHistogram.h"
#include "src/IntegralImage.h"
#include "src/MEstimator.h"
#include "src/CubicBSpline.h"
#include "src/KalmanF.h"
#include "src/RoadRepresentation.h"
#include "src/ElevationMap.h"
//#include "src/EasyVisualOdometry.h"
#include "src/DrivingCorridor.h"
#include "src/DrivingState.h"

using namespace std;
using namespace cv;

//parameters optimized for the car- and motorbike sequences
//#define MOTORBIKE
#define CAR

void ownThresholdl2z_i(cv::Mat&, int); //integer thresholding: lower to zero

double t1;
void tic();
double toc();

int main( void )
{
		int sequence = 4;
		int startimg = 1;
		int endimg = 4000;

	#ifdef CAR
		const int maxModelDisp = 128;			//disparity of closest objects in disparity image
		const int numberOfSplines = 2;
		const int imageRows = 372;
		const float b = 0.57;					//base width
		const float f = 645;//893.5;			//focal length
		const bool compensateRollAngle=false;
		const int udfThresh = 6;				//threshold for DisparityHistogram::filterObstaclesFromUD(...)
		const int vdThresh = 100; 				//threshold for ownThresholdl2z_i(...)
		const bool useKF = false;				//use Kalman filtering

		const string pathpart = "Sequences/";
	#endif
	#ifdef MOTORBIKE
		const int maxModelDisp = 190;
		const int numberOfSplines = 4;
		const int imageRows = 720;
		const float b = 0.28;//0.48
		const float f = 798;
		const bool compensateRollAngle=true;
		const int udfThresh = 30;
		const int vdThresh = 5;
		const bool useKF = false; //no imagepairs for libviso2 available ==> no prediction possible

		const string pathpart = "Sequences/motorrad_parts/";
	#endif

		const int numberOfDBP = numberOfSplines+3;

		cv::Mat x(numberOfDBP, 1, CV_64FC1, 1); //start estimate 1

		KalmanF kf(x, 100, 10000, 0.00001, b, f, imageRows, maxModelDisp);
		//KalmanFilter kf(x, 100, 10000, 0.1, b, f, imageRows, maxModelDisp);

		//EasyVisualOdometry viso(f,660,187,b);
		DrivingState drivingState;
		
		/*VisualOdometryStereo::parameters param;
  
        // calibration parameters for sequence 2010_03_09_drive_0019 
        param.calib.f  = 645.24; // focal length in pixels
        param.calib.cu = 635.96; // principal point (u-coordinate) in pixels
        param.calib.cv = 194.13; // principal point (v-coordinate) in pixels
        param.base     = 0.5707; // baseline in meters	  
    
    
        // init visual odometry
        VisualOdometryStereo viso(param);*/
	
		/*VisualOdometry visualOdometry(param);
		Matcher matcher;
		Matrix visoTrans;*/
		
		
		//cout << "OPa" << endl;
	    //getchar();
		

	#ifdef CAR
		MEstimator mest(numberOfSplines, 30, 10, 10);
		RoadRepresentation roadRepresentation(4,2,5);
		ElevationMap elevationMap(b, f, 0.06);
	#endif
	/*#ifdef MOTORBIKE
		MEstimator mest(numberOfSplines, 30, 10, 4);
		RoadRepresentation roadRepresentation(10,2,5);
		ElevationMap elevationMap(b, f, 0.04);
	#endif*/

        //cout << "OPa2" << endl;
		//getchar();

		//Main loop

		int height = imageRows;
		int width = 1344;
		Size img_size = Size(width,height);
		
		enum { STEREO_BM = 0, STEREO_SGBM = 1 };
		int alg = 1;
		char s[25];	
    	vector<string> scene_categ{ "um", "umm", "uu" };
		vector<int> scene_categ_number{ 95, 96, 97 };
		/*string Left_imglist_path = "../../data_road/training/image_2/";
		string Right_imglist_path = "../../data_road/training/image_3/";
		string calib_path = "../../data_road/training/calib/";
		string extension = ".png";
		string calibextension = ".txt";*/
		/*string Left_imglist_path = "../../dataset/sequences/00/image_0/";
		string Right_imglist_path = "../../dataset/sequences/00/image_1/";
		string calib_path = "../../dataset/training/calib/";
		string extension = ".png";
		string calibextension = ".txt";*/

		string Left_imglist_path = "../../2010_03_09_drive_0019/";
		string Right_imglist_path = "../../2010_03_09_drive_0019/";
		string calib_path = "../../dataset/training/calib/";
		string extension = ".png";
		string calibextension = ".txt";
		
		int color_mode = alg == STEREO_BM ? 0 : 0;
		int cn = color_mode;
		Mat img1; 
		Mat img2;
		int i = 0;
		Mat disp, img1re, img2re;

		Mat disp8;//(img_size, CV_8U, Scalar(0));
		int SADWindowSize = 3, numberOfDisparities = maxModelDisp;
		StereoSGBM sgbm;
		sgbm.preFilterCap = 63;
		sgbm.SADWindowSize = SADWindowSize;
		sgbm.minDisparity = 0;
		sgbm.numberOfDisparities = numberOfDisparities;
		sgbm.P1 = 8 * 5 * sgbm.SADWindowSize * sgbm.SADWindowSize;
		sgbm.P2 = 120 * sgbm.SADWindowSize * sgbm.SADWindowSize;
		sgbm.uniquenessRatio = 0;
		sgbm.speckleWindowSize = -1;
		sgbm.speckleRange = -1;
		sgbm.disp12MaxDiff = 10;
		sgbm.fullDP = false;

		VisualOdometryStereo::parameters param;
  
		// calibration parameters for sequence 2010_03_09_drive_0019 
		param.calib.f  = f; // focal length in pixels
		param.calib.cu = 635.96; // principal point (u-coordinate) in pixels
		param.calib.cv = 194.13; // principal point (v-coordinate) in pixels
		param.base     = b; // baseline in meters

		// init visual odometry
		VisualOdometryStereo viso(param);

		// current pose (this matrix transforms a point from the current
		// frame's camera coordinates to the first frame's camera coordinates)
		Matrix pose = Matrix::eye(4);		
		

		//for(int imgindex = startimg; imgindex <= endimg; ++imgindex)
		//{

        //for (int i = 0; i<3; i++)   {

        //for(int j = 0; j < scene_categ_number.at(i); j++){

		for(int imgindex = startimg; imgindex <= endimg; ++imgindex){

                /*sprintf(s, "%06d", imgindex - 1);
	            img1 = imread((Left_imglist_path + scene_categ.at(i) + "_" + s + extension).c_str(), color_mode);   //c_str() Is a type conversion function to char *
	            img2 = imread((Right_imglist_path + scene_categ.at(i) + "_" + s + extension).c_str(), color_mode);*/
	            //cout << (calib_path + scene_categ.at(i) + "_" + s + calibextension).c_str() << endl;

				sprintf(s, "%06d", imgindex - 1);
	            img1 = imread((Left_imglist_path + "I1_" + s + extension).c_str(), color_mode);   //c_str() Is a type conversion function to char *
	            img2 = imread((Right_imglist_path + "I2_" + s + extension).c_str(), color_mode);
	            //cout << (Left_imglist_path + scene_categ.at(i) + "_" + s + extension).c_str() << endl;

				/*char base_name[256]; sprintf(base_name,"%06d.png",i);
    			string left_img_file_name  = dir + "/image_0/" + base_name;
    			string right_img_file_name = dir + "/image_1/" + base_name;*/


	            imshow("img1",img1);
				imshow("img2",img2);
				img1re = img1;
				img2re = img2;
				sgbm(img1re, img2re, disp);
				disp.convertTo(disp8, CV_8U, 1.0 / 16.0);
				imshow("disp8",disp8);
				//waitKey(0);
				
				//waitKey(0);	
				/*char buf[6];
				sprintf(buf, "%06d", imgindex);



				cout << "Image Index: " << buf << endl;*/

				//stringstream path; //clears the stream
				//string bufstr(buf);
				//path << pathpart << sequence << "/";

			    //get transformation
				//viso.pushImagePair(path.str() + "I1_" + bufstr + ".png", path.str() + "I2_" + bufstr + ".png");
                //cout << imgindex << endl;
                //cout << "OPa3" << endl;
                //getchar();

				//2 frames needed for VisualOdometry
				cv::Mat transMat;
				int32_t dims[] = {img1.cols,img1.rows,img1.cols}; 

				if(imgindex!=startimg)
				{
					viso.process(img1.data,img2.data,dims);

					pose = pose * Matrix::inv(viso.getMotion());
					//cout << pose[0,0] << endl;
					transMat = (cv::Mat_<float>(4, 4) << pose.val[0][0], pose.val[0][1], pose.val[0][2], pose.val[0][3],
									pose.val[1][0], pose.val[1][1], pose.val[1][2], pose.val[1][3],
									pose.val[2][0], pose.val[2][1], pose.val[2][2], pose.val[2][3],
									pose.val[3][0], pose.val[3][1], pose.val[3][2], pose.val[3][3]);



					//cout << transMat << endl;
					//transMat = pose;
					//tchar();
					/*viso.computeStep();
					viso.getTransformation(transMat);*/
				}
				else{
					//assume no movement
					transMat = (cv::Mat_<float>(4, 4) << 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0);
					//cout << "OLAAAAAAA" << endl;
				}

				// print transformation matrix
				//		for(int row=0; row<4; ++row)
				//		{
				//			for(int col=0; col<4; ++col)
				//			{
				//				cout << transMat.at<float>(row, col) << "\t\t";
				//			}
				//			cout << endl;
				//		}

				cv::Mat dispImgOrgi, dispImgOrg, tmpImgf, showImg, dispImg_UDFiltered, show_dispImg, realImg;
				cv::Mat dispImg4Histograms, dispImg_DCFiltered, dispImg_DC_UDFiltered;


				dispImgOrgi = disp8;
				realImg = img1;




				/*dispImgOrgi = cv::imread(path.str() + bufstr + "_disp.png", 0);
				realImg = cv::imread(path.str() + bufstr + "_input.png", 3);*/

			//convert datatype to float
				dispImgOrgi.convertTo(dispImgOrg, CV_32FC1);

				//cout << "OPa3" << endl;
                //getchar();


			/*#ifdef MOTORBIKE
			//remove invalid points
				for(int row=540; row<dispImgOrg.rows; ++row)
				{
					for(int col=0; col<dispImgOrg.cols; ++col)
					{
						dispImgOrg.at<float>(row, col) = 0;
					}
				}
			#endif*/

			//remove all pixels where disp(u,v) >= maxModelDisp (necessary for disparity histograms)
				cv::threshold(dispImgOrg, dispImg4Histograms, maxModelDisp-1, -1, cv::THRESH_TOZERO_INV);

				/*if(compensateRollAngle)
				{
					//estimate roll angle
						float rollAngle = DisparityHistogram::estimateRollAngle(dispImg4Histograms, maxModelDisp, -30, 30, 2);
						cout << "roll angle: " << rollAngle << endl;

					//rotate all images
						cv::Mat rotImg;
						cv::Mat rotMat = cv::getRotationMatrix2D(cv::Point2f(dispImg4Histograms.rows/2., dispImg4Histograms.cols/2.), rollAngle, 1);

						cv::warpAffine(dispImg4Histograms, rotImg, rotMat, dispImg4Histograms.size(), cv::INTER_NEAREST);
						dispImg4Histograms = rotImg.clone();

						cv::warpAffine(dispImgOrg, rotImg, rotMat, dispImg4Histograms.size(), cv::INTER_NEAREST);
						dispImgOrg = rotImg.clone();

						cv::warpAffine(realImg, rotImg, rotMat, dispImg4Histograms.size(), cv::INTER_NEAREST);
						realImg = rotImg.clone();
				}*/

				//original vDisparity
				cv::Mat vDispUnfiltered, vDispUnfilteredtemp;
				DisparityHistogram::calculateVDisparity(dispImg4Histograms, vDispUnfiltered, maxModelDisp);
				vDispUnfiltered.convertTo(vDispUnfilteredtemp, CV_8UC1);
				cv::imshow( "vDispUnfilteredtemp", vDispUnfilteredtemp ) ;
				vDispUnfiltered.convertTo(vDispUnfiltered, CV_32FC1);

			//take driving corridor into account
			/*#ifdef CAR
				drivingState.update(sequence, imgindex);
				DrivingCorridor::filter(dispImg4Histograms, dispImg_DCFiltered, drivingState.steeringAngle);
			#endif*/
			/*#ifdef MOTORBIKE
				DrivingCorridor::filter(dispImg4Histograms, dispImg_DCFiltered, 0);
			#endif*/

			//Calculate uDisp and remove obstacles from dispImgOrg
				cv::Mat uDisp, uDispTemp ;
				DisparityHistogram::calculateUDisparity(dispImg4Histograms, uDisp, maxModelDisp);

				uDisp.convertTo(uDispTemp, CV_8UC1) ;
				cv::imshow( "uDispTemp", uDispTemp ) ;

				DisparityHistogram::filterObstaclesFromUD(dispImgOrg, uDisp, dispImg_UDFiltered, udfThresh);

				Mat dispImg_UDFilteredTemp, dispImg_DC_UDFilteredTemp; 
				dispImg_UDFiltered.convertTo(dispImg_UDFilteredTemp, CV_8UC1);

				//DisparityHistogram::filterObstaclesFromUD(dispImg_DCFiltered, uDisp, dispImg_DC_UDFiltered, udfThresh);

				//dispImg4Histograms.convertTo(dispImg_DC_UDFilteredTemp, CV_8UC1);

				cv::imshow( "dispImg_UDFilteredTemp", dispImg_UDFilteredTemp );
				//cv::imshow( "dispImg_DC_UDFilteredTemp", dispImg_DC_UDFilteredTemp);


			//Calculate vDisp
				cv::Mat vDispOrg_i;
				DisparityHistogram::calculateVDisparity(dispImg_UDFiltered, vDispOrg_i, maxModelDisp);

			//threshold vDisp

				cv::Mat vIntImg, uIntImg, weight, vDispOrg, vDispWeighted, vDispOrg_i_temp;

				

				vDispOrg_i.convertTo(vDispOrg_i_temp, CV_8UC1);

				cv::imshow("vDispOrg_i_temp",vDispOrg_i_temp);

				//cv::imshow( "dispImg_UDFilteredTemp", dispImg_UDFilteredTemp );
				//cv::threshold(vDisp, tmpImgf, 20, -1, cv::THRESH_TOZERO);//only compatibel with 8-Bit images
				//ownThresholdl2z_i(vDispOrg_i, vdThresh);


				vDispOrg_i.convertTo(vDispOrg_i_temp, CV_8UC1);

				cv::imshow("vDispOrg_i_tem_filter",vDispOrg_i_temp);
			//Weight vDisp
				
				vDispOrg_i.convertTo(vDispOrg, CV_32FC1);

				Mat vIntImg_temp;
				

				IntegralImage::vIntegralImage(vDispOrg, vIntImg);

				vIntImg.convertTo(vIntImg_temp, CV_8UC1);

				IntegralImage::vWeight(vIntImg, weight);

				showImg = weight;

			//map weight
				cv::Mat mappedWeight;
				IntegralImage::mapWeightPC(weight, mappedWeight, 0.98);

			//apply weight
				IntegralImage::applyWeight(vDispOrg, mappedWeight, tmpImgf, 10);

				IntegralImage::uIntegralImageR(vDispOrg, uIntImg);
				IntegralImage::uWeightR(uIntImg, weight);

				IntegralImage::mapWeightPC(weight, mappedWeight, 0.98);

				showImg = weight;

				IntegralImage::applyWeight(tmpImgf, mappedWeight, vDispWeighted, 10);


				cv::imshow("vDispWeighted",vDispWeighted);
				//cv::imshow("dispImg4Histograms", dispImg4Histograms);
				cv::imshow("dispImg_UDFilteredTemp", dispImg_UDFilteredTemp);
				cv::imshow("vIntImg_temp",vIntImg_temp);
				//cv::imshow("vIntImg",showImg);
				cv::imshow("weight",weight);
				


			//MEstimator
				cv::Mat c;
				mest.estimate(vDispWeighted);

				cv::Mat spline;

				//cout << drivingState.state << endl;

			//Use Kalman filtering while driving straight
				if(useKF && drivingState.state == DrivingState::straight)
				{
					//cout << "opa" << endl;
					kf.kalmanStep(transMat, mest.H_weighted, mest.z_weighted);
					//cout << "opa2" << endl;
					kf.getStateVector(c);
					//cout << "opa3" << endl;
					kf.getSplineSample(spline);
					//cout << "opa" << endl;

				}
			//Use MEstimator result while turning
				else
				{
					c = mest.c.clone();
					CubicBSpline::getSample(c, 0.01, maxModelDisp, spline);
					kf.reset(c, 0.01);
				}

			//plot spline
				cv::cvtColor(vDispUnfiltered, show_dispImg, CV_GRAY2RGB);
				//CubicBSpline::plotSample(show_dispImg, spline, CV_RGB(0,0,255));

				//road representation
				roadRepresentation.calculateLUTs(vDispWeighted, c);

				float minDisp = roadRepresentation.validSampleRange.minIndex;
				cv::line(show_dispImg, cv::Point(minDisp, 0), cv::Point(minDisp, show_dispImg.rows), CV_RGB(0,255,0));


				//Plot LUTs
				for(int row=0; row<imageRows; ++row)
				{
					cv::circle(show_dispImg, cv::Point(roadRepresentation.LUT_dispOfRow.at<int>(row, 0), row), 1, CV_RGB(0,0,255));
				}

				//		for(int col=0; col<maxModelDisp; ++col)
				//		{
				//			cv::circle(show_dispImg, cv::Point(col, roadRepresentation.LUT_rowOfDisp.at<int>(col, 0)), 1, 100);
				//		}

				cv::imshow( "show_dispImg", show_dispImg ) ;


			//elevation map
				//elevationMap.draw(dispImgOrg, dispImg_UDFiltered, realImg, roadRepresentation, 1);
				//cv::imshow( "dispImgOrg", dispImgOrg ) ;
				cv::imshow( "dispImg_UDFiltered", dispImg_UDFiltered );
				//cv::imshow( "ElevationMap", realImg ) ;
				waitKey(0);

				/*cv::imshow("plain elevation map", elevationMap.plainElevationMap);
				cv::imshow("plain elevation map color encoded", elevationMap.plainElevationMapColored_);
				cv::imshow( "ElevationMap", realImg ) ;
				cv::waitKey(0);*/

				//first frame
				/*if(imgindex==startimg){
					cv::waitKey(0);
				}*/
			//}
		}
}


void ownThresholdl2z_i(cv::Mat& imgsrc, int threshold)
{
	#pragma omp parallel for
	for( int col = 0 ; col < imgsrc.cols ; ++col )
	{
		for( int row = 0 ; row < imgsrc.rows ; ++row )
		{
			if(imgsrc.at<int>(row, col) < threshold){
				imgsrc.at<int>(row, col) = 0;
			}
		}
	}
}

void tic()
{
	timeval tim;
	gettimeofday(&tim, NULL);
	t1=tim.tv_sec+(tim.tv_usec/1000000.0);
}

double toc()
{
	timeval tim;
	double t2;
	gettimeofday(&tim, NULL);
	t2=tim.tv_sec+(tim.tv_usec/1000000.0);

	cout << "time elapsed: " << t2-t1 << endl;

	return t2-t1;
}
