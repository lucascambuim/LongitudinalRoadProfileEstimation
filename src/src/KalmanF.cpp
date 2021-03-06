/*
 * KalmanF.cpp
 *
 *  Created on: 19.07.2011
 *      Author: joos
 */

/*#include <iostream>
#include "KalmanF.h"
#include "CubicBSpline.h"
#include <sys/time.h>*/

#include "KalmanF.h"
//using namespace std;

KalmanF::KalmanF(cv::Mat stateVectorInit, double covarianceOfStateVectorInitScalar_, double processNoiseScalar_, double measurementNoiseScalar_, double b_, double f_, int imageRows_, int imageColumns_)
{
	processNoiseScalar = processNoiseScalar_;
	measurementNoiseScalar = measurementNoiseScalar_;

	b = b_;
	f = f_;
	imagecenter = imageRows_/2; // approximation. Proper: c_v (principle point)
	imagecolumns = imageColumns_;
	imagerows = imageRows_;

	constantPrediction=true;

	x = stateVectorInit.clone();
	P = cv::Mat::eye(x.rows, x.rows, CV_64FC1) * covarianceOfStateVectorInitScalar_;
	Q = cv::Mat::ones(x.rows, x.rows, CV_64FC1) * processNoiseScalar;
}

void KalmanF::reset(cv::Mat stateVectorInit, double covarianceOfStateVectorInitScalar_)
{
	x = stateVectorInit.clone();
	P = cv::Mat::eye(x.rows, x.rows, CV_64FC1) * covarianceOfStateVectorInitScalar_;
	Q = cv::Mat::ones(x.rows, x.rows, CV_64FC1) * processNoiseScalar;

	constantPrediction=true;
}

void KalmanF::kalmanStep(cv::Mat& T, cv::Mat& H_, cv::Mat& z_)
{
	H=H_.clone();
	z=z_.clone();


	predict(T);

	update(H_, z_);

	//one step computed => use resampling techniques for prediction
	constantPrediction=false;
}

void KalmanF::predict(cv::Mat& T)
{
	//No prediction possible yet
	// => constant prediction
	if(constantPrediction)
	{
		//constant prediction
		//x=x;
		P = P + Q;

		cout << "NAAAAA" << endl;
	}

	//predict by resampling spline
	else
	{
		cout << "OPAAAA" << endl;

		//resampled spline is of type double. Integers are too inaccurate and produce a drift of the spline, even if rounded correctly.
		splineResampled.create(spline.rows, spline.cols, CV_64FC1);


		//cout << spline.rows << endl;

		//resample spline
		int invalidSamples=0;
		
		#pragma omp parallel for shared(invalidSamples)
		for(int i=0; i<spline.rows; ++i)
		{
			float d = spline.at<int>(i, 0);
			float v = spline.at<int>(i, 1);

			//catch zero division
			if((int)d == 0){
				d=0.00000001;
			}

			//cout << "AEEEEE" << endl;

			float z = b*f/d;
			float x = 0;
			float y = z/f*(v-imagecenter);

			//cout << "AEEEEE33" << endl;

			cv::Mat P_old = (cv::Mat_<float>(4, 1) << x, y, z, 1);

			//cout << "AEEEEE44" << endl;
			//cout << T << endl;

			cv::Mat P_new = T*P_old;

			//cout << "AEEEEE22" << endl;

			double d_new = b*f/P_new.at<float>(2,0);
			double v_new = f*P_new.at<float>(1,0)/P_new.at<float>(2,0) + imagecenter;



			//new sample point in image range
			if((int)d_new < imagecolumns && (int)v_new < imagerows){
				splineResampled.at<double>(i,0) = d_new;
				splineResampled.at<double>(i,1) = v_new;
			}
			//else mark as invalid
			else{
				splineResampled.at<double>(i,0) = -1;
				splineResampled.at<double>(i,1) = -1;

				#pragma omp atomic
				++invalidSamples;
			}
		}

		//cout << "ola " << endl;

		//write measurement matrices
		cv::Mat H_resampled = cv::Mat::zeros(splineResampled.rows-invalidSamples, x.rows, CV_64FC1);
		cv::Mat z_resampled(splineResampled.rows-invalidSamples, 1, CV_64FC1);

		int numberOfSplines = x.rows-3;
		double knotDistance = (double)imagecolumns/(double)numberOfSplines;



		int counter=0;
		#pragma omp parallel for
		for(int i=0; i<splineResampled.rows; ++i)
		{
			//calculate spline part
			double disp = splineResampled.at<double>(i, 0);

			//catch invalid samples
			if(disp < -0.5){
				continue;
			}

			int splinePart = disp/knotDistance; //first part is referenced by 0

			//calculate curve parameter for this spline part t e[0,1)
			double t = disp/knotDistance-splinePart;

			//get the B-Spline base values for this parameter
			cv::Mat baseVals;
			CubicBSpline::CUBaseFunctions(t, baseVals);

			#pragma omp critical
			{
				//write matrix H
				H_resampled.at<double>(counter, splinePart) = baseVals.at<double>(0,0);
				H_resampled.at<double>(counter, splinePart+1) = baseVals.at<double>(0,1);
				H_resampled.at<double>(counter, splinePart+2) = baseVals.at<double>(0,2);
				H_resampled.at<double>(counter, splinePart+3) = baseVals.at<double>(0,3);

				//cout << H_resampled.at<double>(counter, 0) << " " <<H_resampled.at<double>(counter, 1) << " " <<H_resampled.at<double>(counter, 2) << " " <<H_resampled.at<double>(counter, 3) << " " <<H_resampled.at<double>(counter, 4) << " " <<H_resampled.at<double>(i, 5) << " " <<H_resampled.at<double>(i, 6) << " " <<H_resampled.at<double>(counter, 7) << endl;

				//write measurement vector z
				z_resampled.at<double>(counter, 0) = splineResampled.at<double>(i, 1);

				++counter;
			}
		}

		//solve least squares to predict x
		cv::Mat x_predict(8,1,CV_64FC1);

		if(!solve(H_resampled, z_resampled, x_predict, cv::DECOMP_SVD)){
			std::cout << "Error Solving System!" << std::endl;
		}

		//prediction with system matrix
		//approximate system matrix F
		// x_k = F*x_(k-1)  ==>  F = x_k / x_(k-1)
		cv::Mat F = (x_predict.mul(1/x)*cv::Mat::ones(1, x.rows, CV_64FC1)).mul(cv::Mat::eye(x.rows, x.rows, CV_64FC1));

		//predict
		cv::Mat P_predict = F*P*F.t() + Q;

		//prediction by increasing covariance
		//cv::Mat P_predict = P + Q;

		P = P_predict.clone();
		x = x_predict.clone();
	}
}

void KalmanF::update(cv::Mat& H_, cv::Mat& z_)
{
	R = cv::Mat::eye(z_.rows, z_.rows, CV_64FC1) * measurementNoiseScalar;
	R_inv = cv::Mat::eye(z_.rows, z_.rows, CV_64FC1) * 1./measurementNoiseScalar;

	cv::Mat Ht = H.t(); //transponent of H

//Kalman update

//	cv::Mat y = z-H*x; 						//residual
//	cv::Mat S = H*P*Ht+R; 					//innovation
//	cv::Mat K = P*Ht*S.inv(cv::DECOMP_SVD);	//Kalman gain
//	x = x+K*y;								//update state vector
//	P = (cv::Mat::eye(P.rows, P.rows, CV_64FC1) - K*H)*P; //update state covariance

//Kalman update using information filter
	cv::Mat P_inv = P.inv();

	//substitution
	cv::Mat Y = P_inv;
	cv::Mat I = Ht*R_inv*H;
	cv::Mat y = P_inv*x;
	cv::Mat i = Ht*R_inv*z;

	//update
	Y = Y + I;
	y = y + i;

	//resubstitution
	cv::Mat P_ = Y.inv();
	cv::Mat x_ = P_*y;

	P = P_.clone();
	x = x_.clone();

	//update sample
	CubicBSpline::getSample(x, 0.01, imagecolumns, spline);
}

void KalmanF::getStateVector(cv::Mat& x_)
{
	x_ = x;
}

void KalmanF::getSplineSample(cv::Mat& S_)
{
	S_ = spline;
}
