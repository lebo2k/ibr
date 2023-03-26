#ifndef Calibration_H
#define Calibration_H

#include "Globals.h"

class Calibration {

private:

public:

	int width_, height_; // image width and height in pixels
	double fx_; // fx = f * kx where f is focal length and kx is scaling factor between pixels and meters in x direction
	double fy_; // fy = f * ky where f is focal length and ky is scaling factor between pixels and meters in y direction
	double cx_; // principal point x coordinate
	double cy_; // principal point y coordinate
	double skew_; // skew coefficient
	double K1_, K2_, K3_; // radial distortion coefficients
	double P1_, P2_; // tangential distortion coefficients

	Matrix3d K_; // 3x3 camera intrinsics matrix
	Matrix3d Kinv_; // 3x3 inverse camera intrinsics matrix

	Calibration();
	~Calibration();

	void Init(xml_node<> *calibration_node);
	void ComputeK();
	void RecalibrateNewSS(cv::Size view_size); // recalibrates to a new screen space pixel view_size and updates K_ and Kinv_
	Calibration Copy(); // copies values to new calibration instance
	void Undistort(Mat &img); // undistorts image relative to non-linear radial and tangential distortion
	void Undistort_DepthMap(Matrix<float, Dynamic, Dynamic> &depth_map); // undistorts matrix relative to non-linear radial and tangential distortion
	void UpdateFromKinv(Matrix3d Kinv);

	void Print(); // debug printing
};

#endif