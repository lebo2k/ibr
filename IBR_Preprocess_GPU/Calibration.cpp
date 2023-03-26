#include "Calibration.h"

Calibration::Calibration() {
	// initialize member values
	width_ = 0.;
	height_ = 0.;
	fx_ = 0.;
	fy_ = 0.;
	cx_ = 0.;
	cy_ = 0.;
	skew_ = 0.;
	K1_ = 0.;
	K2_ = 0.;
	K3_ = 0.;
	P1_ = 0.;
	P2_ = 0.;
}

Calibration::~Calibration() {
}

/*
set member values according to data in node argument from Agisoft doc.xml file
*/
void Calibration::Init(xml_node<> *calibration_node) {
	assert(strcmp(calibration_node->name(), "calibration") == 0);

	std::string s;
	xml_node<> *curr_node;

	curr_node = calibration_node->first_node("resolution");

	if (curr_node != 0) {
		s = curr_node->first_attribute("width")->value();
		if (!s.empty()) width_ = convert_string_to_int(s);

		s = curr_node->first_attribute("height")->value();
		if (!s.empty()) height_ = convert_string_to_int(s);
	}

	curr_node = calibration_node->first_node("fx");
	if (curr_node != 0) {
		s = curr_node->value();
		if (!s.empty()) fx_ = convert_string_to_double(s);
	}

	curr_node = calibration_node->first_node("fy");
	if (curr_node != 0) {
		s = curr_node->value();
		if (!s.empty()) fy_ = convert_string_to_double(s);
	}

	curr_node = calibration_node->first_node("cx");
	if (curr_node != 0) {
		s = curr_node->value();
		if (!s.empty()) cx_ = convert_string_to_double(s);
	}

	curr_node = calibration_node->first_node("cy");
	if (curr_node != 0) {
		s = curr_node->value();
		if (!s.empty()) cy_ = convert_string_to_double(s);
	}

	curr_node = calibration_node->first_node("skew");
	if (curr_node != 0) {
		s = curr_node->value();
		if (!s.empty()) skew_ = convert_string_to_double(s);
	}

	curr_node = calibration_node->first_node("K1");
	if (curr_node != 0) {
		s = curr_node->value();
		if (!s.empty()) K1_ = convert_string_to_double(s);
	}

	curr_node = calibration_node->first_node("K2");
	if (curr_node != 0) {
		s = curr_node->value();
		if (!s.empty()) K2_ = convert_string_to_double(s);
	}

	curr_node = calibration_node->first_node("K3");
	if (curr_node != 0) {
		s = curr_node->value();
		if (!s.empty()) K3_ = convert_string_to_double(s);
	}

	curr_node = calibration_node->first_node("P1");
	if (curr_node != 0) {
		s = curr_node->value();
		if (!s.empty()) P1_ = convert_string_to_double(s);
	}

	curr_node = calibration_node->first_node("P2");
	if (curr_node != 0) {
		s = curr_node->value();
		if (!s.empty()) P2_ = convert_string_to_double(s);
	}

	ComputeK();
}

void Calibration::ComputeK() {
	K_.setZero();
	K_(0, 0) = fx_;
	K_(1, 1) = fy_;
	K_(0, 1) = skew_;
	K_(0, 2) = cx_;
	K_(1, 2) = cy_;
	K_(2, 2) = 1.;
	Kinv_ = K_.inverse();
}

void Calibration::UpdateFromKinv(Matrix3d Kinv) {
	Kinv_ = Kinv;
	K_ = Kinv_.inverse();
	fx_ = K_(0, 0);
	fy_ = K_(1, 1);
	skew_ = K_(0, 1);
	cx_ = K_(0, 2);
	cy_ = K_(1, 2);
}

// recalibrates to a new screen space pixel view_size and udpates K_ and Kinv_
void Calibration::RecalibrateNewSS(cv::Size view_size) {
	bool debug = false;

	if (debug) {
		cout << "K before SS recalibration from " << height_ << " x " << width_ << " to " << view_size.height << " x " << view_size.width << endl << K_ << endl;
		cin.ignore();
	}

	// update resolution info
	double scale_factor_x = (double)width_ / (double)view_size.width;
	double scale_factor_y = (double)height_ / (double)view_size.height;
	width_ = view_size.width;
	height_ = view_size.height;

	// update camera intrinsics
	fx_ /= scale_factor_x;
	fy_ /= scale_factor_y;
	cx_ /= scale_factor_x;
	cy_ /= scale_factor_y;
	ComputeK();

	if (debug) {
		cout << "K after SS recalibration" << endl << K_ << endl;
		cin.ignore();
	}
}

// copies values to new calibration instance
Calibration Calibration::Copy() {
	Calibration calib;

	calib.width_ = width_;
	calib.height_ = height_;
	calib.fx_ = fx_;
	calib.fy_ = fy_;
	calib.skew_ = skew_;
	calib.cx_ = cx_;
	calib.cy_ = cy_;
	calib.K1_ = K1_;
	calib.K2_ = K2_;
	calib.K3_ = K3_;
	calib.P1_ = P1_;
	calib.P2_ = P2_;
	calib.K_ = K_;
	calib.Kinv_ = Kinv_;

	return calib;
}

// undistorts image relative to non-linear radial and tangential distortion
void Calibration::Undistort(Mat &img) {
	if ((K1_ == 0.0) &&
		(K2_ == 0.0) &&
		(P1_ == 0.0) &&
		(P2_ == 0.0) &&
		(K3_ == 0.0)) return; // no distortion information available to correct

	std::vector<double> distCoeffs;
	distCoeffs.push_back(K1_);
	distCoeffs.push_back(K2_);
	distCoeffs.push_back(P1_);
	distCoeffs.push_back(P2_);
	distCoeffs.push_back(K3_);
	Mat img_new = cv::Mat::zeros(img.size(), img.type());
	Mat Kcv = cv::Mat::zeros(3, 3, CV_64F);
	EigenOpenCV::eigen2cv(K_, Kcv);
	cv::undistort(img, img_new, Kcv, distCoeffs);
	img_new.copyTo(img);
}

// undistorts matrix relative to non-linear radial and tangential distortion
void Calibration::Undistort_DepthMap(Matrix<float, Dynamic, Dynamic> &depth_map) {
	if ((K1_ == 0.0) &&
		(K2_ == 0.0) &&
		(P1_ == 0.0) &&
		(P2_ == 0.0) &&
		(K3_ == 0.0)) return; // no distortion information available to correct

	Mat img = Mat(depth_map.rows(), depth_map.cols(), CV_32F);
	EigenOpenCV::eigen2cv(depth_map, img);

	std::vector<double> distCoeffs;
	distCoeffs.push_back(K1_);
	distCoeffs.push_back(K2_);
	distCoeffs.push_back(P1_);
	distCoeffs.push_back(P2_);
	distCoeffs.push_back(K3_);
	Mat img_new = cv::Mat::zeros(img.size(), img.type());
	Mat Kcv = Mat(3, 3, CV_64F);
	EigenOpenCV::eigen2cv(K_, Kcv);
	cv::undistort(img, img_new, Kcv, distCoeffs);
	
	EigenOpenCV::cv2eigen(img_new, depth_map);
}

// debug printing
void Calibration::Print() {
	cout << "Calibration" << endl;
	cout << "Width, height " << width_ << ", " << height_ << endl;
	cout << "K = " << endl << " " << K_ << endl << endl;
	cout << "Kinv_ = " << endl << " " << Kinv_ << endl << endl;
	cout << endl;
}