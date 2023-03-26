#include "Camera.h"

// Constructors / destructor

Camera::Camera() {
	id_ = -1;
	fn_ = "";
	dm_ = new DepthMap();

	P_ = cv::Mat::zeros(3, 4, CV_64F);
	Pinv_ = cv::Mat::zeros(4, 3, CV_64F);
	RT_ = cv::Mat::zeros(4, 4, CV_64F);
	RTinv_ = cv::Mat::zeros(4, 4, CV_64F);

	pos_ = Point3d(0., 0., 0.);

	width_ = 0;
	height_ = 0;

	orientation_ = AGO_ORIGINAL;

	enabled_ = true;
	posed_ = false;
	has_depth_map_ = false;

}

Camera::~Camera() {
	delete dm_;
}

// Initialization

// parse string of 16 ordered doubles (in col then row order) into RT_ and RTinv_ camera extrinsics matrices
// it expects the transform being parsed is a camera location in world space, and therefore represents RTinv
// see Scene.h for description of AgisoftToWorld
void Camera::ParseAgisoftCameraExtrinsics(std::string s, Mat *AgisoftToWorld_, Mat *AgisoftToWorldinv_) {
	bool debug = false;

	RTinv_ = ParseString_Matrix64F(s, 4, 4); // camera location in chunk coordinate space; note that the length of vector T in RT is the distance from the camera origin to the world origin, and so is the 4th column of RT-1.  But the former is expressed in coordinates from the camera's perspective (multiply camera space origin [0 0 0 1].transpose() by RT to get the camera's location expressed in chunk coordinate space), and the latter in coordinates from the world space origin's perspective (multiply world space origin [0 0 0 1].transpose() by RT-1 to get the world space origin's location expressed in camera space).  So the camera location in chunk coordinate space is RT-1.
	RT_ = RTinv_.inv();

	RTinv_ = (*AgisoftToWorldinv_ )* RTinv_;
	RT_ = RT_ * (*AgisoftToWorld_);

	UpdatePos();
	UpdateViewDir();

	if (debug) {
		cout << "RT: " << endl << RT_ << endl << endl;
		cout << "RTinv: " << endl << RTinv_ << endl << endl;
	}
}

// set member values according to data in node argument from Agisoft doc.xml file
// see Scene.h for description of AgisoftToWorld
void Camera::Init(xml_node<> *camera_node, Mat *AgisoftToWorld_, Mat *AgisoftToWorldinv_) {
	bool timing = true; double t;
	if (timing) t = (double)getTickCount();
	
	assert(strcmp(camera_node->name(), "camera") == 0, "Camera::Camera() wrong arg node type passed");

	std::string s;
	xml_node<> *curr_node;

	s = camera_node->first_attribute("id")->value();
	if (!s.empty()) {
		id_ = convert_string_to_int(s);
		dm_->cam_id_ = id_;
	}

	s = camera_node->first_attribute("sensor_id")->value();
	if (!s.empty()) sensor_id_ = convert_string_to_int(s);

	s = camera_node->first_attribute("enabled")->value();
	if (s != "true") enabled_ = false;
	else enabled_ = true;

	s = camera_node->first_node("frames")->first_node("frame")->first_node("image")->first_attribute("path")->value(); // assumes one image per camera
	fn_ = s.substr(s.find_last_of("\\/") + 1); // strip out relative path

	curr_node = camera_node->first_node("frames")->first_node("frame")->first_node("mask");
	if (curr_node != 0) {
		s = curr_node->first_attribute("path")->value();
		if (!s.empty()) fn_mask_ = s.substr(s.find_last_of("\\/") + 1); // strip out relative path
	}

	curr_node = camera_node->first_node("resolution");
	s = curr_node->first_attribute("width")->value();
	if (!s.empty()) width_ = convert_string_to_int(s);
	s = curr_node->first_attribute("height")->value();
	if (!s.empty()) height_ = convert_string_to_int(s);

	curr_node = camera_node->first_node("transform");
	if (curr_node != 0) { // transforms may not have been computed for all cameras
		s = curr_node->value();
		ParseAgisoftCameraExtrinsics(s, AgisoftToWorld_, AgisoftToWorldinv_);
		posed_ = true;
	}

	// load image
	std::string fn_full = GLOBAL_FILEPATH_INPUT + fn_;
	imgT_ = imread(fn_full, IMREAD_COLOR); // use cv::IMREAD_ANYDEPTH|cv::IMREAD_ANYCOLOR to retain depth and color types from original, or IMREAD_COLOR to convert to 8-bit color
	assert(imgT_.rows > 0 && imgT_.cols > 0, "Camera::Init() image not found");
	assert(imgT_.rows == height_ && imgT_.cols == width_, "Camera::Init() size of loaded image does not match size given by Agisoft doc.xml");

	// load mask
	std::string fn_mask_full = GLOBAL_FILEPATH_INPUT + fn_mask_;
	imgMask_ = imread(fn_mask_full, IMREAD_GRAYSCALE);
	if (imgMask_.rows > 0 && imgMask_.cols > 0) { // image mask found
		assert(imgMask_.rows == height_ && imgMask_.cols == width_, "Camera::Init() size of loaded image mask does not match size of image given by Agisoft doc.xml");
	} else {
		imgMask_ = cv::Mat(height_, width_, CV_8UC1, 255); // if mask not found, assume all pixels are "opaque" foreground pixels
	}

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Camera::Init() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// initializes projection matrices P and Pinv using the camera intrinsics matrix arg K
// arg _K is a camera intrinsics matrix of size (4, 4) and type CV_64F
void Camera::InitSensor(Sensor *sensor) {
	if (orientation_==AGO_ORIGINAL) assert(sensor->height_ == height_ && sensor->width_ == width_, "Camera::InitSensor() resolution given by camera does not match resolution given by sensor");
	else assert(sensor->height_ == width_ && sensor->width_ == height_, "Camera::InitSensor() resolution given by camera does not match resolution given by sensor");  // portrait mode
	
	calib_ = sensor->calib_.Copy();
	
	UpdateCameraMatrices();
	
}

// returns 3 by (ss_w*ss_h) data structure of type CV_32F with homogeneous pixel positions for a screen space of pixel dimensions ss_w*ss_h
Mat Camera::ConstructSSCoords(int ss_w, int ss_h) {
	bool debug = false;

	Mat I = cv::Mat::ones(Size(ss_w*ss_h, 3), CV_32F); // 3xn matrix of homogeneous screen space points where n is the number of pixels in imgT_
	float *p0, *p1; // one pointer for each row of I to be modified
	p0 = I.ptr<float>(0); // holds pixel x positions
	p1 = I.ptr<float>(1); // holds pixel y positions

	int idx;
	for (int r = 0; r < ss_h; r++) {
		for (int c = 0; c < ss_w; c++) {
			idx = PixIndexFwd(Point(c, r), ss_w);
			p0[idx] = (float)c;
			p1[idx] = (float)r;
		}
	}

	return I;
}

// set depth map using data in node argument from Agisoft doc.xml file and associated .exr file
// depth downscale is the downward scale factor as given by Agisoft for the depth map from the original image
// uses calib_.focal_length, so calib_ must be set by calling InitSensor() before this function, so sensor_id must not be 0
// agisoft_to_world_scale_ is the scale factor associated with the change from Agisoft's space to our world space, given by the chunk transform in the xml file from Agisoft
void Camera::InitDepthMap(xml_node<> *depthmap_node, double agisoft_to_world_scale_, int depth_downscale) {
	bool debug = false;
	bool timing = true; double t;
	if (timing) t = (double)getTickCount();
	
	if (!posed_) return;

	dm_->Init(depthmap_node, agisoft_to_world_scale_, depth_downscale);
	assert(id_ == dm_->cam_id_, "Camera::ImportDepthMap() camera ID of depth map does not match this camera's ID");

	imgD_ = cv::Mat::zeros(dm_->imgD_.rows, dm_->imgD_.cols, CV_32F);
	dm_->imgD_.copyTo(imgD_);

	has_depth_map_ = true;

	if (debug) DepthMap::DisplayDepthImage(&imgD_);

	Downsample(imgD_.cols, imgD_.rows, false);

	InitDepthMapDiscontinuities();

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Camera::InitDepthMap() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// inpaints depth map by searching from each missing pixel (u,v) in 8 canonical directions until reach a pixel with a depth value (ud, pd), then perform a weighted interpolation
// weighted interpolation: for all u,v in image, img(v,u) = sum over (ud, vd) of (dist^-2 * depth) / sum over (ud, vd) of dist^-2
// for both computing new depth values and using existing ones, ignores pixels denoted by the image mask as masked out
void Camera::InpaintDepthMap() {
	bool debug = false;
	assert(imgD_.rows == imgMask_.rows && imgD_.cols == imgMask_.cols, "Camera::InpaintDepthMap() imgD_ and imgMask_ must be of the same size");

	if (debug) DepthMap::DisplayDepthImage(&imgD_);

	float* pD;
	uchar* pM, *pDM;
	float depth, depth_wtd, sum_num, sum_denom;
	int dist;
	bool depth_found;
	for (int r = 0; r < imgD_.rows; r++) {
		pD = imgD_.ptr<float>(r);
		pM = imgMask_.ptr<uchar>(r);
		for (int c = 0; c < imgD_.cols; c++) {
			if (pD[c] != 0.) continue; // skip pixels that already have values
			if (pM[c] == 0) continue; // skip pixels that are masked out
			sum_num = 0.;
			sum_denom = 0.;
			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {
					if ((i == 0) && (j == 0)) continue; // must step in some non-zero direction
					depth_found = GetNearestDepth(Point(c, r), Point(i, j), dist, depth);
					if (depth_found) {
						sum_num += powf((float)dist, -2.) * depth;
						sum_denom += powf((float)dist, -2.);
					}
				}
			}
			if ((sum_num != 0) &&
				(sum_denom != 0)) {
				depth_wtd = sum_num / sum_denom;
				pD[c] = depth_wtd;
			}
		}
	}

	if (debug) DepthMap::DisplayDepthImage(&imgD_);
}

// traverses depth image imgD_ from p in direction step until encounters a pixel with a non-zero depth value that is not masked out in imgMask
// step must be one pixel-neighbor away in one of the 8 canonical lattice directions
// returns true if such a pixel is found, updates arg depth with its value, and updates dist with the distance to the non-zero pixel (diagonal steps count as distance 1); returns false if no such pixel is found within imgD_
// if a masked out pixel is encountered at any point, returns false
// if there is a depth value at p and p is not masked out, returns true and updates arg depth to p's depth
bool Camera::GetNearestDepth(Point p, Point step, int &dist, float &depth) {
	assert(imgD_.rows == imgMask_.rows && imgD_.cols == imgMask_.cols, "Camera::GetNearestDepth() imgD_ and imgMask_ must be of the same size");
	assert(p.x >= 0 && p.y >= 0 && p.x < imgD_.cols && p.y < imgD_.rows, "Camera::GetNearestDepth() p must be within image bounds");
	assert(step.x != 0 || step.y != 0, "Camera::GetNearestDepth() step must be non-zero in one of the two directions");
	assert(step.x == -1 || step.x == 0 || step.x == 1, "Camera::GetNearestDepth() step.x must be either -1, 0, or 1");
	assert(step.y == -1 || step.y == 0 || step.y == 1, "Camera::GetNearestDepth() step.y must be either -1, 0, or 1");
	
	bool found = false;
	dist = 0;
	depth = imgD_.at<float>(p.y, p.x);
	if ((depth != 0.) &&
		(imgMask_.at<uchar>(p.y, p.x) != 0))
		found = true;

	while (!found) {
		p.y = p.y + step.y;
		p.x = p.x + step.x;
		dist++;

		if ((p.x < 0) ||
			(p.y < 0) ||
			(p.x >= imgD_.cols) ||
			(p.y >= imgD_.rows))
			break;

		depth = imgD_.at<float>(p.y, p.x);
		if ((depth != 0.) &&
			(imgMask_.at<uchar>(p.y, p.x) != 0))
			found = true;
	}

	return found;
}

void Camera::UndistortPhotos() {
	bool debug = false;

	if (debug) display_mat(&imgT_, "imgT before undistortion");
	if (debug) display_mat(&imgD_, "imgD before undistortion");
	if (debug) display_mat(&imgMask_, "imgMask_ before undistortion");

	calib_.Undistort(imgT_);
	calib_.Undistort(imgD_);
	calib_.Undistort(imgMask_);

	if (debug) display_mat(&imgT_, "imgT after undistortion");
	if (debug) display_mat(&imgD_, "imgD after undistortion");
	if (debug) display_mat(&imgMask_, "imgMask_ after undistortion");
}

// called by InitDepthMap() to initialize imgDdisc_
void Camera::InitDepthMapDiscontinuities() {
	bool debug = false;

	imgDdisc_ = cv::Mat::zeros(Size(imgD_.cols, imgD_.rows), CV_8UC1);

	unsigned char* p;
	for (int r = 0; r < imgDdisc_.rows; r++) {
		p = imgDdisc_.ptr<unsigned char>(r);
		for (int c = 0; c < imgDdisc_.cols; c++) {
			bool disc = DetermineDepthMapDiscontinuity(Point(c, r));
			if (disc) p[c] = 255;
		}
	}

	if (debug) display_mat(&imgDdisc_, "imgDdisc_");
}

// tests pixel at point p for high depth map discontinuity and returns result boolean
bool Camera::DetermineDepthMapDiscontinuity(Point p) {
	if ((p.x <= 0) ||
		(p.y <= 0) ||
		(p.x >= (imgD_.cols - 2)) ||
		(p.y >= (imgD_.rows - 2)))
		return false; // needs 1-pixel border to compute; on border, assume no high depth discontinuities

	double td = dm_->max_depth_ / 4.; // threshold Td for identifying high discontinuities in depth; threshold should equal about 25% of the maximal depth value
	double sum = -9.0 * imgD_.at<float>(p.y, p.x);
	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			sum += imgD_.at<float>(p.y + j, p.x + i);
		}
	}
	if (abs(sum) > td) return true;
	else return false;
}

// downsamples images and updates resolution info, camera intrinsics, and projection matrices accordingly; if include_depth_map flag is TRUE, also downsamples the associated depth map
// used to downsample this way with all images at same scale_factor, but resulted in rounding that left imgT and imgD with different sizes since only downsampling imgT here
void Camera::Downsample(float scale_factor, bool include_depth_map) {
	// update resolution info
	width_ = round((width_ / scale_factor), 0);
	height_ = round((height_ / scale_factor), 0);

	// update image
	Mat imgTnew_ = cv::Mat::zeros(height_, width_, CV_8UC3);
	resize(imgT_, imgTnew_, cv::Size(width_, height_), 0.0, 0.0, CV_INTER_AREA); // to shrink an image, it will generally look best with CV_INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)
	imgT_ = imgTnew_;

	// update calibration and camera intrinsics
	calib_.RecalibrateNewSS(cv::Size(width_, height_));

	// update projection matrices
	UpdateCameraMatrices();

	// update image mask
	Mat imgMasknew = cv::Mat::zeros(height_, width_, CV_8UC1);
	resize(imgMask_, imgMasknew, cv::Size(width_, height_), 0.0, 0.0, CV_INTER_AREA); // to shrink an image, it will generally look best with CV_INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)
	imgMask_ = imgMasknew;

	// update depth map image
	if (include_depth_map) {
		Mat imgDnew_ = cv::Mat::zeros(height_, width_, CV_8UC3);
		resize(imgD_, imgDnew_, cv::Size(width_, height_), 0.0, 0.0, CV_INTER_AREA); // to shrink an image, it will generally look best with CV_INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)
		imgD_ = imgDnew_;

		Mat imgDdiscnew_ = cv::Mat::zeros(height_, width_, CV_8UC3);
		resize(imgDdisc_, imgDdiscnew_, cv::Size(width_, height_), 0.0, 0.0, CV_INTER_AREA); // to shrink an image, it will generally look best with CV_INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)
		imgDdisc_ = imgDdiscnew_;
	}
}

// downsamples images and updates resolution info, camera intrinsics, and projection matrices accordingly; if include_depth_map flag is TRUE, also downsamples the associated depth map
void Camera::Downsample(int target_width, int target_height, bool include_depth_map) {
	// update resolution info
	double scale_factor_x = (double)width_ / (double)target_width;
	double scale_factor_y = (double)height_ / (double)target_height;
	width_ = target_width;
	height_ = target_height;

	// update image
	Mat imgTnew = cv::Mat::zeros(height_, width_, CV_8UC3);
	resize(imgT_, imgTnew, cv::Size(width_, height_), 0.0, 0.0, CV_INTER_AREA); // to shrink an image, it will generally look best with CV_INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)
	imgT_ = imgTnew;

	// update calibration and camera intrinsics
	calib_.RecalibrateNewSS(cv::Size(width_, height_));

	// update projection matrices
	UpdateCameraMatrices();

	// update image mask
	Mat imgMasknew = cv::Mat::zeros(height_, width_, CV_8UC1);
	resize(imgMask_, imgMasknew, cv::Size(width_, height_), 0.0, 0.0, CV_INTER_AREA); // to shrink an image, it will generally look best with CV_INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)
	imgMask_ = imgMasknew;

	// update depth map image
	if (include_depth_map) {
		Mat imgDnew = cv::Mat::zeros(height_, width_, CV_8UC3);
		resize(imgD_, imgDnew, cv::Size(width_, height_), 0.0, 0.0, CV_INTER_AREA); // to shrink an image, it will generally look best with CV_INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)
		imgD_ = imgDnew;

		Mat imgDdiscnew = cv::Mat::zeros(height_, width_, CV_8UC3);
		resize(imgDdisc_, imgDdiscnew, cv::Size(width_, height_), 0.0, 0.0, CV_INTER_AREA); // to shrink an image, it will generally look best with CV_INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)
		imgDdisc_ = imgDdiscnew;
	}
}

// inverse projects screen space points (screen space dimensions ss_width x ss_height) with depths given by imgD from screen space to world space using Kinv and RTinv, returning a 4xn matrix of type 64F of the corresponding points in world space
// imgD is a 2D depth image matrix of n points of type CV_32F whose depth values are in units that match Kinv
// Kinv is a 3x3 inverse calibration matrix of type CV_64F
// RTinv is a 4x4 inverse RT matrix of type CV_64F
// Iws must be a 4x(ss_width*ss_height) matrix of type CV_32F
// udpates Iws with the result: a 4xn matrix of type CV_64F of homogeneous world space points as (x,y,z,w)
void Camera::InverseProjectSStoWS(int ss_width, int ss_height, Mat *imgD, Mat *Kinv, Mat *RTinv, Mat *Iws) {
	bool timing = true; double t;
	if (timing) t = (double)getTickCount();
	
	assert(Iws->rows == 4 && Iws->cols == ss_width*ss_height && Iws->type() == CV_32F, "Camera::InverseProjectSStoWS() Iws must have 4 rows, columns numbering the screen space points, and be of type CV_32F");
	assert(imgD->type() == CV_32F, "Camera::InverseProjectSStoWS() imgD must be of type CV_32F");
	assert(Kinv->rows == 3 & Kinv->cols == 3 && Kinv->type() == CV_64F, "Camera::InverseProjectSStoWS() Kinv must be a 3x3 matrix of type CV_64F");
	assert(RTinv->rows == 4 & RTinv->cols == 4 && RTinv->type() == CV_64F, "Camera::InverseProjectSStoWS() RTinv must be a 4x4 matrix of type CV_64F");

	// scale u,v,w by the desired depth amount to get homogeneous coordinates that reflect the depth after transformation
	Mat I = ConstructSSCoords(ss_width, ss_height);
	float* pD;
	int width = imgD->cols;
	float *pIscaled0, *pIscaled1, *pIscaled2;
	pIscaled0 = I.ptr<float>(0);
	pIscaled1 = I.ptr<float>(1);
	pIscaled2 = I.ptr<float>(2);
	// use depth values from imgD_
	for (int r = 0; r < imgD->rows; r++) {
		pD = imgD->ptr<float>(r);
		for (int c = 0; c < imgD->cols; c++) {
			int idx = PixIndexFwd(Point(c, r), width);
			float lamda = pD[c];
			pIscaled0[idx] = lamda * pIscaled0[idx];
			pIscaled1[idx] = lamda * pIscaled1[idx];
			pIscaled2[idx] = lamda;
		}
	}

	// transform screen space to camera space - transform u,v to x,y, then add rows for z (equal to each depth value) and w (equal to 1.0)
	Mat Kinv_uvonly = cv::Mat::zeros(Size(3, 2), CV_64F);
	double *pDs1, *pDs2, *pDd1, *pDd2;
	pDs1 = Kinv->ptr<double>(0);
	pDs2 = Kinv->ptr<double>(1);
	pDd1 = Kinv_uvonly.ptr<double>(0);
	pDd2 = Kinv_uvonly.ptr<double>(1);
	for (int c = 0; c < 3; c++) {
		pDd1[c] = pDs1[c];
		pDd2[c] = pDs2[c];
	}
	Mat Ics_xyonly = Kinv_uvonly * I; // Ics is homogeneous 4xn matrix of camera space points
	I.release();
	Iws->setTo(1.); // Iws at this point is still in camera space until we multiply it below by inverse extrinsics
	Iws->row(0) = Ics_xyonly.row(0);
	Iws->row(1) = Ics_xyonly.row(1);
	Ics_xyonly.release();

	// in camera space, set z to depth value and w to 1 (already scaled x,y in homogeneous screen space)
	float *pIcs2;
	pIcs2 = Iws->ptr<float>(2);
	// use depth values from imgD_
	for (int r = 0; r < imgD->rows; r++) {
		pD = imgD->ptr<float>(r);
		for (int c = 0; c < imgD->cols; c++) {
			int idx = PixIndexFwd(Point(c, r), width);
			pIcs2[idx] = pD[c];
		}
	}

	// transform camera space positions to world space
	(*Iws) = (*RTinv) * (*Iws); // Iws is homogeneous 4xn matrix of world space points

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Camera::InverseProjectSStoWS() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

void Camera::InitWorldSpaceProjection() {
	Iws_ = cv::Mat::zeros(Size(width_*height_, 4), CV_64F); // 4xn matrix of homogeneous world space points where n is the number of pixels in imgT_
	InverseProjectSStoWS(width_, height_, &imgD_, &calib_.Kinv_, &RTinv_, &Iws_);
}

// Warping

// reprojects the camera view into a new camera with projection matrix P_dest
// only reprojects pixels for which there is depth info
// imgT is modified to include texture, imgD to include depth values, and imgMask to include binary mask values (255 pixel is opaque and 0 pixel is transparent), and all must be same size and types CV_8UC3 and CV_32F, respectively.
void Camera::Reproject(Mat *P_dest, Mat *imgT, Mat *imgD, Mat *imgMask) {
	bool debug = false;
	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	assert(imgT_.rows == imgD_.rows && imgT_.cols == imgD_.cols, "Camera::Reproject() imgD_, and imgMask_ must have same size");
	assert(imgMask_.rows == imgD_.rows && imgMask_.cols == imgD_.cols, "Camera::Reproject() imgD_, and imgMask_ must have same size");
	assert(imgT->rows == imgD->rows && imgT->cols == imgD->cols, "Camera::Reproject() imgT, imgD, and imgMask args must have same size");
	assert(imgMask->rows == imgD->rows && imgMask->cols == imgD->cols, "Camera::Reproject() imgT, imgD, and imgMask args must have same size");
	assert(imgT->type() == CV_8UC3, "Camera::Reproject() imgT must have type CV_8UC3");
	assert(imgD->type() == CV_32F, "Camera::Reproject() imgD must have type CV_32F");
	assert(imgMask->type() == CV_8UC1, "Camera::Reproject() imgMask must have type CV_8UC1");

	if (debug) cout << "Reprojecting from camera " << id_ << endl;

	// clear result matrices
	imgT->setTo(Vec3b(0, 0, 0));
	imgD->setTo(0.);
	imgMask->setTo(0);

	// reproject into world space coordinates to virtual camera's screen space
	Mat I_dest = (*P_dest) * Iws_; // note the matrix multiplication property: Ainv * A = A * Ainv

	// where there is a depth value, no depth discontinuity, it is not masked out, and it falls within screen space bounds for the virtual view, copy the texture and depth info over to imgT and imgD, respectively; project mask values as well, despite masking out imgT and imgD during projection, because errors in depth values will result in different images having different ideas of what is masked out and will need to use projected masks to make the final determination during NVS blending
	float* pD;
	Vec3b* pT; // pointer to texture image values
	uchar* pM; // pointer to mask image values
	double xproj, yproj, hproj;
	for (int r = 0; r<imgD_.rows; r++) { // traversing current depth image for this camera, which is of same size as current texture image and image mask for this camera
		pT = imgT_.ptr<Vec3b>(r);
		pD = imgD_.ptr<float>(r);
		pM = imgMask_.ptr<uchar>(r);
		for (int c = 0; c<imgD_.cols; c++) {
			if (pD[c] == 0.) continue; // no depth information
			if (imgDdisc_.at<uchar>(r, c) == 255) continue; // high depth discontinuity

			// retrieve position and normalize homogeneous coordinates
			int idx = PixIndexFwd(Point(c, r), width_);
			xproj = I_dest.at<double>(0, idx);
			yproj = I_dest.at<double>(1, idx);
			hproj = I_dest.at<double>(2, idx);
			xproj /= hproj;
			yproj /= hproj;

			// copy over qualifying data
			if ((xproj >= 0) &&
				(yproj >= 0) &&
				(xproj < imgD->cols) &&
				(yproj < imgD->rows)) { // ensure reprojection falls within I0 screen space

				Point p = RoundSSPoint(Point2d(xproj, yproj), imgD->cols, imgD->rows); // round to an integer pixel position; how do I interpolate bilinearly so that all floating point values are filled in before integer values are calculated?  Can that be done, knowing that not all values will be filled in?
				if (pM[c] != 0) {// pixel is not masked out
					imgD->at<float>(p.y, p.x) = pD[c]; // copy over depth data for imgD
					imgT->at<Vec3b>(p.y, p.x) = pT[c]; // copy over color data for imgT
				}
				imgMask->at<uchar>(p.y, p.x) = pM[c]; // copy over mask data for imgMask
			}
		}
	}

	if (debug) {
		display_mat(imgT, "imgT reproject");
		DepthMap::DisplayDepthImage(imgD);
		display_mat(imgMask, "imgMask reproject");
	}

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Camera::Reproject() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// Update functions

// updates member pos_
void Camera::UpdatePos() {
	pos_ = GetCameraPositionWS(&RTinv_);
}

// updates member view_dir_
void Camera::UpdateViewDir() {
	view_dir_ = GetCameraViewDirectionWS(&RTinv_);
}

// updates P_, Pinv_
// requires that calib.K_, R_, and T_ are set
void Camera::UpdateCameraMatrices() {
	ComputeProjectionMatrices(&calib_.K_, &calib_.Kinv_, &RT_, &RTinv_, P_, Pinv_);
}

// Convenience functions

// returns camera position in world space using RTinv inverse extrinsics matrix from argument
// RTinv must be a 4x4 matrix camera extrinsics matrix of type CV_64F
// note that RTinv holds position in world space while RT holds position in camera space
Point3d Camera::GetCameraPositionWS(Mat *RTinv) {
	assert(RTinv->rows == 4 && RTinv->cols == 4 && RTinv->type() == CV_64F, "Camera::GetCameraPositionWS() RTinv must be a 4x4 matrix of type CV_64F");
	Point3d pos;
	pos.x = RTinv->at<double>(0, 3);
	pos.y = RTinv->at<double>(1, 3);
	pos.z = RTinv->at<double>(2, 3);
	return pos;
}

// returns normalized camera view direction in world space using RTinv inverse extrinsics matrix from argument
// RTinv must be a 4x4 matrix camera extrinsics matrix of type CV_64F
// note that RTinv holds position in world space while RT holds position in camera space
Point3d Camera::GetCameraViewDirectionWS(Mat *RTinv) {
	assert(RTinv->rows == 4 && RTinv->cols == 4 && RTinv->type() == CV_64F, "Camera::GetCameraPositionWS() RTinv must be a 4x4 matrix of type CV_64F");
	Mat dir_cs = cv::Mat::zeros(4, 1, CV_64F); // view direction in camera space (0,0,1,0) since homogeneous value for a direction should be set to 0
	dir_cs.at<double>(2, 0) = 1.;
	Mat dir_ws = cv::Mat::zeros(4, 1, CV_64F); // view direction in world space
	dir_ws = (*RTinv) * dir_cs;

	Point3d view_dir;
	view_dir.x = dir_ws.at<double>(0, 0);
	view_dir.y = dir_ws.at<double>(1, 0);
	view_dir.z = dir_ws.at<double>(2, 0);

	normalize(view_dir);

	return view_dir;
}

// converts 3x3 camera intrinsics matrix to 3x4 version with right column of [0 0 0]T
Mat Camera::Extend_K(Mat *K) {
	Mat K_extended = cv::Mat::zeros(3, 4, CV_64F);
	Mat zc = cv::Mat::zeros(3, 1, CV_64F);
	hconcat(*K, zc, K_extended);
	return K_extended;
}

// converts 3x3 inverse camera intrinsics matrix to 4x3 version with bottom row [0 0 f]
Mat Camera::Extend_Kinv(Mat *Kinv) {
	Mat zr = cv::Mat::zeros(1, 3, CV_64F);
	zr.at<double>(0, 2) = 1;
	Mat Kinv_extended = cv::Mat::zeros(4, 3, CV_64F);
	vconcat(*Kinv, zr, Kinv_extended);
	return Kinv_extended;
}

// updates P and Pinv to be 3x4 and 4x3 and projection and inverse projection matrices, respectively, from camera intrinsics K and extrinsics RT
// requires that P and Pinv are 4x4 matrices of type CV_64FC1
// requires that K is a 3x3 camera intrinsics matrix of type CV_64FC1
// requires that RT is a 3x4 camera extrinsics matrix of type CV_64FC1
void Camera::ComputeProjectionMatrices(Mat *K, Mat *Kinv, Mat *RT, Mat *RTinv, Mat &P, Mat &Pinv) {
	bool debug = false;

	assert(P.rows == 3 && P.cols == 4 && P.type() == CV_64F, "Scene::NVS() P must be a 3x4 matrix of type CV_64F");
	assert(Pinv.rows == 4 && Pinv.cols == 3 && Pinv.type() == CV_64F, "Scene::NVS() Pinv must be a 4x3 matrix of type CV_64F");
	assert(K->rows == 3 && K->cols == 3 && K->type() == CV_64F, "Scene::NVS() K must be a 3x3 camera intrinsics matrix of type CV_64F");
	assert(RT->rows == 4 && RT->cols == 4 && RT->type() == CV_64F, "Scene::NVS() RT must be a 4x4 camera extrinsics matrix of type CV_64F");
	assert(RTinv->rows == 4 && RTinv->cols == 4 && RTinv->type() == CV_64F, "Scene::NVS() RT must be a 4x4 inverse camera extrinsics matrix of type CV_64F");

	// extended versions of K and Kinv (3x4 and 4x3, resp, instead of 3x3): K[I | 03]
	Mat K_extended = Extend_K(K);
	Mat Kinv_extended = Extend_Kinv(Kinv);

	// build P
	P = K_extended * (*RT);

	// build Pinv
	Pinv = (*RTinv) * Kinv_extended; // (AB)inv = Binv * Ainv

	if (debug) {
		cout << "K: " << endl << (*K) << endl << endl;
		cout << "K_extended: " << endl << K_extended << endl << endl;
		cout << "Kinv: " << endl << (*Kinv) << endl << endl;
		cout << "Kinv_extended: " << endl << Kinv_extended << endl << endl;
		cout << "RT: " << endl << (*RT) << endl << endl;
		cout << "RTinv: " << endl << (*RTinv) << endl << endl;
		cout << "P: " << endl << P << endl << endl;
		cout << "Pinv: " << endl << Pinv << endl << endl;

		Mat ident;
		ident = (*K) * (*Kinv);
		cout << "I = K * Kinv: " << endl << ident << endl << endl;
		ident = K_extended * Kinv_extended;
		cout << "I = K_extended * Kinv_extended: " << endl << ident << endl << endl;
		ident = (*RT) * (*RTinv);
		cout << "I = RT * RTinv: " << endl << ident << endl << endl;
		ident = P * Pinv;
		cout << "I = P * Pinv: " << endl << ident << endl << endl;

		cin.ignore();
	}
}

// given pointer to 3xn matrix PosSS that holds tranformed screen space positions where n is the number of pixels in width_*height_ and the order is column then row, updates the rounded transformed screen space coordinates for a given screen space pixel position and returns boolean whether is inside the target screen space or not
// sizeSS is the size of the screen space being transformed into
// returns true if transformed position is within target screen space bounds, false otherwise
// pt_ss_transformed is updated with the transformed position, rounded to the nearest pixel in the target screen space (note: will appear to be in target screen space regardless of return value, so pay attention to return value)
bool Camera::GetTransformedPosSS(Mat *PosSS, Point pt_ss, cv::Size sizeTargetSS, Point &pt_ss_transformed) {
	// retrieve position
	int idx = PixIndexFwd(pt_ss, width_);
	double xproj, yproj, hproj;
	xproj = PosSS->at<double>(0, idx);
	yproj = PosSS->at<double>(1, idx);
	hproj = PosSS->at<double>(2, idx);

	// normalize homogeneous coordinates
	xproj /= hproj;
	yproj /= hproj;

	pt_ss_transformed = RoundSSPoint(Point2d(xproj, yproj), sizeTargetSS.width, sizeTargetSS.height); // round to an integer pixel position

	if ((xproj >= 0) &&
		(yproj >= 0) &&
		(xproj < sizeTargetSS.width) &&
		(yproj < sizeTargetSS.height))
		return true;
	else return false;
}

// given pointer to 4xn matrix PosWS that holds tranformed world space positions where n is the number of pixels in width_*height_ and the order is column then row, returns the transformed world space coordinates for a given screen space pixel position
Point3d Camera::GetTransformedPosWS(Mat *PosWS, Point pt_ss) {
	// retrieve position
	int idx = PixIndexFwd(pt_ss, width_);
	double xproj, yproj, zproj, hproj;
	xproj = PosWS->at<double>(0, idx);
	yproj = PosWS->at<double>(1, idx);
	zproj = PosWS->at<double>(2, idx);
	hproj = PosWS->at<double>(3, idx);

	// normalize homogeneous coordinates
	xproj /= hproj;
	yproj /= hproj;
	zproj /= hproj;

	Point3d pt_ws = Point3d(xproj, yproj, zproj);
	return pt_ws;
}

// rounds the position of a sub-pixel point in screen space to an integer pixel point in screen space
Point Camera::RoundSSPoint(Point2d ptd, int width, int height)
{
	Point pt;
	pt.x = round(ptd.x, 0);
	pt.y = round(ptd.y, 0);

	if (pt.x < 0) pt.x = 0;
	else if (pt.x >= width) pt.x = width - 1;

	if (pt.y < 0) pt.y = 0;
	else if (pt.y >= height) pt.y = height - 1;

	return pt;
}

// returns bounding volume around the world space point cloud Iws_ and updates bv_min and bv_max
// assumes homogeneous coordinate of Iws_ is 1 for all points
void Camera::DeterminePointCloudBoundingVolume(Point3d &bv_min, Point3d &bv_max) {
	assert(imgMask_.rows == imgD_.rows && imgMask_.cols == imgD_.cols, "Camera::DeterminePointCloudBoundingVolume() imgD_, and imgMask_ must have same size");
	assert(Iws_.cols == imgD_.rows * imgD_.cols, "Camera::DeterminePointCloudBoundingVolume() imgD_, imgMask_, and Iws_ must have the same number of pixel locations");

	double *pX, *pY, *pZ;
	pX = Iws_.ptr<double>(0);
	pY = Iws_.ptr<double>(1);
	pZ = Iws_.ptr<double>(2);
	bv_min = Point3d(0., 0., 0.);
	bv_max = Point3d(0., 0., 0.);
	bool first = true;

	float* pD;
	uchar* pM; // pointer to mask image values
	uchar* pDM; // point to depth interpolation mask values
	for (int r = 0; r<imgD_.rows; r++) {
		pD = imgD_.ptr<float>(r);
		pM = imgMask_.ptr<uchar>(r);
		for (int c = 0; c < imgD_.cols; c++) {
			if (pD[c] == 0.) continue; // no depth information
			if (pM[c] == 0) continue; // masked out

			// retrieve position and normalize homogeneous coordinates
			int idx = PixIndexFwd(Point(c, r), width_);

			if (first) {
				bv_min.x = pX[idx];
				bv_min.y = pY[idx];
				bv_min.z = pZ[idx];
				bv_max.x = pX[idx];
				bv_max.y = pY[idx];
				bv_max.z = pZ[idx];
			}
			else {
				if (pX[idx] < bv_min.x) bv_min.x = pX[idx];
				if (pY[idx] < bv_min.y) bv_min.y = pY[idx];
				if (pZ[idx] < bv_min.z) bv_min.z = pZ[idx];
				if (pX[idx] > bv_max.x) bv_max.x = pX[idx];
				if (pY[idx] > bv_max.y) bv_max.y = pY[idx];
				if (pZ[idx] > bv_max.z) bv_max.z = pZ[idx];
			}
			first = false;
		}
	}
}

// I/O
char* Camera::GetFilename(std::string filepath, std::string scene_name)
{
	std::string *id_str = new std::string;
	convert_int_to_string(id_, id_str);
	std::string fn;
	fn = filepath + scene_name + "_cam" + *id_str + ".yml";
	char* fn_chars = convert_string_to_chars(fn);
	delete id_str;
	return fn_chars;
}
char* Camera::GetFilenameImage(std::string filepath, std::string scene_name, char* fn_img_chars)
{
	std::string *id_str = new std::string;
	convert_int_to_string(id_, id_str);
	std::string *fn_img = convert_chars_to_string(fn_img_chars);
	std::string fn;
	fn = filepath + scene_name + "_cam" + *id_str + "_" + *fn_img + ".png";
	char* fn_chars = convert_string_to_chars(fn);
	delete id_str;
	delete fn_img;
	return fn_chars;
}
char* Camera::GetFilenameMatrix(std::string filepath, std::string scene_name, char* fn_mat_chars)
{
	std::string *id_str = new std::string;
	convert_int_to_string(id_, id_str);
	std::string *fn_mat = convert_chars_to_string(fn_mat_chars);
	std::string fn;
	fn = filepath + scene_name + "_cam" + *id_str + "_" + *fn_mat + ".yml";
	char* fn_chars = convert_string_to_chars(fn);
	delete id_str;
	delete fn_mat;
	return fn_chars;
}
void Camera::Save_Iws(std::string scene_name)
{
	CvMat Iws = Iws_;
	char* fn_Iws = GetFilenameMatrix(GLOBAL_FILEPATH_DATA, scene_name, "Iws");
	cvSave(fn_Iws, &Iws);
	delete[] fn_Iws;
}
void Camera::Load_Iws(std::string scene_name)
{
	char* fn_Iws = GetFilenameMatrix(GLOBAL_FILEPATH_DATA, scene_name, "Iws");
	CvMat *Iws = (CvMat*)cvLoad(fn_Iws);
	Iws_ = cvarrToMat(Iws).clone();
	cvReleaseMat(&Iws);
	delete[] fn_Iws;
}

// Debugging

void Camera::Print() {
	cout << "Camera" << endl;
	cout << "ID " << id_ << endl;
	cout << "Filename " << fn_ << endl;
	cout << "Sensor ID " << sensor_id_ << endl;
	cout << "Width, height " << width_ << ", " << height_ << endl;
	calib_.Print();
	cout << "P = " << endl << " " << P_ << endl << endl;
	cout << "Pinv = " << endl << " " << Pinv_ << endl << endl;
	if (orientation_ == AGO_ORIGINAL) cout << "Orientation original" << endl;
	else cout << "Orientation size" << endl;
	if (enabled_) cout << "Enabled TRUE" << endl;
	else cout << "Enabled FALSE" << endl;
	if (posed_) cout << "Posed TRUE" << endl;
	else cout << "Posed FALSE" << endl;
	display_mat(&imgT_, "Camera image");
	if (has_depth_map_) DepthMap::DisplayDepthImage(&imgD_);
	cout << endl;
}