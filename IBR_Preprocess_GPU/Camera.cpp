#include "Camera.h"

// Constructors / destructor

Camera::Camera() {
	id_ = -1;
	fn_ = "";
	dm_ = new DepthMap();

	P_.setZero();
	Pinv_.setZero();
	RT_.setZero();
	RTinv_.setZero();

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
// it expects the transform being parsed is a camera location in Agisoft space, and therefore represents RTinv
// see Scene.h for description of AgisoftToWorld
void Camera::ParseAgisoftCameraExtrinsics(std::string s, Matrix4d AgisoftToWorld_, Matrix4d AgisoftToWorldinv_) {
	bool debug = false;

	RTinv_ = ParseString_Matrixd(s, 4, 4); // camera location in chunk coordinate space; note that the length of vector T in RT is the distance from the camera origin to the world origin, and so is the 4th column of RT-1.  But the former is expressed in coordinates from the camera's perspective (multiply camera space origin [0 0 0 1].transpose() by RT to get the camera's location expressed in chunk coordinate space), and the latter in coordinates from the world space origin's perspective (multiply world space origin [0 0 0 1].transpose() by RT-1 to get the world space origin's location expressed in camera space).  So the camera location in chunk coordinate space is RT-1.
	RT_ = RTinv_.inverse();
	
	RTinv_ = AgisoftToWorldinv_ * RTinv_;
	RT_ = RT_ * AgisoftToWorld_;

	UpdatePos();
	UpdateViewDir();

	if (debug) {
		cout << "cid: " << id_ << endl;
		cout << "AgisoftToWorld_: " << endl << AgisoftToWorld_ << endl << endl;
		cout << "AgisoftToWorldinv_: " << endl << AgisoftToWorldinv_ << endl << endl;
		cout << "RT: " << endl << RT_ << endl << endl;
		cout << "RTinv: " << endl << RTinv_ << endl << endl;
		cin.ignore();
	}
}

// set member values according to data in node argument from Agisoft doc.xml file
// see Scene.h for description of AgisoftToWorld
void Camera::Init(string scene_name, xml_node<> *camera_node, Matrix4d AgisoftToWorld_, Matrix4d AgisoftToWorldinv_) {
	bool debug = false;

	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	if (debug) cout << "Camera::Init() of id " << id_ << " for filename " << fn_ << endl;
	
	assert(strcmp(camera_node->name(), "camera") == 0);

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

	/*
	// can no longer use Agisoft-saved versions of masks because any gray lines are converted to full 255 uchar value and essentially lost
	curr_node = camera_node->first_node("frames")->first_node("frame")->first_node("mask");
	if (curr_node != 0) {
		s = curr_node->first_attribute("path")->value();
		if (!s.empty()) fn_mask_ = s.substr(s.find_last_of("\\/") + 1); // strip out relative path
	}
	*/
	fn_mask_ = "Masks/" + fn_;

	curr_node = camera_node->first_node("resolution");
	s = curr_node->first_attribute("width")->value();
	if (!s.empty()) width_ = convert_string_to_int(s);
	s = curr_node->first_attribute("height")->value();
	if (!s.empty()) height_ = convert_string_to_int(s);

	// retrieve orientation information
	int orientation = 1;
	curr_node = camera_node->first_node("frames");
	curr_node = curr_node->first_node("frame");
	curr_node = curr_node->first_node("image");
	curr_node = curr_node->first_node("meta");
	if (curr_node != 0) {
		curr_node = curr_node->first_node("property");
		while (curr_node != 0) {
			s = curr_node->first_attribute("name")->value();
			if (s == "Exif/Orientation") {
				s = curr_node->first_attribute("value")->value();
				orientation = convert_string_to_int(s);
				break;
			}
			curr_node = curr_node->next_sibling();
		}
	}
	switch (orientation) {
		case 8:
			orientation_ = AGO_ROTATED_RIGHT;
			break;
		case 6:
			orientation_ = AGO_ROTATED_LEFT;
			break;
		case 1:
			orientation_ = AGO_ORIGINAL;
			break;
		default:
			orientation_ = AGO_ORIGINAL;
			break;
	}

	// load image
	std::string fn_full = GLOBAL_FILEPATH_DATA + scene_name + "\\" + GLOBAL_FOLDER_PHOTOS + fn_;
	imgT_ = imread(fn_full, IMREAD_COLOR); // use cv::IMREAD_ANYDEPTH|cv::IMREAD_ANYCOLOR to retain depth and color types from original, or IMREAD_COLOR to convert to 8-bit color
	assert(imgT_.rows > 0 && imgT_.cols > 0);
	assert(imgT_.rows == height_ && imgT_.cols == width_);
	
	// load mask - must do this after orientation information is retrieved so height_ and width_ are set appropriately
	std::string fn_mask_full = GLOBAL_FILEPATH_DATA + scene_name + "\\" + GLOBAL_FOLDER_MASKS + fn_;
	imgMask_ = imread(fn_mask_full, IMREAD_GRAYSCALE);
	if (imgMask_.rows > 0 && imgMask_.cols > 0) { // image mask found
		assert(imgMask_.rows*imgMask_.cols == height_*width_); // mask orientation may be different, but dimensions should be same as imgT_
	} else {
		imgMask_ = cv::Mat(height_, width_, CV_8UC1, 255); // if mask not found, assume all pixels are "opaque" foreground pixels
	}
	imgMask_color_ = imread(fn_mask_full, IMREAD_COLOR);
	if (imgMask_color_.rows > 0 && imgMask_color_.cols > 0) { // image mask found
		assert(imgMask_color_.rows*imgMask_color_.cols == height_*width_); // mask orientation may be different, but dimensions should be same as imgT_
	}
	else {
		imgMask_color_ = cv::Mat(height_, width_, CV_8UC3, 255); // if mask not found, assume all pixels are "opaque" foreground pixels
	}

	// set up imgMask_valid_
	imgMask_valid_ = cv::Mat(height_, width_, CV_8UC1);
	imgMask_.copyTo(imgMask_valid_);

	// update imgMask_ to ensure any colored-in areas in imgMask_color_ are set appropriate to either masked-in or masked-out (since otherwise the pixels' grayscale values may be considered unpredicably above or below the masked-in threshold)
	Vec3b *pMc;
	uchar *pM, *pMv;
	Vec3b color;
	for (int r = 0; r < imgMask_.rows; r++) {
		pM = imgMask_.ptr<uchar>(r);
		pMv = imgMask_valid_.ptr<uchar>(r);
		pMc = imgMask_color_.ptr<Vec3b>(r);
		for (int c = 0; c < imgMask_.cols; c++) {
			color = pMc[c];
			if (((color[0] > GLOBAL_MIN_MASK_COLOR_HURDLE) &&
				((color[1] < GLOBAL_MIN_MASKSEG_LINEVAL) ||
				(color[2] < GLOBAL_MIN_MASKSEG_LINEVAL)))) { // blue
				pM[c] = 255;
				pMv[c] = 255;
			}
			else if (((color[2] > GLOBAL_MIN_MASK_COLOR_HURDLE) &&
				((color[0] < GLOBAL_MIN_MASKSEG_LINEVAL) ||
				(color[1] < GLOBAL_MIN_MASKSEG_LINEVAL)))) { // red
				pM[c] = 0;
				pMv[c] = 255;
			}
		}
	}

	//if (id_ == 27) debug = true; // remove *****************************************

	// orient mask to agree with the image
	switch (orientation) {
	case 8: // must be rotated right (transpose with horizontal flip)
		transpose(imgMask_, imgMask_);
		flip(imgMask_, imgMask_, 1);
		transpose(imgMask_color_, imgMask_color_);
		flip(imgMask_color_, imgMask_color_, 1);
		transpose(imgMask_valid_, imgMask_valid_);
		flip(imgMask_valid_, imgMask_valid_, 1);
		break;
	case 6: // must be rotated left (transpose with vertical flip)
		transpose(imgMask_, imgMask_);
		flip(imgMask_, imgMask_, 0);
		transpose(imgMask_color_, imgMask_color_);
		flip(imgMask_color_, imgMask_color_, 0);
		transpose(imgMask_valid_, imgMask_valid_);
		flip(imgMask_valid_, imgMask_valid_, 0);
		break;
	case 1: // fine as is - same as default
		break;
	default:
		break;
	}

	// imgMask_valid_ must be initialized first
	InitCloseup();

	// must do this after orientation information is retrieved and orientation_ is set so can update RT_ and RTinv_ accordingly
	curr_node = camera_node->first_node("transform");
	if (curr_node != 0) { // transforms may not have been computed for all cameras
		s = curr_node->value();
		ParseAgisoftCameraExtrinsics(s, AgisoftToWorld_, AgisoftToWorldinv_);
		posed_ = true;
	}

	if (debug) {
		cout << "rows " << imgT_.rows << ", cols " << imgT_.cols << endl;
		display_mat(&imgT_, "imgT_", orientation_);
		display_mat(&imgMask_, "imgMask_", orientation_);
		display_mat(&imgMask_color_, "imgMask_color_", orientation_);
		display_mat(&imgMask_valid_, "imgMask_valid_", orientation_);
	}

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Camera::Init() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// segment image mask based on lines in mask
void Camera::SegmentImage(int height, int width) {
	bool debug = false;

	if (debug) display_mat(&imgMask_, "imgMask_", orientation_);
	
	Mat mask = Mat::zeros(imgMask_.size(), CV_8UC1);
	cv::threshold(imgMask_, mask, GLOBAL_MAX_MASKSEG_LINEVAL, 255, THRESH_BINARY);
	if (debug) display_mat(&mask, "maskthresh", orientation_);

	// get rid of small markings and small holes
	int morph_type = MORPH_RECT; // MORPH_ELLIPSE
	int morph_size = 5;
	Mat element = getStructuringElement(morph_type,
		Size(2 * morph_size + 1, 2 * morph_size + 1),
		Point(morph_size, morph_size));
	cv::erode(mask, mask, element);
	//cv::dilate(mask, mask, element);

	if (debug) display_mat(&mask, "mask", orientation_);

	/*
	// delete labels for blobs smaller than a minimum area
	map<unsigned int, int> lcs = GetLabelCounts(&seg_);
	for (map<unsigned int, int>::iterator it = lcs.begin(); it != lcs.end(); ++it) {
		unsigned int label = (*it).first;
		int count = (*it).second;
		//cout << "label " << label << ", count " << count << endl;
		if (count < GLOBAL_MIN_SEGMENT_PIXELS) seg_ = (seg_.array() == label).select(Matrix<unsigned int, Dynamic, Dynamic>::Zero(seg_.rows(), seg_.cols()), seg_);
	}
	*/

	Mat masknew = cv::Mat::zeros(height, width, CV_8UC3);
	resize(mask, masknew, cv::Size(width, height), 0.0, 0.0, CV_INTER_AREA); // to shrink an image, it will generally look best with CV_INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)

	if (debug) display_mat(&masknew, "masknew", orientation_);

	seg_ = EigenOpenCV::SegmentUsingBlobs(masknew, seglabel_counts_);

	/*
	cout << "cid " << id_ << endl;
	for (map<unsigned int, int>::iterator it = seglabel_counts_.begin(); it != seglabel_counts_.end(); ++it) {
		unsigned int seglabel = (*it).first;
		int labelcount = (*it).second;
		double factor = 180000.;
		int num_cams_to_use = round(exp(max(0., static_cast<double>(GLOBAL_PIXEL_THRESHOLD_FOR_MIN_INPUT_CAMS - labelcount)) / factor), 0);
		cout << "for segment " << seglabel << " with " << labelcount << " pixels, using num_cams_to_use " << num_cams_to_use << endl;
	}
	cin.ignore();
	*/

	if (debug) DisplayImages::DisplaySegmentedImage(&seg_, height, width);

	Mat imgMasknew = cv::Mat::zeros(height, width, CV_8UC3); // note that this will be a resized mask and, in the process, its lines will lighten and so GLOBAL_MIN_MASKSEG_LINEVAL and GLOBAL_MAX_MASKSEG_LINEVAL may no longer be good measures for the lines.  However, it's only used to test masked-in (including on a segment line), which it still works fine for.  Can't use masknew for that purpose because of erosion - will never pass the check below to assign a segment label to the line edge cases missing one.
	resize(imgMask_, imgMasknew, cv::Size(width, height), 0.0, 0.0, CV_INTER_AREA); // to shrink an image, it will generally look best with CV_INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)

	// push 0-labeled edges among used pixels to one neighboring label or the other (if edge is more than 1 pixel thick, may need to search more than one pixel away); adhere to label more represented among neighbors; higher neighboring label number wins races; search close first, then widen search area until find a match or hit max search area
	int max_search_pixels_side = 5;
	int search_pixels_side;
	unsigned int nval, mask_val;
	Matrix<unsigned int, Dynamic, Dynamic> seg_copy = seg_;
	for (int c = max_search_pixels_side; c < (seg_.cols() - max_search_pixels_side); c++) {
		for (int r = max_search_pixels_side; r < (seg_.rows() - max_search_pixels_side); r++) {
			if (imgMasknew.at<uchar>(r, c) < GLOBAL_MIN_MASKSEG_LINEVAL) continue; // ensure masked-in (including on a segment line)
			if (seg_copy(r, c) != 0) continue; // ensure no label assigned

			map<unsigned int, int> lcounts;
			bool done = false;
			search_pixels_side = 1;
			while ((!done) &&
				(search_pixels_side <= max_search_pixels_side)) {
				for (int i = -1 * search_pixels_side; i <= search_pixels_side; i++) {
					for (int j = -1 * search_pixels_side; j <= search_pixels_side; j++) {
						mask_val = static_cast<unsigned int>(masknew.at<uchar>(r + j, c + i));
						nval = seg_copy(r + j, c + i);
						if ((mask_val <= GLOBAL_MAX_MASKSEG_LINEVAL) || // masked-out or on a line segment
							(nval == 0)) continue;
						if (lcounts.find(nval) == lcounts.end())
							lcounts[nval] = 1;
						else lcounts[nval] = lcounts[nval] + 1;
						done = true;
					}
				}
				search_pixels_side++;
			}

			unsigned int best_label;
			int high_count = 0;
			for (map<unsigned int, int>::iterator it = lcounts.begin(); it != lcounts.end(); ++it) {
				unsigned int l = (*it).first;
				int c = (*it).second;
				//cout << "l " << l << ", c " << c << endl;
				if (c > high_count) {
					best_label = l;
					high_count = c;
				}
				else if ((c == high_count) &&
					(l > best_label))
					best_label = l;
			}
			//cout << "best label " << best_label << endl;
			//cin.ignore();

			if (high_count > 0)
				seg_(r, c) = best_label;
		}
	}

	if (debug) DisplayImages::DisplaySegmentedImage(&seg_, height, width);
}

// initializes values for closeup_xmin_, closeup_xmax_, closeup_ymin_, closeup_ymax_ using valid mask data
// bool closeup_xmin, closeup_xmax, closeup_ymin, closeup_ymax: true in cases where photo is a close-up that doesn't fully capture the object within the screen space on the indicated side (value assigned by testing for valid masked-in pixels along the appropriate screen space side's edge)
// imgMask_valid_ must be initialized first
void Camera::InitCloseup() {
	closeup_xmin_ = false;
	for (int r = 0; r < height_; r++)
		if (imgMask_valid_.at<uchar>(r, 0) > GLOBAL_MIN_MASKSEG_LINEVAL) closeup_xmin_ = true;

	closeup_xmax_ = false;
	for (int r = 0; r < height_; r++)
		if (imgMask_valid_.at<uchar>(r, (width_-1)) > GLOBAL_MIN_MASKSEG_LINEVAL) closeup_xmax_ = true;

	closeup_ymin_ = false;
	for (int c = 0; c < width_; c++)
		if (imgMask_valid_.at<uchar>(0, c) > GLOBAL_MIN_MASKSEG_LINEVAL) closeup_ymin_ = true;

	closeup_ymax_ = false;
	for (int c = 0; c < width_; c++)
		if (imgMask_valid_.at<uchar>((height_-1), c) > GLOBAL_MIN_MASKSEG_LINEVAL) closeup_ymax_ = true;
}

// initializes projection matrices P and Pinv using the camera intrinsics matrix arg K
// arg _K is a camera intrinsics matrix of size (4, 4) and type CV_64F
void Camera::InitSensor(Sensor *sensor) {
	if (orientation_==AGO_ORIGINAL) assert(sensor->height_ == height_ && sensor->width_ == width_);
	else assert(sensor->height_*sensor->width_ == height_*width_);  // orientation may be different, but number of pixels should be same as imgT_
	
	calib_ = sensor->calib_.Copy();
	
	UpdateCameraMatrices();
	
}

// returns 3 by (ss_w*ss_h) data structure with homogeneous pixel positions for a screen space of pixel dimensions ss_w*ss_h assuming row-major order of indices
Matrix<float, 3, Dynamic> Camera::ConstructSSCoordsRM(int ss_w, int ss_h) {
	bool debug = false;

	Matrix<float, 3, Dynamic> I(3, ss_w*ss_h); // 3xn matrix of homogeneous screen space points where n is the number of pixels in imgT_
	I.row(2).setConstant(1.);

	int idx = 0;
	for (int r = 0; r < ss_h; r++) {
		for (int c = 0; c < ss_w; c++) {
			//idx = PixIndexFwdRM(Point(c, r), ss_w);
			I(0, idx) = (float)c;
			I(1, idx) = (float)r;
			idx++;
		}
	}

	return I;
}

// set depth map using data in node argument from Agisoft doc.xml file and associated .exr file
// depth downscale is the downward scale factor as given by Agisoft for the depth map from the original image
// uses calib_.focal_length, so calib_ must be set by calling InitSensor() before this function, so sensor_id must not be 0
// agisoft_to_world_scale_ is the scale factor associated with the change from Agisoft's space to our world space, given by the chunk transform in the xml file from Agisoft
void Camera::InitDepthMap(string scene_name, xml_node<> *depthmap_node, double agisoft_to_world_scale_, int depth_downscale) {
	bool debug = false;
	bool timing = true; double t;
	if (timing) t = (double)getTickCount();
	
	if (debug) cout << "Camera::InitDepthMap() of id " << id_ << " for filename " << fn_ << endl;

	if (!posed_) return;

	dm_->Init(scene_name, depthmap_node, agisoft_to_world_scale_, depth_downscale, orientation_);
	assert(id_ == dm_->cam_id_);

	has_depth_map_ = true;

	SegmentImage(dm_->depth_map_.rows(), dm_->depth_map_.cols());

	DownsampleToMatchDepthMap();	

	// now that mask and depth map are the same size and orientation, apply mask to depth map
	uchar *pM;
	for (int r = 0; r < dm_->depth_map_.rows(); r++) {
		pM = imgMask_.ptr<uchar>(r);
		for (int c = 0; c < dm_->depth_map_.cols(); c++) {
			if (pM[c] == 0) dm_->depth_map_(r, c) = 0.;
		}
	}

	dm_->UpdateMinMaxDepths(&imgMask_); // must do after downsampling so that imgMask_ and depth image are of the same size

	InitDepthMapDiscontinuities();
	
	if (debug) {
		dm_->DisplayDepthImage();
		DisplayImages::DisplaySegmentedImage(&seg_, height_, width_, orientation_);
	}

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
	assert(dm_->depth_map_.rows() == imgMask_.rows && dm_->depth_map_.cols() == imgMask_.cols);

	if (debug) dm_->DisplayDepthImage();
	
	uchar* pM;
	float depth, depth_wtd, sum_num, sum_denom;
	int dist;
	bool depth_found;
	for (int r = 0; r < dm_->depth_map_.rows(); r++) {
		pM = imgMask_.ptr<uchar>(r);
		for (int c = 0; c < dm_->depth_map_.cols(); c++) {
			if (dm_->depth_map_(r, c) != 0.) continue; // skip pixels that already have values
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
				dm_->depth_map_(r, c) = depth_wtd;
			}
		}
	}

	if (debug) dm_->DisplayDepthImage();
}

// traverses depth image imgD_ from p in direction step until encounters a pixel with a non-zero depth value that is not masked out in imgMask
// step must be one pixel-neighbor away in one of the 8 canonical lattice directions
// returns true if such a pixel is found, updates arg depth with its value, and updates dist with the distance to the non-zero pixel (diagonal steps count as distance 1); returns false if no such pixel is found within imgD_
// if a masked out pixel is encountered at any point, returns false
// if there is a depth value at p and p is not masked out, returns true and updates arg depth to p's depth
bool Camera::GetNearestDepth(Point p, Point step, int &dist, float &depth) {
	assert(dm_->depth_map_.rows() == imgMask_.rows && dm_->depth_map_.cols() == imgMask_.cols);
	assert(p.x >= 0 && p.y >= 0 && p.x < dm_->depth_map_.cols() && p.y < dm_->depth_map_.rows());
	assert(step.x != 0 || step.y != 0);
	assert(step.x == -1 || step.x == 0 || step.x == 1);
	assert(step.y == -1 || step.y == 0 || step.y == 1);
	
	bool found = false;
	dist = 0;
	depth = dm_->depth_map_(p.y, p.x);
	if ((depth != 0.) &&
		(imgMask_.at<uchar>(p.y, p.x) != 0))
		found = true;

	while (!found) {
		p.y = p.y + step.y;
		p.x = p.x + step.x;
		dist++;

		if ((p.x < 0) ||
			(p.y < 0) ||
			(p.x >= dm_->depth_map_.cols()) ||
			(p.y >= dm_->depth_map_.rows()))
			break;

		depth = dm_->depth_map_(p.y, p.x);
		if ((depth != 0.) &&
			(imgMask_.at<uchar>(p.y, p.x) != 0))
			found = true;
	}

	return found;
}

void Camera::UndistortPhotos() {
	bool debug = false;

	if ((!enabled_) ||
		(!posed_)) return; // for some reason, not currently stable on depth map viewing in these cases

	if (debug) display_mat(&imgT_, "imgT before undistortion", orientation_);
	if (debug) dm_->DisplayDepthImage();
	if (debug) display_mat(&imgMask_, "imgMask_ before undistortion", orientation_);
	if (debug) display_mat(&imgMask_color_, "imgMask_color_ before undistortion", orientation_);

	calib_.Undistort(imgT_);
	if (has_depth_map_) calib_.Undistort_DepthMap(dm_->depth_map_);
	calib_.Undistort(imgMask_);
	calib_.Undistort(imgMask_color_);

	if (debug) display_mat(&imgT_, "imgT after undistortion", orientation_);
	if ((debug) && (has_depth_map_)) dm_->DisplayDepthImage();
	if (debug) display_mat(&imgMask_, "imgMask_ after undistortion", orientation_);
	if (debug) display_mat(&imgMask_color_, "imgMask_color_ after undistortion", orientation_);
}

// returns a cv::Mat of type CV_8UC3 that is imgT after applying mask imgMask_
// note that depth values are not taken into account here at all - i.e. does NOT mask out zero depth pixels (pixels for which no depth information is available)
Mat Camera::MaskedImgT() {
	assert(imgT_.rows == imgMask_.rows && imgT_.cols == imgMask_.cols);
	Mat imgT_masked = cv::Mat::zeros(imgT_.rows, imgT_.cols, CV_8UC3);;
	Vec3b* pT;
	Vec3b* pTm;
	uchar* pM;
	for (int r = 0; r < imgMask_.rows; r++) {
		pT = imgT_.ptr<Vec3b>(r);
		pTm = imgT_masked.ptr<Vec3b>(r);
		pM = imgMask_.ptr<uchar>(r);
		for (int c = 0; c < imgMask_.cols; c++) {
			if (pM[c] == 0) continue; // unused pixel
			pTm[c] = pT[c];
		}
	}
	return imgT_masked;
}

// called by InitDepthMap() to initialize imgDdisc_
void Camera::InitDepthMapDiscontinuities() {
	bool debug = false;

	imgDdisc_ = cv::Mat::zeros(Size(dm_->depth_map_.cols(), dm_->depth_map_.rows()), CV_8UC1);

	unsigned char* p;
	for (int r = 0; r < imgDdisc_.rows; r++) {
		p = imgDdisc_.ptr<unsigned char>(r);
		for (int c = 0; c < imgDdisc_.cols; c++) {
			bool disc = DetermineDepthMapDiscontinuity(Point(c, r));
			if (disc) p[c] = 255;
		}
	}

	if (debug) display_mat(&imgDdisc_, "imgDdisc_", orientation_);
}

// tests pixel at point p for high depth map discontinuity and returns result boolean
bool Camera::DetermineDepthMapDiscontinuity(Point p) {
	if ((p.x <= 0) ||
		(p.y <= 0) ||
		(p.x >= (dm_->depth_map_.cols() - 2)) ||
		(p.y >= (dm_->depth_map_.rows() - 2)))
		return false; // needs 1-pixel border to compute; on border, assume no high depth discontinuities

	double td = (dm_->max_depth_ - dm_->min_nonzero_depth_) / GLOBAL_THRESHOLD_FACTOR_DEPTH_DISCONTINUITY;
	double sum = -9.0 * dm_->depth_map_(p.y, p.x);
	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			sum += dm_->depth_map_(p.y + j, p.x + i);
		}
	}
	if (abs(sum) > td) return true;
	else return false;
}

// downsamples images and updates resolution info, camera intrinsics, and projection matrices accordingly; if include_depth_map flag is TRUE, also downsamples the associated depth map
void Camera::DownsampleToMatchDepthMap() {
	// update resolution info
	int target_width = dm_->depth_map_.cols();
	int target_height = dm_->depth_map_.rows();
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

	// update image mask color
	Mat imgMaskcolornew = cv::Mat::zeros(height_, width_, CV_8UC3);
	resize(imgMask_color_, imgMaskcolornew, cv::Size(width_, height_), 0.0, 0.0, CV_INTER_AREA); // to shrink an image, it will generally look best with CV_INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)
	imgMask_color_ = imgMaskcolornew;

	// update image mask valid
	Mat imgMaskvalidnew = cv::Mat::zeros(height_, width_, CV_8UC1);
	resize(imgMask_valid_, imgMaskvalidnew, cv::Size(width_, height_), 0.0, 0.0, CV_INTER_AREA); // to shrink an image, it will generally look best with CV_INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)
	imgMask_valid_ = imgMaskvalidnew;
}

void Camera::DownsampleAll(float downsample_factor) {
	bool debug = false;

	int target_height = round(static_cast<float>(height_) * downsample_factor);
	int target_width = round(static_cast<float>(width_) * downsample_factor);

	// update resolution info
	width_ = target_width;
	height_ = target_height;

	// update depth map
	if (has_depth_map_)
		dm_->Downsample(downsample_factor);

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

	// update image mask color
	Mat imgMaskcolornew = cv::Mat::zeros(height_, width_, CV_8UC3);
	resize(imgMask_color_, imgMaskcolornew, cv::Size(width_, height_), 0.0, 0.0, CV_INTER_AREA); // to shrink an image, it will generally look best with CV_INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)
	imgMask_color_ = imgMaskcolornew;

	// update image mask valid
	Mat imgMaskvalidnew = cv::Mat::zeros(height_, width_, CV_8UC1);
	resize(imgMask_valid_, imgMaskvalidnew, cv::Size(width_, height_), 0.0, 0.0, CV_INTER_AREA); // to shrink an image, it will generally look best with CV_INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow) or CV_INTER_LINEAR (faster but still looks OK)
	imgMask_valid_ = imgMaskvalidnew;

	if (debug) {
		cout << "Camera::DownsampleAll() for cid " << id_ << endl;
		display_mat(&imgMask_, "imgMask_", orientation_);
		display_mat(&imgMask_color_, "imgMask_color_", orientation_);
	}

	dm_->UpdateMinMaxDepths(&imgMask_); // must do after downsampling so that imgMask_ and depth image are of the same size

	InitDepthMapDiscontinuities();
}

// returns 1x(ss_width*ss_height) matrix of pixel priority information, indexed in the same order as Iws_
// each priority is the distance from the corresponding world space point to the camera position in world space units; lower distances mean higher priority during rendering against data from other cameras
Matrix<float, 1, Dynamic> Camera::GetWSPriorities() {
	Matrix<float, Dynamic, Dynamic, RowMajor> dmWS = dm_->GetDepthMapInWorldSpace(); // Iws and related matrices are row-major interpretations of their 2D counterparts
	dmWS.resize(1, width_*height_);
	return dmWS;
}

// inverse projects screen space points (screen space dimensions ss_width x ss_height) with depths given by imgD from screen space to world space using Kinv and RTinv, updating a 4xn matrix of type float of the corresponding points in world space
// imgD is a 2D depth image matrix of size ss_height x ss_width (n points) whose depth values are in units that match Kinv
// Kinv is a 3x3 inverse calibration matrix of type CV_64F
// RTinv is a 4x4 inverse RT matrix of type CV_64F
// Iws must be a 4x(ss_width*ss_height) matrix
// updates Iws with homogeneous world space points as (x,y,z,w)
// expects everything in row-major
void Camera::InverseProjectSStoWS(int ss_width, int ss_height, Matrix<float, Dynamic, Dynamic> *depth_map, Matrix3d *Kinv, Matrix4d *RTinv, Matrix<float, 4, Dynamic> *Iws) {
	bool debug = false;

	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	assert(Iws->cols() == ss_width*ss_height);
	assert(depth_map->rows() == ss_height && depth_map->cols() == ss_width);

	Matrix<float, Dynamic, Dynamic, RowMajor> dm(ss_height, ss_width); // Iws and related matrices are row-major interpretations of their 2D counterparts
	dm = (*depth_map);
	dm.resize(1, ss_width * ss_height);
	
	if (debug) DebugPrintMatrix(depth_map, "depth_map");
	if (debug) DebugPrintMatrix(&dm, "dm");

	// scale u,v,w by the desired depth amount to get homogeneous coordinates that reflect the depth after transformation
	Matrix<float, 3, Dynamic> I = ConstructSSCoordsRM(ss_width, ss_height);
	// use depth values from depth_map
	I.row(0) = I.row(0).cwiseProduct(dm);
	I.row(1) = I.row(1).cwiseProduct(dm);
	I.row(2) = I.row(2).cwiseProduct(dm);

	if (debug) DebugPrintMatrix(&I, "I");
	
	// transform screen space to camera space - transform u,v to x,y, then add rows for z (equal to each depth value) and w (equal to 1.0)
	Matrix<float, 2, 3> Kinv_uvonly;
	Kinv_uvonly.row(0) = Kinv->row(0).cast<float>();
	Kinv_uvonly.row(1) = Kinv->row(1).cast<float>();
	Matrix<float, 2, Dynamic> Ics_xyonly = Kinv_uvonly * I; // Ics is homogeneous 4xn matrix of camera space points
	I.resize(3, 0);
	Iws->setOnes(); // Iws at this point is still in camera space until we multiply it below by inverse extrinsics
	Iws->row(0) = Ics_xyonly.row(0);
	Iws->row(1) = Ics_xyonly.row(1);
	Ics_xyonly.resize(2, 0);

	// in camera space, set z to depth value and w to 1 (already scaled x,y in homogeneous screen space)
	// use depth values from dm
	Iws->row(2) = dm;
	dm.resize(0, 0);

	if (debug) {
		cout << "Camera::InverseProjectSStoWS() Iws camera space positions (intermediate value assignment before conversion to world space)" << endl;
		DebugPrintMatrix(Iws, "Iws");
	}
	
	// transform camera space positions to world space
	(*Iws) = (*RTinv).cast<float>() * (*Iws); // Iws is homogeneous 4xn matrix of world space points; RTinv includes transformation from Agisoft space to world space
	Matrix<float, 1, Dynamic> H = Iws->row(3).array().inverse();
	
	if (debug) {
		Matrix4f RTinv2 = (*RTinv).cast<float>();
		DebugPrintMatrix(&RTinv2, "RTinv2");
		cout << "Camera::InverseProjectSStoWS() Iws world space positions" << endl;
		DebugPrintMatrix(Iws, "Iws");
		DebugPrintMatrix(&H, "H");

	}

	// normalize by homogeneous value
	Iws->row(0) = Iws->row(0).cwiseProduct(H);
	Iws->row(1) = Iws->row(1).cwiseProduct(H);
	Iws->row(2) = Iws->row(2).cwiseProduct(H);
	Iws->row(3).setOnes();

	if (debug) {
		cout << "Camera::InverseProjectSStoWS() Iws world space positions normalized" << endl;
		DebugPrintMatrix(Iws, "Iws");
	}

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Camera::InverseProjectSStoWS() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

void Camera::InitWorldSpaceProjection() {
	bool debug = false;

	Iws_.resize(4, 0); // clear ay existing data
	Iws_.resize(4, width_*height_); // 4xn matrix of homogeneous world space points where n is the number of pixels in imgT_
	InverseProjectSStoWS(width_, height_, &dm_->depth_map_, &calib_.Kinv_, &RTinv_, &Iws_);
	UpdatePointCloudBoundingVolume();
}

// Warping

// reprojects the camera view into a new camera with projection matrix P_dest
// only reprojects pixels for which there is depth info
// imgT is modified to include texture, imgD to include depth values in this virtual view's camera space, and imgMask to include binary mask values (255 pixel is opaque and 0 pixel is transparent), and all must be same size and types CV_8UC3 and CV_32F, respectively.
// can't really project the mask because there is no depth information for background pixels, so no way to project them and tell the difference between them and holes that appear in the projection due to revealed occlusions; it would be nice to reverse project destination pixels to source to determine whether they're masked-in or masked-out, but we have no destination depth information, which would be necessary for the reverse projection; but can create mask that has both kinds of empty pixels masked-out so we don't confuse them with black pixels that actually hold reprojected color
void Camera::Reproject(Matrix<double, 3, 4> *P_dest, Matrix4d *RT_dest, Mat *imgT, Matrix<float, Dynamic, Dynamic> *depth_map, Mat *imgMask) {
	bool debug = false;
	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	assert(imgT_.rows == dm_->depth_map_.rows() && imgT_.cols == dm_->depth_map_.cols());
	assert(imgMask_.rows == imgT_.rows && imgMask_.cols == imgT_.cols);
	assert(imgT->rows == depth_map->rows() && imgT->cols == depth_map->cols());
	assert(imgMask->rows == depth_map->rows() && imgMask->cols == depth_map->cols());
	assert(imgT->type() == CV_8UC3);
	assert(imgMask->type() == CV_8UC1);

	if (debug) cout << "Reprojecting from camera " << id_ << endl;
	
	// clear result matrices
	imgT->setTo(Vec3b(0, 0, 0));
	depth_map->setZero();
	imgMask->setTo(0);
	
	// reproject world space coordinates to virtual camera's screen space
	Matrix<float, 3, Dynamic> I_dest = (*P_dest).cast<float>() * Iws_; // note the matrix multiplication property: Ainv * A = A * Ainv
	Matrix<float, 1, Dynamic> H = I_dest.row(2).array().inverse();
	I_dest.row(0) = I_dest.row(0).cwiseProduct(H);
	I_dest.row(1) = I_dest.row(1).cwiseProduct(H);

	// reproject world space coordinates to virtual camera's camera space so can record virtual CS depths
	Matrix<float, 4, Dynamic> Ics_dest = (*RT_dest).cast<float>() * Iws_; // note the matrix multiplication property: Ainv * A = A * Ainv
	Ics_dest.row(2) = Ics_dest.row(2).cwiseQuotient(Ics_dest.row(3));

	// where there is a depth value, no depth discontinuity, it is not masked out, and it falls within screen space bounds for the virtual view, copy the texture and depth info over to imgT and imgD, respectively; project mask values as well, despite masking out imgT and imgD during projection, because errors in depth values will result in different images having different ideas of what is masked out and will need to use projected masks to make the final determination during NVS blending
	Vec3b* pT; // pointer to texture image values
	uchar* pM; // pointer to mask image values
	double xproj, yproj, hproj;
	for (int r = 0; r< dm_->depth_map_.rows(); r++) { // traversing current depth image for this camera, which is of same size as current texture image and image mask for this camera
		pT = imgT_.ptr<Vec3b>(r);
		pM = imgMask_.ptr<uchar>(r);
		for (int c = 0; c< dm_->depth_map_.cols(); c++) {
			if (abs(dm_->depth_map_(r, c)) < GLOBAL_FLOAT_ERROR) continue; // no depth information
			if (imgDdisc_.at<uchar>(r, c) == 255) continue; // high depth discontinuity

			/*
			if (imgDdisc_.at<uchar>(r, c) == 255) {
				// retrieve position and normalize homogeneous coordinates
				int idx2 = PixIndexFwdRM(Point(c, r), width_);
				xproj = I_dest(0, idx2);
				yproj = I_dest(1, idx2);
				hproj = I_dest(2, idx2);
				xproj /= hproj;
				yproj /= hproj;

				// copy over qualifying data
				if ((xproj >= 0) &&
					(yproj >= 0) &&
					(xproj < depth_map->cols()) &&
					(yproj < depth_map->rows())) { // ensure reprojection falls within I0 screen space

					Point p2 = RoundSSPoint(Point2d(xproj, yproj), depth_map->cols(), depth_map->rows()); // round to an integer pixel position; how do I interpolate bilinearly so that all floating point values are filled in before integer values are calculated?  Can that be done, knowing that not all values will be filled in?
					imgT->at<Vec3b>(p2.y, p2.x) = Vec3b(0, 0, 255);
				}
				continue; // high depth discontinuity
			}
			*/

			// retrieve position and normalize homogeneous coordinates
			int idx = PixIndexFwdRM(Point(c, r), width_);
			xproj = I_dest(0, idx);
			yproj = I_dest(1, idx);
			hproj = I_dest(2, idx);
			xproj /= hproj;
			yproj /= hproj;

			// copy over qualifying data
			if ((xproj >= 0) &&
				(yproj >= 0) &&
				(xproj < depth_map->cols()) &&
				(yproj < depth_map->rows())) { // ensure reprojection falls within I0 screen space

				Point p = RoundSSPoint(Point2d(xproj, yproj), depth_map->cols(), depth_map->rows()); // round to an integer pixel position; how do I interpolate bilinearly so that all floating point values are filled in before integer values are calculated?  Can that be done, knowing that not all values will be filled in?
				if (pM[c] != 0) { // pixel is not masked out
					(*depth_map)(p.y, p.x) = Ics_dest(2, idx); // copy over depth data for imgD
					imgT->at<Vec3b>(p.y, p.x) = pT[c]; // copy over color data for imgT
				}
				imgMask->at<uchar>(p.y, p.x) = pM[c]; // copy over mask data for imgMask
			}
		}
	}
	
	if (debug) {
		display_mat(imgT, "imgT reproject", orientation_);
		dm_->DisplayDepthImage();
		display_mat(imgMask, "imgMask reproject", orientation_);
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
// requires that calib.K_, RT_, and RTinv_ are set
void Camera::UpdateCameraMatrices() {
	ComputeProjectionMatrices(&calib_.K_, &calib_.Kinv_, &RT_, &RTinv_, &P_, &Pinv_);
}

// Convenience functions

// returns camera position in world space using RTinv inverse extrinsics matrix from argument
// RTinv must be a 4x4 matrix camera extrinsics matrix of type CV_64F
// note that RTinv holds position in world space while RT holds position in camera space
Point3d Camera::GetCameraPositionWS(Matrix4d *RTinv) {
	Point3d pos;
	pos.x = (*RTinv)(0, 3);
	pos.y = (*RTinv)(1, 3);
	pos.z = (*RTinv)(2, 3);
	return pos;
}

// returns a normalized camera view direction in world space using RTinv inverse extrinsics matrix from argument
// RTinv must be a 4x4 matrix camera extrinsics matrix of type CV_64F
// note that RTinv holds position in world space while RT holds position in camera space
Point3f Camera::GetCameraViewDirectionWS(Matrix4f *RTinv) {
	Matrix<float, 4, 1> dir_cs; // view direction in camera space (0,0,1,0) since homogeneous value for a direction should be set to 0
	dir_cs.setZero();
	dir_cs(2, 0) = 1.;
	Matrix<float, 4, 1> dir_ws; // view direction in world space
	dir_ws = (*RTinv) * dir_cs;

	Point3f view_dir;
	view_dir.x = dir_ws(0, 0);
	view_dir.y = dir_ws(1, 0);
	view_dir.z = dir_ws(2, 0);

	normalize(view_dir);

	return view_dir;
}

// returns a normalized camera view direction in world space using RTinv inverse extrinsics matrix from argument
// RTinv must be a 4x4 matrix camera extrinsics matrix of type CV_64F
// note that RTinv holds position in world space while RT holds position in camera space
Point3d Camera::GetCameraViewDirectionWS(Matrix4d *RTinv) {
	Matrix<double, 4, 1> dir_cs; // view direction in camera space (0,0,1,0) since homogeneous value for a direction should be set to 0
	dir_cs.setZero();
	dir_cs(2, 0) = 1.;
	Matrix<double, 4, 1> dir_ws; // view direction in world space
	dir_ws = (*RTinv) * dir_cs;

	Point3d view_dir;
	view_dir.x = dir_ws(0, 0);
	view_dir.y = dir_ws(1, 0);
	view_dir.z = dir_ws(2, 0);

	normalize(view_dir);

	return view_dir;
}

// returns a normalized camera up direction in world space using RTinv inverse extrinsics matrix from argument
// RTinv must be a 4x4 matrix camera extrinsics matrix of type CV_64F
// note that RTinv holds position in world space while RT holds position in camera space
Point3f Camera::GetCameraUpDirectionWS(Matrix4f *RTinv) {
	Matrix<float, 4, 1> dir_cs; // up direction in camera space (0,1,0,0) since homogeneous value for a direction should be set to 0
	dir_cs.setZero();
	dir_cs(1, 0) = 1.;
	Matrix<float, 4, 1> dir_ws; // view direction in world space
	dir_ws = (*RTinv) * dir_cs;

	Point3f up_dir;
	up_dir.x = dir_ws(0, 0);
	up_dir.y = dir_ws(1, 0);
	up_dir.z = dir_ws(2, 0);

	normalize(up_dir);

	return up_dir;
}

// returns a normalized camera up direction in world space using RTinv inverse extrinsics matrix from argument
// RTinv must be a 4x4 matrix camera extrinsics matrix of type CV_64F
// note that RTinv holds position in world space while RT holds position in camera space
Point3d Camera::GetCameraUpDirectionWS(Matrix4d *RTinv) {
	Matrix<double, 4, 1> dir_cs; // up direction in camera space (0,1,0,0) since homogeneous value for a direction should be set to 0
	dir_cs.setZero();
	dir_cs(1, 0) = 1.;
	Matrix<double, 4, 1> dir_ws; // view direction in world space
	dir_ws = (*RTinv) * dir_cs;

	Point3d up_dir;
	up_dir.x = dir_ws(0, 0);
	up_dir.y = dir_ws(1, 0);
	up_dir.z = dir_ws(2, 0);

	normalize(up_dir);

	return up_dir;
}

// converts 3x3 camera intrinsics matrix to 3x4 version with right column of [0 0 0]T
Matrix<double,3,4> Camera::Extend_K(Matrix3d *K) {
	Matrix<double, 3, 4> K_ext;
	K_ext.block(0, 0, 3, 3) << (*K);
	K_ext.col(3) << 0., 0., 0.;
	return K_ext;
}

// converts 3x3 inverse camera intrinsics matrix to 4x3 version with bottom row [0 0 f]
Matrix<double, 4, 3> Camera::Extend_Kinv(Matrix3d *Kinv) {
	Matrix<double, 4, 3> Kinv_ext;
	Kinv_ext.block(0, 0, 3, 3) << (*Kinv);
	Kinv_ext.row(3) << 0., 0., 1.;
	return Kinv_ext;
}

// updates P and Pinv to be 3x4 and 4x3 and projection and inverse projection matrices, respectively, from camera intrinsics K and extrinsics RT
// requires that P and Pinv are 4x4 matrices of type CV_64FC1
// requires that K is a 3x3 camera intrinsics matrix of type CV_64FC1
// requires that RT is a 3x4 camera extrinsics matrix of type CV_64FC1
void Camera::ComputeProjectionMatrices(Matrix3d *K, Matrix3d *Kinv, Matrix4d *RT, Matrix4d *RTinv, Matrix<double, 3, 4> *P, Matrix<double, 4, 3> *Pinv) {
	bool debug = false;
	
	// extended versions of K and Kinv (3x4 and 4x3, resp, instead of 3x3): K[I | 03]
	Matrix<double, 3, 4> K_extended = Extend_K(K);
	Matrix<double, 4, 3> Kinv_extended = Extend_Kinv(Kinv);

	// build P
	(*P) = K_extended * (*RT);

	// build Pinv
	(*Pinv) = (*RTinv) * Kinv_extended; // (AB)inv = Binv * Ainv
	
	if (debug) {
		cout << "K: " << endl << (*K) << endl << endl;
		cout << "K_extended: " << endl << K_extended << endl << endl;
		cout << "Kinv: " << endl << (*Kinv) << endl << endl;
		cout << "Kinv_extended: " << endl << Kinv_extended << endl << endl;
		cout << "RT: " << endl << (*RT) << endl << endl;
		cout << "RTinv: " << endl << (*RTinv) << endl << endl;
		cout << "P: " << endl << (*P) << endl << endl;
		cout << "Pinv: " << endl << (*Pinv) << endl << endl;

		Matrix3d ident3d;
		ident3d = (*K) * (*Kinv);
		cout << "I = K * Kinv: " << endl << ident3d << endl << endl;
		ident3d = K_extended * Kinv_extended;
		cout << "I = K_extended * Kinv_extended: " << endl << ident3d << endl << endl;
		Matrix4d ident4d;
		ident4d = (*RT) * (*RTinv);
		cout << "I = RT * RTinv: " << endl << ident4d << endl << endl;
		ident3d = (*P) * (*Pinv);
		cout << "I = P * Pinv: " << endl << ident3d << endl << endl;

		cin.ignore();
	}
	
}

// given pointer to 3xn matrix PosSS that holds tranformed screen space positions where n is the number of pixels in width_*height_ and the order is column then row, updates the rounded transformed screen space coordinates for a given screen space pixel position and returns boolean whether is inside the target screen space or not
// sizeSS is the size of the screen space being transformed into
// returns true if transformed position is within target screen space bounds, false otherwise
// pt_ss_transformed is updated with the transformed position, rounded to the nearest pixel in the target screen space (note: will appear to be in target screen space regardless of return value, so pay attention to return value)
bool Camera::GetTransformedPosSS(Matrix<float, 3, Dynamic> *PosSS, Point pt_ss, cv::Size sizeTargetSS, Point &pt_ss_transformed) {
	// retrieve position
	int idx = PixIndexFwdRM(pt_ss, width_);
	double xproj, yproj, hproj;
	xproj = (*PosSS)(0, idx);
	yproj = (*PosSS)(1, idx);
	hproj = (*PosSS)(2, idx);

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
Point3d Camera::GetTransformedPosWS(Matrix<float, 4, Dynamic> *PosWS, Point pt_ss) {
	// retrieve position
	int idx = PixIndexFwdRM(pt_ss, width_);
	double xproj, yproj, zproj, hproj;
	xproj = (*PosWS)(0, idx);
	yproj = (*PosWS)(1, idx);
	zproj = (*PosWS)(2, idx);
	hproj = (*PosWS)(3, idx);

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
void Camera::UpdatePointCloudBoundingVolume() {
	bool debug = false;

	assert(imgMask_.rows == dm_->depth_map_.rows() && imgMask_.cols == dm_->depth_map_.cols());
	assert(Iws_.cols() == dm_->depth_map_.rows() * dm_->depth_map_.cols());
	
	bv_min_ = Point3d(0., 0., 0.);
	bv_max_ = Point3d(0., 0., 0.);
	bool first = true;

	
	// can't do it this way because need to obey mask
	//bv_min_.x = Iws_.row(0).minCoeff();
	//bv_min_.y = Iws_.row(1).minCoeff();
	//bv_min_.z = Iws_.row(2).minCoeff();
	//bv_max_.x = Iws_.row(0).maxCoeff();
	//bv_max_.y = Iws_.row(1).maxCoeff();
	//bv_max_.z = Iws_.row(2).maxCoeff();

	uchar* pM; // pointer to mask image values
	float h;
	for (int r = 0; r< dm_->depth_map_.rows(); r++) {
		pM = imgMask_.ptr<uchar>(r);
		for (int c = 0; c < dm_->depth_map_.cols(); c++) {
			if (dm_->depth_map_(r, c) == 0.) continue; // no depth information
			if (pM[c] == 0) continue; // masked out

			// retrieve position and normalize homogeneous coordinates
			int idx = PixIndexFwdRM(Point(c, r), width_);
			h = Iws_(3, idx);
			if (h != 1.) {
				Iws_(0, idx) = Iws_(0, idx) / h;
				Iws_(1, idx) = Iws_(1, idx) / h;
				Iws_(2, idx) = Iws_(2, idx) / h;
				Iws_(3, idx) = Iws_(3, idx) / h;
			}

			if (first) {
				if (debug) cout << Iws_.col(idx) << endl << endl;
				bv_min_.x = Iws_(0, idx);
				bv_min_.y = Iws_(1, idx);
				bv_min_.z = Iws_(2, idx);
				bv_max_.x = Iws_(0, idx);
				bv_max_.y = Iws_(1, idx);
				bv_max_.z = Iws_(2, idx);
			}
			else {
				if (debug) cout << Iws_.col(idx) << endl << endl;
				if (Iws_(0, idx) < bv_min_.x) bv_min_.x = Iws_(0, idx);
				if (Iws_(1, idx) < bv_min_.y) bv_min_.y = Iws_(1, idx);
				if (Iws_(2, idx) < bv_min_.z) bv_min_.z = Iws_(2, idx);
				if (Iws_(0, idx) > bv_max_.x) bv_max_.x = Iws_(0, idx);
				if (Iws_(1, idx) > bv_max_.y) bv_max_.y = Iws_(1, idx);
				if (Iws_(1, idx) > bv_max_.z) bv_max_.z = Iws_(2, idx);
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
	fn = filepath + scene_name + "_cam" + *id_str + ".adf";
	char* fn_chars = convert_string_to_chars(fn);
	delete id_str;
	return fn_chars;
}
char* Camera::GetFilenameMatrix(std::string filepath, std::string scene_name, char* fn_mat_chars)
{
	std::string *id_str = new std::string;
	convert_int_to_string(id_, id_str);
	std::string fn;
	fn = filepath + scene_name + "_cam" + *id_str + "_matrix_" + fn_mat_chars + ".yml";
	char* fn_chars = convert_string_to_chars(fn);
	delete id_str;
	return fn_chars;
}

void Camera::RLE_WriteCount(int contig_pixel_count, std::vector<unsigned short> *rls) {
	if (contig_pixel_count > USHRT_MAX) {
		while (contig_pixel_count > USHRT_MAX) {
			rls->push_back(USHRT_MAX);
			rls->push_back(0); // intervening count of type used pixel
			contig_pixel_count -= USHRT_MAX;
			if (contig_pixel_count <= USHRT_MAX) rls->push_back(contig_pixel_count);
		}
	}
	else rls->push_back(contig_pixel_count);
}

// note: all matrices are indexed in row-major pixel order (across a row, then down to the next one), except Kinv and RTinv which are stored column-major pixel order in accordance with the Eigen default
// note: to qualify as a used pixel, it must be both not masked out (mask val != 0) and have a valid depth (depth val != 0.)
// scene_name cannot be more than 49 chars (to leave room for \0 terminator at the end)
// filename format of 2 files written:
//		1. GLOBAL_FILEPATH_DATA + scene_name + "\\" + scene_name + "_cam" + to_string(id_) + ".adf"
//		2. GLOBAL_FILEPATH_DATA + scene_name + "\\" + scene_name + "_cam" + to_string(id_) + ".jpg"
// saves compressed data needed for rendering as a binary file with the following data, in order:
// 1. 50 character (byte) asset name that is binary-zero-terminated (char \0); followed by binary-zero-terminated (char \0) 8-character (byte) "Check01"
// 2. int camera ID; followed by binary-zero-terminated (char \0) 8-character (byte) "Check02"
// 3. unsigned int for # rows, unsigned int for # cols; followed by binary-zero-terminated (char \0) 8-character (byte) "Check03"
// 4. bounding volume as floats of in order of: xmin, ymin, zmin, xmax, ymax, zmax; followed by binary-zero-terminated (char \0) 8-character (byte) "Check04"
// 5. disparity information in order of: minimum disparity, maximum disparity, disparity step; followed by binary-zero-terminated (char \0) 8-character (byte) "Check05"
// 6. 3x3 inverse K matrix of 9 floats, then 4x4 inverse RT matrix of 16 floats; followed by binary-zero-terminated (char \0) 8-character (byte) "Check06"
// 7. unsigned int number of used pixels, unsigned int number of run length unsigned shorts, unsigned int number of unused rows at top; followed by binary-zero-terminated (char \0) 8-character (byte) "Check07"
// 8. run lengths; start with an unsigned short containing the number of contiguous unused pixels' depths in the current raster row (starting at the first non-blank row) to follow (0 if starts with used pixels), followed by the number of contiguous used pixels next, followed by the number of contiguous unused pixels' depths to follow, etc. repeated to the end of the image.  If a count exceeds USHRT_MAX, write a USHRT_MAX count, then a zero, then repeat for the remainder. ; followed by binary-zero-terminated (char \0) 8-character (byte) "Check08"
// 9. RLE pixel quantized disparity labels, each as an unsigned short; they are in raster-scan order, skipping any unused pixels; followed by binary-zero-terminated (char \0) 8-character (byte) "Check09"
// saves a separate image file after masking using imgMask_ and excluding any pixels for which depth values aren't available
void Camera::SaveRenderingDataRLE(std::string scene_name, float min_disp, float max_disp, float disp_step) {
	assert(scene_name.length() < 50);

	bool debug = false;

	std::string fn = GLOBAL_FILEPATH_DATA + scene_name + "\\" + scene_name + "_cam" + to_string(id_) + ".adf";
	FILE* pFile = fopen(fn.c_str() , "wb"); // write binary mode
	float f;
	char check_chars[8] = { 'C', 'h', 'e', 'c', 'k', '0', '0', '\0' };

	// 1. write 50 character (byte) asset name that is binary-zero-terminated (char \0)
	char asset_name[50];
	std::size_t length = scene_name.copy(asset_name, scene_name.length(), 0);
	asset_name[length] = '\0';
	std::fwrite((void*)asset_name, sizeof(char), 50, pFile);
	check_chars[6] = '1';
	std::fwrite((void*)check_chars, sizeof(char), 8, pFile);

	// 2. int camera ID
	std::fwrite((void*)&id_, sizeof(int), 1, pFile);
	check_chars[6] = '2';
	std::fwrite((void*)check_chars, sizeof(char), 8, pFile);

	// 3. unsigned int for # rows, unsigned int for # cols
	std::fwrite((void*)&height_, sizeof(unsigned int), 1, pFile);
	std::fwrite((void*)&width_, sizeof(unsigned int), 1, pFile);
	check_chars[6] = '3';
	std::fwrite((void*)check_chars, sizeof(char), 8, pFile);

	// 4. bounding volume as floats of in order of: xmin, ymin, zmin, xmax, ymax, zmax
	Point3d bv_min, bv_max;
	UpdatePointCloudBoundingVolume();
	f = bv_min_.x;
	std::fwrite((void*)&f, sizeof(float), 1, pFile);
	f = bv_min_.y;
	std::fwrite((void*)&f, sizeof(float), 1, pFile);
	f = bv_min_.z;
	std::fwrite((void*)&f, sizeof(float), 1, pFile);
	f = bv_max_.x;
	std::fwrite((void*)&f, sizeof(float), 1, pFile);
	f = bv_max_.y;
	std::fwrite((void*)&f, sizeof(float), 1, pFile);
	f = bv_max_.z;
	std::fwrite((void*)&f, sizeof(float), 1, pFile);
	check_chars[6] = '4';
	std::fwrite((void*)check_chars, sizeof(char), 8, pFile);

	// 5. disparity information in order of: minimum disparity, maximum disparity, disparity step
	std::fwrite((void*)&min_disp, sizeof(float), 1, pFile);
	std::fwrite((void*)&max_disp, sizeof(float), 1, pFile);
	std::fwrite((void*)&disp_step, sizeof(float), 1, pFile);
	check_chars[6] = '5';
	std::fwrite((void*)check_chars, sizeof(char), 8, pFile);
	short max_disp_quant = round((max_disp - min_disp) / disp_step); // max_disp is at min_depth and disp_step is positive moving from min_disp to max_disp
	short min_disp_quant = 0; // max_disp is at min_depth and disp_step is positive moving from min_disp to max_disp
	if (debug) {
		cout << "min_disp" << endl << min_disp << endl;
		cout << "max_disp" << endl << max_disp << endl;
		cout << "disp_step" << endl << disp_step << endl;
		cout << "max_disp_quant" << endl << max_disp_quant << endl;
		cout << "min_disp_quant" << endl << min_disp_quant << endl;
		cout << endl;
	}

	// 6. 3x3 inverse K matrix of 9 floats, then 4x4 inverse RT matrix of 16 floats
	//fwrite((void*)calib_.Kinv_.data(), sizeof(float), 9, pFile);
	for (int r = 0; r < 3; r++) {
		for (int c = 0; c < 3; c++) {
			f = calib_.Kinv_(r, c);
			fwrite((void*)&f, sizeof(float), 1, pFile);
		}
	}
	//fwrite((void*)RTinv_.data(), sizeof(float), 16, pFile);
	for (int r = 0; r < 4; r++) {
		for (int c = 0; c < 4; c++) {
			f = RTinv_(r, c);
			fwrite((void*)&f, sizeof(float), 1, pFile);
		}
	}
	if (debug) {
		cout << "K_" << endl << calib_.K_ << endl << endl;
		cout << "Kinv_" << endl << calib_.Kinv_ << endl << endl;
		cout << "RT_" << endl << RT_ << endl << endl;
		cout << "RTinv_" << endl << RTinv_ << endl << endl;
		cout << "P_" << endl << P_ << endl << endl;
		cout << "Pinv_" << endl << Pinv_ << endl << endl;
	}
	check_chars[6] = '6';
	std::fwrite((void*)check_chars, sizeof(char), 8, pFile);

	// 7. int number of used pixels, int number of run length unsigned shorts, int number of unused rows at top
	// 8. run lengths; start with an unsigned short containing the number of contiguous unused pixels' depths in the current raster row (starting at the first non-blank row) to follow (0 if starts with used pixels), followed by the number of contiguous used pixels next, followed by the number of contiguous unused pixels' depths to follow, etc. repeated to the end of the image.  If a count exceeds USHRT_MAX, write a USHRT_MAX count, then a zero, then repeat for the remainder.
	unsigned int num_pixels_used = 0;
	unsigned int contig_pixel_count = 0;
	std::vector<unsigned short> rls; // run lengths
	unsigned int unused_rows_top = 0; // the number of unused rows at the top of the image
	unsigned int unused_rows_bottom = 0; // the number of unused rows at the bottom of the image
	bool passed_top = false; // boolean denoting whether we have iterated past the top rows of empty pixels into actual data
	bool last_rl_used; // boolean denoting whether the last run-length was a count of used pixels (true) or unused pixels (false)
	uchar* pM;
	last_rl_used = false;
	for (int r = 0; r < imgMask_.rows; r++) {
		pM = imgMask_.ptr<uchar>(r);
		if (!passed_top) {
			unused_rows_top++;
			contig_pixel_count = 0;
		}
		for (int c = 0; c < imgMask_.cols; c++) {
			contig_pixel_count++;
			if ((pM[c] == 0) ||
				(dm_->depth_map_(r, c) == 0.)) { // unused pixel
				if (last_rl_used) {
					RLE_WriteCount(contig_pixel_count, &rls);
					contig_pixel_count = 0; // reset whenever the type of pixel (used/unused) is changed on recording a new count
					last_rl_used = false;
				}
			}
			else { // used pixel
				num_pixels_used++;
				if (!last_rl_used) {
					RLE_WriteCount(contig_pixel_count, &rls);
					contig_pixel_count = 0; // reset whenever the type of pixel (used/unused) is changed on recording a new count
					last_rl_used = true;
				}
				unused_rows_bottom = (height_ - 1) - r;
				passed_top = true;
			}
		}
	}
	if (last_rl_used) {
		RLE_WriteCount(contig_pixel_count, &rls);
		contig_pixel_count = 0; // reset whenever the type of pixel (used/unused) is changed on recording a new count
		last_rl_used = false;
	}
	/*
	// remove blank bottom rows from packing_counts vector list
	int unused_counts = floor((float)unused_rows_bottom / (float)USHRT_MAX);
	if (unused_rows_bottom > unused_counts) unused_counts++;
	unused_counts += (unused_counts - 1); // 0's in-between max counts
	rls.erase(rls.begin() + (rls.size() - unused_counts + 1), rls.end()); // if I remove the +1, it incorrectly leaves off a run length, but does it work with it?
	*/

	// write data
	int num_rl_ushorts = rls.size();
	std::fwrite((void*)&num_pixels_used, sizeof(unsigned int), 1, pFile);
	std::fwrite((void*)&num_rl_ushorts, sizeof(unsigned int), 1, pFile);
	std::fwrite((void*)&unused_rows_top, sizeof(unsigned int), 1, pFile);
	check_chars[6] = '7';
	std::fwrite((void*)check_chars, sizeof(char), 8, pFile);
	if (debug) {
		cout << "num_pixels_used" << endl << num_pixels_used << endl << endl << "num_rl_ushorts" << endl << num_rl_ushorts << endl << endl;
		cout << "unused_rows_top" << endl << unused_rows_top << endl << endl;
	}
	int pc;
	for (std::vector<unsigned short>::iterator it = rls.begin(); it != rls.end(); ++it) {
		pc = (*it);
		std::fwrite((void*)&pc, sizeof(unsigned short), 1, pFile);
	}
	check_chars[6] = '8';
	std::fwrite((void*)check_chars, sizeof(char), 8, pFile);
	
	// 9. RLE pixel quantized disparity labels, each as a short; they are in raster-scan order, skipping any unused pixels
	float disp;
	unsigned short disp_label;
	unsigned short max_disp_label = round((max_disp - min_disp) / disp_step);
	unsigned short min_disp_label = 0;
	int num_disp_written = 0;
	for (int r = 0; r < imgMask_.rows; r++) {
		pM = imgMask_.ptr<uchar>(r);
		for (int c = 0; c < imgMask_.cols; c++) {
			if ((pM[c] == 0) ||
				(dm_->depth_map_(r, c) == 0.)) continue; // unused pixel
			disp = 1. / dm_->depth_map_(r, c);
			disp_label = round((disp - min_disp) / disp_step);
			if (disp_label > max_disp_label) disp_label = max_disp_label;
			else if (disp_label < min_disp_label) disp_label = min_disp_label;
			std::fwrite((void*)&disp_label, sizeof(unsigned short), 1, pFile); // write used pixel quantized disparity label
			num_disp_written++;
		}
	}
	if (debug) cout << "num_disp_written " << num_disp_written << endl;
	check_chars[6] = '9';
	std::fwrite((void*)check_chars, sizeof(char), 8, pFile);
	/*
	// 10. RLE image data, each pixel as 3 uchars, BGR ordered, with only used pixels included
	Vec3b *pT;
	Vec3b pix;
	for (int r = 0; r < imgMask_.rows; r++) {
		pT = imgT_.ptr<Vec3b>(r);
		pM = imgMask_.ptr<uchar>(r);
		for (int c = 0; c < imgMask_.cols; c++) {
			if ((pM[c] == 0) ||
				(dm_->depth_map_(r, c) == 0.)) continue; // unused pixel
			pix = pT[c];
			std::fwrite((void*)&pix[0], sizeof(unsigned char), 1, pFile);
			std::fwrite((void*)&pix[1], sizeof(unsigned char), 1, pFile);
			std::fwrite((void*)&pix[2], sizeof(unsigned char), 1, pFile);
		}
	}
	check_chars[6] = '10';
	std::fwrite((void*)check_chars, sizeof(char), 8, pFile);
	*/
	std::fclose(pFile);
	
	/*
	// saves a separate image file of used pixels only
	int ncols = ceil((float)num_pixels_used / (float)imgT_.rows);
	Mat imgUsed = Mat(imgT_.rows, ncols, CV_8UC3);
	Vec3b *pTall;
	Vec3b *pTused;
	int idx_pix_used = 0;
	pTused = imgUsed.ptr<Vec3b>(0);
	int curr_used_row = 0;
	Point p_used;
	for (int r = 0; r < imgMask_.rows; r++) {
		pTall = imgT_.ptr<Vec3b>(r);
		pM = imgMask_.ptr<uchar>(r);
		for (int c = 0; c < imgMask_.cols; c++) {
			if ((pM[c] == 0) ||
				(dm_->depth_map_(r, c) == 0.)) continue; // unused pixel
			p_used = PixIndexBwdRM(idx_pix_used, ncols);
			if (p_used.y > curr_used_row) {
				curr_used_row = p_used.y;
				pTused = imgUsed.ptr<Vec3b>(p_used.y);
			}
			pTused[p_used.x] = pTall[c];
			idx_pix_used++;
		}
	}
	fn = GLOBAL_FILEPATH_DATA + scene_name + "\\" + scene_name + "_camUsed" + to_string(id_) + ".jpg";
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY); // from 0 to 100 (the higher is the better). Default value is 95.
	compression_params.push_back(GLOBAL_JPG_WRITE_QUALITY);
	try {
		imwrite(fn, imgUsed, compression_params);
	}
	catch (runtime_error& ex) {
		fprintf(stderr, "Exception converting image to JPG format: %s\n", ex.what());
	}
	*/

	SaveMaskedImage(scene_name); // saves a separate image file after masking using imgMask_ and excluding any pixels for which depth values aren't available
	
	if (debug) {
		cout << "completed save for cam id " << id_ << endl;
		cin.ignore();
	}
}

// saves a separate image file after masking using imgMask_ and excluding any pixels for which depth values aren't available
void Camera::SaveMaskedImage(std::string scene_name) {
	Mat imgT_masked = cv::Mat::zeros(imgT_.rows, imgT_.cols, CV_8UC3);
	Vec3b* pT;
	Vec3b* pTm;
	uchar* pM;
	for (int r = 0; r < imgMask_.rows; r++) {
		pT = imgT_.ptr<Vec3b>(r);
		pTm = imgT_masked.ptr<Vec3b>(r);
		pM = imgMask_.ptr<uchar>(r);
		for (int c = 0; c < imgMask_.cols; c++) {
			if ((pM[c] == 0) ||
				(dm_->depth_map_(r, c) == 0.)) continue; // unused pixel
			pTm[c] = pT[c];
		}
	}
	string fn = GLOBAL_FILEPATH_DATA + scene_name + "\\" + scene_name + "_cam" + to_string(id_) + ".jpg";
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY); // from 0 to 100 (the higher is the better). Default value is 95.
	compression_params.push_back(GLOBAL_JPG_WRITE_QUALITY);
	try {
		imwrite(fn, imgT_masked, compression_params);
	}
	catch (runtime_error& ex) {
		fprintf(stderr, "Exception converting image to JPG format: %s\n", ex.what());
	}

	/*
	// code for saving an image at higher JPEG quality
	Mat imgUsed = Mat(1, 10000, CV_8UC3);
	imgUsed.setTo(100);
	std::string fn = GLOBAL_FILEPATH_DATA + "asdfasdf_camUsedtest.jpg";
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY); // from 0 to 100 (the higher is the better). Default value is 95.
	compression_params.push_back(GLOBAL_JPG_WRITE_QUALITY);
	try {
	imwrite(fn, imgUsed, compression_params);
	}
	catch (runtime_error& ex) {
	fprintf(stderr, "Exception converting image to JPG format: %s\n", ex.what());
	}
	return 0;
	*/
}

// updates values for min_disp, max_disp, disp_step
// note: all matrices are indexed in row-major pixel order (across a row, then down to the next one), except Kinv and RTinv which are stored column-major pixel order in accordance with the Eigen default
// note: to qualify as a used pixel, it must be both not masked out (mask val != 0) and have a valid depth (depth val != 0.) - also not that this means we can't reconstruct imgMask from the loaded data
// filename format of 2 files written:
//		1. GLOBAL_FILEPATH_DATA + scene_name + "\\" + scene_name + "_cam" + to_string(id_) + ".adf"
//		2. GLOBAL_FILEPATH_DATA + scene_name + "\\" + scene_name + "_cam" + to_string(id_) + ".jpg"
// loads compressed data needed for rendering as a binary file with the following data, in order:
// 1. 50 character (byte) asset name that is binary-zero-terminated (char \0); followed by binary-zero-terminated (char \0) 8-character (byte) "Check01"
// 2. int camera ID; followed by binary-zero-terminated (char \0) 8-character (byte) "Check02"
// 3. unsigned int for # rows, unsigned int for # cols; followed by binary-zero-terminated (char \0) 8-character (byte) "Check03"
// 4. bounding volume as floats of in order of: xmin, ymin, zmin, xmax, ymax, zmax; followed by binary-zero-terminated (char \0) 8-character (byte) "Check04"
// 5. disparity information in order of: minimum disparity, maximum disparity, disparity step; followed by binary-zero-terminated (char \0) 8-character (byte) "Check05"
// 6. 3x3 inverse K matrix of 9 floats, then 4x4 inverse RT matrix of 16 floats; followed by binary-zero-terminated (char \0) 8-character (byte) "Check06"
// 7. unsigned int number of used pixels, unsigned int number of run length unsigned shorts, unsigned int number of unused rows at top; followed by binary-zero-terminated (char \0) 8-character (byte) "Check07"
// 8. run lengths; start with an unsigned short containing the number of contiguous unused pixels' depths in the current raster row (starting at the first non-blank row) to follow (0 if starts with used pixels), followed by the number of contiguous used pixels next, followed by the number of contiguous unused pixels' depths to follow, etc. repeated to the end of the image.  If a count exceeds USHRT_MAX, write a USHRT_MAX count, then a zero, then repeat for the remainder. ; followed by binary-zero-terminated (char \0) 8-character (byte) "Check08"
// 9. RLE pixel quantized disparity labels, each as an unsigned short; they are in raster-scan order, skipping any unused pixels; followed by binary-zero-terminated (char \0) 8-character (byte) "Check09"
// loads a separate image file after masking using imgMask_ and excluding any pixels for which depth values aren't available
void Camera::LoadRenderingDataRLE(std::string scene_name, float &min_disp, float &max_disp, float &disp_step) {
	bool debug = false;

	std::string fn = GLOBAL_FILEPATH_DATA + scene_name + "\\" + scene_name + "_cam" + to_string(id_) + ".adf";
	FILE* pFile = fopen(fn.c_str(), "rb"); // read binary mode
	float f;
	char check_chars[8];

	if (pFile == NULL) {
		cerr << "Camera::LoadRenderingData() file not found" << endl;
		return;
	}

	// 1. 50 character (byte) asset name that is binary-zero-terminated (char \0)
	char asset_chars[50];
	std::fread(asset_chars, sizeof(char), 50, pFile);
	char* pch = strchr(asset_chars, '\0');
	if (pch == NULL) pch = asset_chars + 49;
	char *asset_chars_name = new char[pch - asset_chars + 1];
	memcpy(asset_chars_name, &asset_chars[0], pch - asset_chars + 1);
	string s = convert_chars_to_string(asset_chars_name);
	delete[] asset_chars_name;
	if (debug) cout << "Asset name found was " << s << endl;
	std::fread(check_chars, sizeof(char), 8, pFile);
	if (debug) cout << "Check01 confirmation: " << check_chars << endl;

	// 2. int camera ID
	std::fread((void*)&id_, sizeof(int), 1, pFile);
	std::fread(check_chars, sizeof(char), 8, pFile);
	if (debug) cout << "Check02 confirmation: " << check_chars << endl;

	// 3. int for # rows, int for # cols
	unsigned int height_ui, width_ui;
	std::fread((void*)&height_ui, sizeof(unsigned int), 1, pFile);
	std::fread((void*)&width_ui, sizeof(unsigned int), 1, pFile);
	height_ = height_ui;
	width_ = width_ui;
	if (debug) cout << height_ << " rows and " << width_ << " cols" << endl;
	std::fread(check_chars, sizeof(char), 8, pFile);
	if (debug) cout << "Check03 confirmation: " << check_chars << endl;

	// 4. bounding volume as floats of in order of: xmin, ymin, zmin, xmax, ymax, zmax
	Point3f bv_min, bv_max;
	std::fread((void*)&f, sizeof(float), 1, pFile);
	bv_min.x = f;
	std::fread((void*)&f, sizeof(float), 1, pFile);
	bv_min.y = f;
	std::fread((void*)&f, sizeof(float), 1, pFile);
	bv_min.z = f;
	std::fread((void*)&f, sizeof(float), 1, pFile);
	bv_max.x = f;
	std::fread((void*)&f, sizeof(float), 1, pFile);
	bv_max.y = f;
	std::fread((void*)&f, sizeof(float), 1, pFile);
	bv_max.z = f;
	if (debug) cout << "bv_min" << endl << bv_min << endl << endl << "bv_max" << endl << bv_max << endl << endl;
	std::fread(check_chars, sizeof(char), 8, pFile);
	if (debug) cout << "Check04 confirmation: " << check_chars << endl;

	// 5. disparity information in order of: minimum disparity, maximum disparity, disparity step
	std::fread((void*)&min_disp, sizeof(float), 1, pFile);
	std::fread((void*)&max_disp, sizeof(float), 1, pFile);
	std::fread((void*)&disp_step, sizeof(float), 1, pFile);
	short max_disp_quant = round((max_disp - min_disp) / disp_step); // max_disp is at min_depth and disp_step is positive moving from min_disp to max_disp
	short min_disp_quant = 0; // max_disp is at min_depth and disp_step is positive moving from min_disp to max_disp
	if (debug) {
		cout << "min_disp" << endl << min_disp << endl;
		cout << "max_disp" << endl << max_disp << endl;
		cout << "disp_step" << endl << disp_step << endl;
		cout << "max_disp_quant" << endl << max_disp_quant << endl;
		cout << "min_disp_quant" << endl << min_disp_quant << endl;
		cout << endl;
	}
	std::fread(check_chars, sizeof(char), 8, pFile);
	if (debug) cout << "Check05 confirmation: " << check_chars << endl;

	// 6. 3x3 inverse K matrix of 9 floats, then 4x4 inverse RT matrix of 16 floats
	Matrix3d Kinv;
	for (int r = 0; r < 3; r++) {
		for (int c = 0; c < 3; c++) {
			std::fread((void*)&f, sizeof(float), 1, pFile);
			Kinv(r, c) = f;
		}
	}
	calib_.UpdateFromKinv(Kinv);
	for (int r = 0; r < 4; r++) {
		for (int c = 0; c < 4; c++) {
			std::fread((void*)&f, sizeof(float), 1, pFile);
			RTinv_(r, c) = f;
		}
	}
	RT_ = RTinv_.inverse();
	UpdateCameraMatrices();
	if (debug) {
		cout << "K_" << endl << calib_.K_ << endl << endl;
		cout << "Kinv_" << endl << calib_.Kinv_ << endl << endl;
		cout << "RT_" << endl << RT_ << endl << endl;
		cout << "RTinv_" << endl << RTinv_ << endl << endl;
		cout << "P_" << endl << P_ << endl << endl;
		cout << "Pinv_" << endl << Pinv_ << endl << endl;
	}
	std::fread(check_chars, sizeof(char), 8, pFile);
	if (debug) cout << "Check06 confirmation: " << check_chars << endl;

	// 7. int number of used pixels, int number of run length unsigned shorts, int number of unused rows at top
	unsigned int num_pixels_used, num_rl_ushorts, unused_rows_top;
	std::fread((void*)&num_pixels_used, sizeof(unsigned int), 1, pFile);
	std::fread((void*)&num_rl_ushorts, sizeof(unsigned int), 1, pFile);
	std::fread((void*)&unused_rows_top, sizeof(unsigned int), 1, pFile);
	if (debug) {
		cout << "num_pixels_used" << endl << num_pixels_used << endl << endl << "num_rl_ushorts" << endl << num_rl_ushorts << endl << endl;
		cout << "unused_rows_top" << endl << unused_rows_top << endl << endl;
	}
	std::fread(check_chars, sizeof(char), 8, pFile);
	if (debug) cout << "Check07 confirmation: " << check_chars << endl;

	// 8. run lengths; start with an unsigned short containing the number of contiguous unused pixels' depths in the current raster row (starting at the first non-blank row) to follow (0 if starts with used pixels), followed by the number of contiguous used pixels next, followed by the number of contiguous unused pixels' depths to follow, etc. repeated to the end of the image.  If a count exceeds USHRT_MAX, write a USHRT_MAX count, then a zero, then repeat for the remainder.
	unsigned short* rls = new unsigned short[num_rl_ushorts]; // run lengths
	std::fread((void*)rls, sizeof(unsigned short), num_rl_ushorts, pFile);
	std::fread(check_chars, sizeof(char), 8, pFile);
	if (debug) cout << "Check08 confirmation: " << check_chars << endl;

	// 9. sparse pixel quantized disparities, each as a short; they are in raster-scan order, skipping any unused pixels
	unsigned short* disp_labels = new unsigned short[num_pixels_used];
	std::fread((void*)disp_labels, sizeof(unsigned short), num_pixels_used, pFile);
	std::fread(check_chars, sizeof(char), 8, pFile);
	if (debug) cout << "Check09 confirmation: " << check_chars << endl;

	dm_->depth_map_.setZero(); // clear before loading so unused pixels are blank
	int idx_used = 0; // index into used pixels for disparity labels (disp_quant)
	int idx_all = unused_rows_top * width_; // index into all pixels for depth_map; skip blank rows
	unsigned short disp_label;
	bool used_count = false;
	unsigned short rl;
	Point p;
	for (int i = 0; i < (int)num_rl_ushorts; i++) {
		rl = rls[i];

		if (used_count) {
			for (int j = 0; j < rl; j++) {
				p = PixIndexBwdRM(idx_all, width_);
				disp_label = disp_labels[idx_used];
				dm_->depth_map_(p.y, p.x) = 1. / (((float)disp_label * disp_step) + min_disp);
				idx_used++;
				idx_all++;
			}
		}
		else idx_all += rl;

		used_count = !used_count;
	}
	delete[] disp_labels;
	delete[] rls;
	/*
	// 10. RLE image data, each pixel as 3 uchars, BGR ordered, with only used pixels included
	Vec3b* pixels = new Vec3b[num_pixels_used];
	std::fread((void*)disp_labels, sizeof(unsigned short), num_pixels_used, pFile);
	std::fread(check_chars, sizeof(char), 8, pFile);
	if (debug) cout << "Check10 confirmation: " << check_chars << endl;

	imgT_ = Mat::zeros(imgT_.rows, imgT_.cols, CV_8UC3); // clear before loading so unused pixels are blank
	idx_used = 0; // index into used pixels for disparity labels (disp_quant)
	idx_all = unused_rows_top * width_; // index into all pixels for depth_map; skip blank rows
	used_count = false;
	for (int i = 0; i < num_rl_ushorts; i++) {
		rl = rls[i];

		if (used_count) {
			for (int j = 0; j < rl; j++) {
				p = PixIndexBwdRM(idx_all, width_);
				imgT_.at<Vec3b>(p.y, p.x) = pixels[idx_used];
				idx_used++;
				idx_all++;
			}
		}
		else idx_all += rl;

		used_count = !used_count;
	}
	delete[] pixels;
	delete[] rls;
	*/
	
	std::fclose(pFile);
	
	// loads a separate image file after masking using imgMask_ and excluding any pixels for which depth values aren't available
	fn = GLOBAL_FILEPATH_DATA + scene_name + "\\" + scene_name + "_cam" + to_string(id_) + ".jpg";
	imgT_ = imread(fn, CV_LOAD_IMAGE_COLOR);


	if (debug) {
		cout << "completed load of file 1 for cam id " << id_ << endl;
		dm_->DisplayDepthImage();
		display_mat(&imgT_, "loaded imgT", orientation_);
	}
	
}

void Camera::Save_K(std::string scene_name) {
	Mat mat;
	EigenOpenCV::eigen2cv(calib_.K_, mat);
	CvMat cvmat = mat;
	string filepath = GLOBAL_FILEPATH_DATA + scene_name + "\\";
	char* fn_chars = GetFilenameMatrix(filepath, scene_name, "K");
	cvSave(fn_chars, &cvmat);
	delete[] fn_chars;
}

void Camera::Save_RT(std::string scene_name) {
	Mat mat;
	EigenOpenCV::eigen2cv(RT_, mat);
	CvMat cvmat = mat;
	string filepath = GLOBAL_FILEPATH_DATA + scene_name + "\\";
	char* fn_chars = GetFilenameMatrix(filepath, scene_name, "RT");
	cvSave(fn_chars, &cvmat);
	delete[] fn_chars;
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
	display_mat(&imgT_, "Camera image", orientation_);
	if (has_depth_map_) dm_->DisplayDepthImage();
	cout << endl;
}

void Camera::SavePointCloud(string scene_name) {

	std::string fn = GLOBAL_FILEPATH_DATA + scene_name + "\\" + scene_name + "_pointcloud.obj";
	ofstream myfile;
	myfile.open(fn);

	//double *p = Iws_used.data();
	for (int c = 0; c < Iws_.cols(); c++) {
		myfile << "v " << Iws_(0, c) << " " << Iws_(1, c) << " " << Iws_(2, c) << endl;
		//p++; // skip homogeneous coordinate
	}

	myfile.close();
}

void Camera::BuildMesh() {
	bool debug = false;

	mesh_vertices_.erase(mesh_vertices_.begin(), mesh_vertices_.end());
	mesh_faces_.erase(mesh_faces_.begin(), mesh_faces_.end());
	mesh_normals_.erase(mesh_normals_.begin(), mesh_normals_.end());

	InitWorldSpaceProjection(); // ensure Iws_ is up to date
	Matrix<double, Dynamic, 4> I = Iws_.transpose().cast<double>();
	Matrix<bool, Dynamic, Dynamic, RowMajor> mask(height_, width_);
	EigenOpenCV::cv2eigen(imgMask_, mask);
	mask.resize(height_*width_, 1);

	int num_points = height_*width_;
	int h = height_;
	int w = width_;
	bool *pULm = mask.data();
	bool *pLLm = mask.data() + w;
	bool *pURm = mask.data() + 1;
	bool *pLRm = mask.data() + w + 1;
	double *pULx = I.col(0).data();
	double *pLLx = I.col(0).data() + w;
	double *pURx = I.col(0).data() + 1;
	double *pLRx = I.col(0).data() + w + 1;
	double *pULy = I.col(1).data();
	double *pLLy = I.col(1).data() + w;
	double *pURy = I.col(1).data() + 1;
	double *pLRy = I.col(1).data() + w + 1;
	double *pULz = I.col(2).data();
	double *pLLz = I.col(2).data() + w;
	double *pURz = I.col(2).data() + 1;
	double *pLRz = I.col(2).data() + w + 1;
	// pre-fill first column of vertices
	for (int i = 0; i < h; i++) {
		// create vertices
		if (*pULm) { // if masked-in
			Point3d vUL(*pULx, *pULy, *pULz);
			mesh_vertices_[i] = vUL;
		}
		if (*pLLm) { // if masked-in
			Point3d vLL(*pLLx, *pLLy, *pLLz);
			mesh_vertices_[i + w] = vLL;
		}

		pULm++;
		pLLm++;
		pULx++;
		pLLx++;
		pULy++;
		pLLy++;
		pULz++;
		pLLz++;
	}
	// reset pointers used
	pULm = mask.data();
	pLLm = mask.data() + w;// 1;
	pULx = I.col(0).data();
	pLLx = I.col(0).data() + w;// 1;
	pULy = I.col(1).data();
	pLLy = I.col(1).data() + w;// 1;
	pULz = I.col(2).data();
	pLLz = I.col(2).data() + w;// 1;
	int idx_face = 0;
	for (int i = 0; i < (num_points - w - 1); i++) { // we are looking ahead in the data one column and row, so must stop one column and row short
		// create vertices on right
		if (*pURm) { // if masked-in
			Point3d vUR(*pURx, *pURy, *pURz);
			mesh_vertices_[i + 1] = vUR;
		}
		if (*pLRm) { // if masked-in
			Point3d vLR(*pLRx, *pLRy, *pLRz);
			mesh_vertices_[i + w + 1] = vLR;
		}

		// create faces and normals
		if ((*pULm) &&
			(*pLRm) &&
			(*pLLm)) {
			Point3d e1 = mesh_vertices_[i + w + 1] - mesh_vertices_[i];
			Point3d e2 = mesh_vertices_[i + 1] - mesh_vertices_[i + w + 1];
			Point3d e3 = mesh_vertices_[i] - mesh_vertices_[i + 1];
			double d1 = vecdist(e1, e2);
			double d2 = vecdist(e1, e3);
			double d3 = vecdist(e2, e3);
			if ((d1 < GLOBAL_MESH_EDGE_DISTANCE_MAX) &&
				(d2 < GLOBAL_MESH_EDGE_DISTANCE_MAX) &&
				(d3 < GLOBAL_MESH_EDGE_DISTANCE_MAX)) {
				Vec3i f(i, i + w + 1, i + 1); // counter-clockwise vertex order
				mesh_faces_[idx_face] = f;
				Point3d n = e1.cross(e2);
				mesh_normals_[idx_face] = n;
				idx_face++;
			}
		}
		if ((*pULm) &&
			(*pLRm) &&
			(*pURm)) {
			Point3d e1 = mesh_vertices_[i + w] - mesh_vertices_[i];
			Point3d e2 = mesh_vertices_[i + w + 1] - mesh_vertices_[i + w];
			Point3d e3 = mesh_vertices_[i] - mesh_vertices_[i + w + 1];
			double d1 = vecdist(e1, e2);
			double d2 = vecdist(e1, e3);
			double d3 = vecdist(e2, e3);
			if ((d1 < GLOBAL_MESH_EDGE_DISTANCE_MAX) &&
				(d2 < GLOBAL_MESH_EDGE_DISTANCE_MAX) &&
				(d3 < GLOBAL_MESH_EDGE_DISTANCE_MAX)) {
				Vec3i f(i, i + w, i + w + 1); // counter-clockwise vertex order
				mesh_faces_[idx_face] = f;
				Point3d n = e1.cross(e2);
				mesh_normals_[idx_face] = n;
				idx_face++;
			}
		}

		pULm++;
		pLLm++;
		pURm++;
		pLRm++;
		pULx++;
		pLLx++;
		pURx++;
		pLRx++;
		pULy++;
		pLLy++;
		pURy++;
		pLRy++;
		pULz++;
		pLLz++;
		pURz++;
		pLRz++;
	}

	if (debug) {
		cout << "Number of faces " << mesh_faces_.size() << endl;
		Mat img = Mat::zeros(height_, width_, CV_8UC3);
		imgT_.copyTo(img);
		for (map<int, Point3d>::iterator it = mesh_vertices_.begin(); it != mesh_vertices_.end(); ++it) {
			int idx = (*it).first;
			Point p = PixIndexBwdRM(idx, width_);
			img.at<Vec3b>(p.y, p.x) = Vec3b(0, 0, 255);
		}
		for (map<int, Vec3i>::iterator it = mesh_faces_.begin(); it != mesh_faces_.end(); ++it) {
			int fid = (*it).first;
			int idx0 = (*it).second[0];
			Point p0 = PixIndexBwdRM(idx0, width_);
			img.at<Vec3b>(p0.y, p0.x) = Vec3b(255, 0, 0);
			int idx1 = (*it).second[1];
			Point p1 = PixIndexBwdRM(idx1, width_);
			img.at<Vec3b>(p1.y, p1.x) = Vec3b(255, 0, 0);
			int idx2 = (*it).second[2];
			Point p2 = PixIndexBwdRM(idx2, width_);
			img.at<Vec3b>(p2.y, p2.x) = Vec3b(255, 0, 0);
		}
		display_mat(&img, "Img with mesh vertices highlighted", orientation_);
	}
}

// reprojects the camera view into a new camera with projection matrix P_dest
// only reprojects pixels for which there is depth info
// imgT is modified to include texture, imgD to include depth values in the virtual view's camera space, and imgMask to include binary mask values (255 pixel is opaque and 0 pixel is transparent), and all must be same size and types CV_8UC3 and CV_32F, respectively.
// can't really project the mask because there is no depth information for background pixels, so no way to project them and tell the difference between them and holes that appear in the projection due to revealed occlusions; it would be nice to reverse project destination pixels to source to determine whether they're masked-in or masked-out, but we have no destination depth information, which would be necessary for the reverse projection; but can create mask that has both kinds of empty pixels masked-out so we don't confuse them with black pixels that actually hold reprojected color
void Camera::ReprojectMesh(Point3d view_dir, Matrix<double, 3, 4> *P_dest, Matrix4d *RT_dest, Mat *imgT, Matrix<float, Dynamic, Dynamic> *depth_map, Mat *imgMask) {
	bool debug = false;
	bool timing = true; double t;
	if (timing) t = (double)getTickCount();
	
	assert(imgT_.rows == dm_->depth_map_.rows() && imgT_.cols == dm_->depth_map_.cols());
	assert(imgMask_.rows == imgT_.rows && imgMask_.cols == imgT_.cols);
	assert(imgT->rows == depth_map->rows() && imgT->cols == depth_map->cols());
	assert(imgMask->rows == depth_map->rows() && imgMask->cols == depth_map->cols());
	assert(imgT->type() == CV_8UC3);
	assert(imgMask->type() == CV_8UC1);

	if (debug) cout << "Reprojecting mesh from camera " << id_ << endl;

	// clear result matrices
	imgT->setTo(Vec3b(0, 0, 0));
	depth_map->setZero();
	imgMask->setTo(0);

	// reproject world space coordinates to virtual camera's screen space
	Matrix<float, 3, Dynamic> I_dest = (*P_dest).cast<float>() * Iws_; // note the matrix multiplication property: Ainv * A = A * Ainv
	Matrix<float, 1, Dynamic> H = I_dest.row(2).array().inverse();
	I_dest.row(0) = I_dest.row(0).cwiseProduct(H);
	I_dest.row(1) = I_dest.row(1).cwiseProduct(H);

	// reproject world space coordinates to virtual camera's camera space so can record virtual CS depths
	Matrix<float, 4, Dynamic> Ics_dest = (*RT_dest).cast<float>() * Iws_; // note the matrix multiplication property: Ainv * A = A * Ainv
	Ics_dest.row(2) = Ics_dest.row(2).cwiseQuotient(Ics_dest.row(3));
	
	// where there is a depth value, no depth discontinuity, it is not masked out, and it falls within screen space bounds for the virtual view, copy the texture and depth info over to imgT and imgD, respectively; project mask values as well, despite masking out imgT and imgD during projection, because errors in depth values will result in different images having different ideas of what is masked out and will need to use projected masks to make the final determination during NVS blending
	Point2d v1_proj, v2_proj, v3_proj;
	for (map<int, Vec3i>::iterator itf = mesh_faces_.begin(); itf != mesh_faces_.end(); ++itf) {// for each face
		int fid = (*itf).first;

		Point3d normal = mesh_normals_[fid];
		
		if (normal.ddot(view_dir) >= 0) continue; // cull back-facing faces
		int vid1 = (*itf).second[0];
		int vid2 = (*itf).second[1];
		int vid3 = (*itf).second[2];
		v1_proj.x = static_cast<double>(I_dest(0, vid1));
		v1_proj.y = static_cast<double>(I_dest(1, vid1));
		v2_proj.x = static_cast<double>(I_dest(0, vid2));
		v2_proj.y = static_cast<double>(I_dest(1, vid2));
		v3_proj.x = static_cast<double>(I_dest(0, vid3));
		v3_proj.y = static_cast<double>(I_dest(1, vid3));
		
		//if (debug) {// debugging
		//	Point p1 = PixIndexBwdRM(vid1, width_);
		//	Point p2 = PixIndexBwdRM(vid2, width_);
		//	Point p3 = PixIndexBwdRM(vid3, width_);
		//	if (imgMask_.at<uchar>(p1.y, p1.x) &&
		//		imgMask_.at<uchar>(p2.y, p2.x) &&
		//		imgMask_.at<uchar>(p3.y, p3.x)) {
		//		cin.ignore();
		//		cout << "v1_proj " << v1_proj << endl;
		//		cout << "v2_proj " << v2_proj << endl;
		//		cout << "v3_proj " << v3_proj << endl;
		//	}
		//}

		// limit pixels of interest to bounding box of face
		double xmind, ymind, xmaxd, ymaxd;
		xmind = min(v1_proj.x, v2_proj.x);
		xmind = min(xmind, v3_proj.x);
		xmaxd = max(v1_proj.x, v2_proj.x);
		xmaxd = max(xmaxd, v3_proj.x);
		ymind = min(v1_proj.y, v2_proj.y);
		ymind = min(ymind, v3_proj.y);
		ymaxd = max(v1_proj.y, v2_proj.y);
		ymaxd = max(ymaxd, v3_proj.y);
		int xmin = floor(xmind);
		int ymin = floor(ymind);
		int xmax = ceil(xmaxd);
		int ymax = ceil(ymaxd);

		//cout << "x " << xmin << " to " << xmax << ", y " << ymin << " to " << ymax << endl;
		
		for (int x = xmin; x <= xmax; x++) {
			if ((x < 0) ||
				(x >= width_))
				continue;
			for (int y = ymin; y <= ymax; y++) {
				if ((y < 0) ||
					(y >= height_))
					continue;

				//cout << "valid point (" << x << ", " << y << ")" << endl;
				
				Point2d p(static_cast<double>(x), static_cast<double>(y));
				Point3d bary;
				bool inside = Math3d::SetBarycentricTriangle(p, v1_proj, v2_proj, v3_proj, bary); // mesh was created with faces of vertices in counter-clockwise order; when reprojected with normal opposite of view dir, should retain counter-clockwise order in the new screen space
				
				// determine whether pixel of interest is inside or outside face; if inside, update color, depth (from original CS), and mask of projection
				if (inside) {
					// interpolate vertex colors
					Point p1 = PixIndexBwdRM(vid1, width_);
					Point p2 = PixIndexBwdRM(vid2, width_);
					Point p3 = PixIndexBwdRM(vid3, width_);
					// interpolate vertex depths
					float depth_new = static_cast<float>(bary.x) * Ics_dest(2, vid1) + static_cast<float>(bary.y) * Ics_dest(2, vid2) + static_cast<float>(bary.z) * Ics_dest(2, vid3);
					float depth_curr = (*depth_map)(p.y, p.x);
					//if ((depth_new - depth_curr) <= GLOBAL_MESH_EDGE_DISTANCE_MAX) // only replace the depth if there isn't an occlusion, so depth_new must not be much farther from the camera than depth_curr
					if ((depth_new > 0) &&
						((depth_curr == 0) ||
						(depth_new < depth_curr))) {
						(*depth_map)(p.y, p.x) = depth_new; // copy over depth data for imgD
						Vec3b color1 = imgT_.at<Vec3b>(p1.y, p1.x);
						Vec3b color2 = imgT_.at<Vec3b>(p2.y, p2.x);
						Vec3b color3 = imgT_.at<Vec3b>(p3.y, p3.x);
						double b1 = static_cast<double>(color1[0]);
						double g1 = static_cast<double>(color1[1]);
						double r1 = static_cast<double>(color1[2]);
						double b2 = static_cast<double>(color2[0]);
						double g2 = static_cast<double>(color2[1]);
						double r2 = static_cast<double>(color2[2]);
						double b3 = static_cast<double>(color3[0]);
						double g3 = static_cast<double>(color3[1]);
						double r3 = static_cast<double>(color3[2]);
						Vec3b color_new;
						color_new[0] = static_cast<uchar>(b1*bary.x + b2*bary.y + b3*bary.z);
						color_new[1] = static_cast<uchar>(g1*bary.x + g2*bary.y + g3*bary.z);
						color_new[2] = static_cast<uchar>(r1*bary.x + r2*bary.y + r3*bary.z);
						imgT->at<Vec3b>(p.y, p.x) = color_new; // copy over color data for imgT
					}
					imgMask->at<uchar>(p.y, p.x) = 255;
				}
				
			}
		}
	}
	
	if (debug) {
		display_mat(imgT, "imgT reproject", orientation_);
		dm_->DisplayDepthImage(depth_map);
		display_mat(imgMask, "imgMask reproject", orientation_);
	}

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Camera::ReprojectMesh() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}

// like ReprojectMesh() but only computes depths
// updates depth_map where appropriate, but leaves other existing values in the map as is
// projects this camera's mesh into the screen space defined by P_dest to find barycentric depths in camera space using RT_dest
// sets pixels in change_map to true where changes were made, but does not set rest to false (must do that beforehand)
// only consider a camera an "expert" on a pixel depth if the angle between the normal of the face and the originating camera angle is within a certain threshold
void Camera::ReprojectMeshDepths(Point3d view_dir, Matrix<double, 3, 4> *P_dest, Matrix4d *RT_dest, Matrix<float, Dynamic, Dynamic> *depth_map, Matrix<bool, Dynamic, 1> *change_map) {
	assert(change_map->rows() == depth_map->rows()*depth_map->cols());

	bool debug = false;
	bool timing = true; double t;
	if (timing) t = (double)getTickCount();

	if (debug) cout << "Reprojecting mesh from camera " << id_ << endl;

	// reproject world space coordinates to virtual camera's screen space
	Matrix<float, 3, Dynamic> I_dest = (*P_dest).cast<float>() * Iws_; // note the matrix multiplication property: Ainv * A = A * Ainv
	Matrix<float, 1, Dynamic> H = I_dest.row(2).array().inverse();
	I_dest.row(0) = I_dest.row(0).cwiseProduct(H);
	I_dest.row(1) = I_dest.row(1).cwiseProduct(H);

	// reproject world space coordinates to virtual camera's camera space so can record virtual CS depths
	Matrix<float, 4, Dynamic> Ics_dest = (*RT_dest).cast<float>() * Iws_; // note the matrix multiplication property: Ainv * A = A * Ainv
	Ics_dest.row(2) = Ics_dest.row(2).cwiseQuotient(Ics_dest.row(3));
	
	// where there is a depth value, no depth discontinuity, it is not masked out, and it falls within screen space bounds for the virtual view, copy the depth info over to imgT and imgD, respectively; project mask values as well, despite masking out imgT and imgD during projection, because errors in depth values will result in different images having different ideas of what is masked out and will need to use projected masks to make the final determination during NVS blending
	Point2d p, v1_proj, v2_proj, v3_proj;
	Point3d bary, normal;
	double xmind, ymind, xmaxd, ymaxd;
	int ymin, ymax, xmin, xmax;
	bool inside;
	float depth_new, depth_curr;
	for (map<int, Vec3i>::iterator itf = mesh_faces_.begin(); itf != mesh_faces_.end(); ++itf) {// for each face
		int fid = (*itf).first;

		normal = mesh_normals_[fid];
		if (normal.ddot(view_dir) >= 0)
			continue; // back-face culling
		//if (AngleBetweenVectors(normal, view_dir_) > GLOBAL_FACE_EXPERT_ANGLE_THRESHOLD)
		//	continue; // only consider a camera an "expert" on a pixel depth if the angle between the normal of the face and the originating camera angle is within a certain threshold

		int vid1 = (*itf).second[0];
		int vid2 = (*itf).second[1];
		int vid3 = (*itf).second[2];
		v1_proj.x = static_cast<double>(I_dest(0, vid1));
		v1_proj.y = static_cast<double>(I_dest(1, vid1));
		v2_proj.x = static_cast<double>(I_dest(0, vid2));
		v2_proj.y = static_cast<double>(I_dest(1, vid2));
		v3_proj.x = static_cast<double>(I_dest(0, vid3));
		v3_proj.y = static_cast<double>(I_dest(1, vid3));

		// limit pixels of interest to bounding box of face
		xmind = min(v1_proj.x, v2_proj.x);
		xmind = min(xmind, v3_proj.x);
		xmaxd = max(v1_proj.x, v2_proj.x);
		xmaxd = max(xmaxd, v3_proj.x);
		ymind = min(v1_proj.y, v2_proj.y);
		ymind = min(ymind, v3_proj.y);
		ymaxd = max(v1_proj.y, v2_proj.y);
		ymaxd = max(ymaxd, v3_proj.y);
		xmin = floor(xmind);
		ymin = floor(ymind);
		xmax = ceil(xmaxd);
		ymax = ceil(ymaxd);

		for (int x = xmin; x <= xmax; x++) {
			if ((x < 0) ||
				(x >= width_))
				continue;
			for (int y = ymin; y <= ymax; y++) {
				if ((y < 0) ||
					(y >= height_))
					continue;

				p.x = static_cast<double>(x);
				p.y = static_cast<double>(y);
				inside = Math3d::SetBarycentricTriangle(p, v1_proj, v2_proj, v3_proj, bary); // mesh was created with faces of vertices in counter-clockwise order; when reprojected with normal opposite of view dir, should retain counter-clockwise order in the new screen space

				// determine whether pixel of interest is inside or outside face; if inside, depth (from original CS)
				if (inside) {
					depth_new = static_cast<float>(bary.x) * Ics_dest(2, vid1) + static_cast<float>(bary.y) * Ics_dest(2, vid2) + static_cast<float>(bary.z) * Ics_dest(2, vid3); // interpolate vertex depths
					depth_curr = (*depth_map)(p.y, p.x);
					if (((depth_new - depth_curr) <= GLOBAL_DEPTH_EXPECTED_COMPUTATION_DIST_ERROR) &&
						(depth_new > 0)) { // only replace the depth if there isn't an occlusion, so depth_new must not be much farther from the camera than depth_curr
						(*depth_map)(p.y, p.x) = depth_new;
						int idx = PixIndexFwdCM(p, depth_map->rows());
						(*change_map)(idx, 0) = true;
					}
				}
			}
		}
	}
	
	if (debug)
		dm_->DisplayDepthImage(depth_map);

	if (timing) {
		t = (double)getTickCount() - t;
		cout << "Camera::ReprojectMesh() execution time = " << t*1000. / getTickFrequency() << " ms" << endl;
	}
}