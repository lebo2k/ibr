#include "DepthMap.h"

DepthMap::DepthMap() {
	fn_ = "";
	cam_id_ = -1;
	min_nonzero_depth_ = 0.;
	max_depth_ = 0.;
	width_ = 0;
	height_ = 0;

	orientation_ = AGO_ORIGINAL;
}

DepthMap::~DepthMap() {
}

// imgMask is a binary mask to use when computing minimum and maximum depth values - only those values where the mask is >0 are considered
void DepthMap::UpdateMinMaxDepths(Mat* imgMask) {
	bool debug = false;

	float val;
	max_depth_ = 0;
	min_nonzero_depth_ = 0;
	uchar* pM;
	for (int r = 0; r < height_; r++) {
		pM = imgMask->ptr<uchar>(r);
		for (int c = 0; c < width_; c++) {
			val = depth_map_(r, c);
			if ((val > 0) &&
				(pM[c] > 0)) {
				if (val > max_depth_)
					max_depth_ = val;
				if ((val < min_nonzero_depth_) ||
					(min_nonzero_depth_ == 0))
					min_nonzero_depth_ = val;
			}
		}
	}

	max_depth_ws_ = max_depth_ * agisoft_to_world_scale_;
	min_nonzero_depth_ws_ = min_nonzero_depth_ * agisoft_to_world_scale_;

	if (debug) {
		cout << "Depth map for camera " << cam_id_ << endl;
		cout << "min_nonzero_depth_ " << min_nonzero_depth_ << endl;
		cout << "max_depth_ " << max_depth_ << endl;
		cout << "min_nonzero_depth_ws_ " << min_nonzero_depth_ws_ << endl;
		cout << "max_depth_ws_ " << max_depth_ws_ << endl;
		cout << endl << endl;
	}
}

// set member values according to data in node argument from Agisoft doc.xml file
// agisoft_to_world_scale_ is the scale factor associated with the change from Agisoft's space to our world space, given by the chunk transform in the xml file from Agisoft
// depth downscale is the downward scale factor as given by Agisoft for the depth map from the original image
void DepthMap::Init(string scene_name, xml_node<> *depthmap_node, double agisoft_to_world_scale, int depth_downscale, GLOBAL_AGI_CAMERA_ORIENTATION orientation) {
	bool debug = false;

	assert(strcmp(depthmap_node->name(), "depth_map") == 0);

	depth_downscale_ = depth_downscale;
	agisoft_to_world_scale_ = agisoft_to_world_scale;
	orientation_ = orientation;

	std::string s;
	xml_node<> *curr_node;

	s = depthmap_node->first_attribute("camera_id")->value();
	if (!s.empty()) cam_id_ = convert_string_to_int(s);

	fn_ = depthmap_node->first_attribute("path")->value();

	curr_node = depthmap_node->first_node("calibration");
	calib_.Init(curr_node);
	
	// load depth map .exr image into array
	Array2D<float> zPixels;
	char* fn = convert_string_to_chars(GLOBAL_FILEPATH_DATA + scene_name + "\\" + GLOBAL_FOLDER_AGISOFT + fn_);
	//readHeader(fn);
	readGZ1(fn, zPixels, width_, height_);
	delete[] fn;
	assert(height_ > 0 && width_ > 0);
	assert(height_ == calib_.height_ && width_ == calib_.width_);
	
	// load depth map values into imgD_ and find minimum and maximum depth values
	depth_map_ = Matrix<float, Dynamic, Dynamic>(height_, width_);
	for (int r = 0; r < height_; r++) {
		for (int c = 0; c < width_; c++) {
			float val = (float)zPixels[r][c];
			depth_map_(r, c) = val;
		}
	}
	/*
	int tmp;
	MatrixXf dmtmp;
	switch (orientation) {
	case AGO_ROTATED_RIGHT: // must be rotated left (transpose with vertical flip)
		depth_map_.transposeInPlace();
		dmtmp = depth_map_;
		for (int c = 0; c < depth_map_.cols(); c++) {
			for (int r = 0; r < depth_map_.rows(); r++) {		
				int newr = depth_map_.rows() - 1 - r;
				depth_map_(newr, c) = dmtmp(r, c);
			}
		}
		tmp = height_;
		height_ = width_;
		width_ = tmp;
		break;
	case AGO_ROTATED_LEFT: // must be rotated right (transpose with horizontal flip)
		depth_map_.transposeInPlace();	
		dmtmp = depth_map_;
		for (int c = 0; c < depth_map_.cols(); c++) {
			for (int r = 0; r < depth_map_.rows(); r++) {
				int newc = depth_map_.cols() - 1 - c;
				depth_map_(r, newc) = dmtmp(r, c);
			}
		}
		tmp = height_;
		height_ = width_;
		width_ = tmp;
		break;
	case AGO_ORIGINAL: // fine as is - same as default
		break;
	default:
		break;
	}
	*/
	if (debug) DisplayDepthImage(&depth_map_);
}

void DepthMap::Export(string filepath) {
	bool debug = false;

	// load depth map values into array for writing
	float *zPixels = new float[height_*width_];
	int i = 0;
	for (int r = 0; r < height_; r++) {
		for (int c = 0; c < width_; c++) {
			zPixels[i] = depth_map_(r, c);
			i++;
		}
	}

	if (debug) {
		for (int i = 0; i < height_*width_; i++) {
			cout << "i " << i << ": " << zPixels[i] << endl;
		}
	}

	// write data to file
	string fn_out = "depth" + to_string(cam_id_) + ".exr";
	char* fn = convert_string_to_chars(filepath + fn_out);
	writeGZ1(fn, zPixels, width_, height_);
	delete[] fn;

	delete zPixels;
}

// imgMask is a binary mask to use when computing minimum and maximum depth values - only those values where the mask is >0 are considered
void DepthMap::UpdateFromWSDepths(float *data, Mat *imgMask) {
	depth_map_ = Map<Matrix<float, Dynamic, Dynamic>>(data, height_, width_);
	depth_map_ = depth_map_.array() * (1 / agisoft_to_world_scale_); // convert to agisoft space unit scale

	UpdateMinMaxDepths(imgMask);
}

// reads data from a single-channel .exr depth file into array zPixels and sets corresponding width and height values
void DepthMap::readGZ1(const char fileName[], Array2D<float> &zPixels, int &width, int &height) {
	InputFile file(fileName);
	Box2i dw = file.header().dataWindow();
	width = dw.max.x - dw.min.x + 1;
	height = dw.max.y - dw.min.y + 1;
	zPixels.resizeErase(height, width);
	FrameBuffer frameBuffer;
	frameBuffer.insert("Z", // name
		Slice(FLOAT, // type
		(char *)(&zPixels[0][0] - // base
		dw.min.x -
		dw.min.y * width),
		sizeof (zPixels[0][0]) * 1, // xStride
		sizeof (zPixels[0][0]) * width,// yStride
		1, 1, // x/y sampling
		FLT_MAX)); // fillValue
	file.setFrameBuffer(frameBuffer);
	file.readPixels(dw.min.y, dw.max.y);
}

// writes data to a single-channel .exr depth file with corresponding width and height values; data must be in scan-line order
void DepthMap::writeGZ1(const char fileName[], const float *zPixels, int width, int height) {
	Header header(width, height);
	header.channels().insert("Z", Channel(FLOAT));

	OutputFile file(fileName, header);
	FrameBuffer frameBuffer;
	frameBuffer.insert("Z", // name
		Slice(FLOAT, // type
		(char *)zPixels, // base
		sizeof (*zPixels) * 1, // xStride
		sizeof (*zPixels) * width)); // yStride
	file.setFrameBuffer(frameBuffer);
	file.writePixels(height);
}

// retrieves a depth value from array zPixels at location (x,y)
float DepthMap::DepthVal(Array2D<float> zPixels, int x, int y) {
	return (float)zPixels[y][x];
}

// returns the depth map image converted to world space (since is stored in Agisoft space to make reprojection faster and easier)
Matrix<float, Dynamic, Dynamic> DepthMap::GetDepthMapInWorldSpace() {
	return agisoft_to_world_scale_ * depth_map_;
}

// returns the depth value converted to world space units from Agisoft units
float DepthMap::GetDepthInWorldSpace(float depth) {
	return agisoft_to_world_scale_ * depth;
}

// displays a grayscale visualization of the depth map image given in the argument (Mat of type CV_32F)
void DepthMap::DisplayDepthImage(Matrix<float, Dynamic, Dynamic> *depth_map, GLOBAL_AGI_CAMERA_ORIENTATION orientation, std::string winname) {

	// find max and min depths in image
	float max_depth = 0;
	float min_nonzero_depth = 0;
	float val;
	for (int r = 0; r < depth_map->rows(); r++) {
		for (int c = 0; c < depth_map->cols(); c++) {
			val = (*depth_map)(r, c);
			if (val > 0) {
				if (val > max_depth)
					max_depth = val;
				if ((val < min_nonzero_depth) ||
					(min_nonzero_depth == 0))
					min_nonzero_depth = val;
			}
		}
	}

	Mat imgGray = Mat(depth_map->rows(), depth_map->cols(), CV_8UC1, Scalar(0));

	if (max_depth != min_nonzero_depth) {
		float rangeDepth = max_depth - min_nonzero_depth;

		for (int r = 0; r < depth_map->rows(); r++) {
			for (int c = 0; c < depth_map->cols(); c++) {
				val = (*depth_map)(r, c);
				if (val > 0) {
					float scaledVal = (max_depth - val) / rangeDepth;
					float grayVal = round(scaledVal * 255.0);
					if (grayVal < 0) grayVal = 0;
					if (grayVal > 255) grayVal = 255;
					imgGray.at<uchar>(r, c) = (uchar)grayVal;
				}
			}
		}
	}
	
	if (winname.compare("") == 0)
		display_mat(&imgGray, "Depth Map", orientation);
	else {
		display_mat_existingwindow(&imgGray, winname, orientation);
		waitKey(1);
	}
}

// displays a grayscale visualization of the depth map image given in the argument (Mat of type CV_32F)
void DepthMap::DisplayDepthImage(std::string winname) {

	// find max and min depths in image
	float max_depth = 0;
	float min_nonzero_depth = 0;
	float val;
	for (int r = 0; r < depth_map_.rows(); r++) {
		for (int c = 0; c < depth_map_.cols(); c++) {
			val = depth_map_(r, c);
			if (val > 0) {
				if (val > max_depth)
					max_depth = val;
				if ((val < min_nonzero_depth) ||
					(min_nonzero_depth == 0))
					min_nonzero_depth = val;
			}
		}
	}

	Mat imgGray = Mat(depth_map_.rows(), depth_map_.cols(), CV_8UC1, Scalar(0));

	if (max_depth != min_nonzero_depth) {
		float rangeDepth = max_depth - min_nonzero_depth;

		for (int r = 0; r < depth_map_.rows(); r++) {
			for (int c = 0; c < depth_map_.cols(); c++) {
				val = depth_map_(r, c);
				if (val > 0) {
					float scaledVal = (max_depth - val) / rangeDepth;
					float grayVal = round(scaledVal * 255.0);
					if (grayVal < 0) grayVal = 0;
					if (grayVal > 255) grayVal = 255;
					imgGray.at<uchar>(r, c) = (uchar)grayVal;
				}
			}
		}
	}
	
	if (winname.compare("") == 0)
		display_mat(&imgGray, "Depth Map", orientation_);
	else {
		display_mat_existingwindow(&imgGray, winname, orientation_);
		waitKey(1);
	}
}

void DepthMap::readHeader(const char fileName[]) {
	//
	// Read an image's header from a file, and if the header
	// contains comments and camera transformation attributes,
	// print the values of those attributes.
	//
	//	- open the file
	//	- get the file header
	//	- look for the attributes
	//

	InputFile file(fileName);

	const StringAttribute *comments =
		file.header().findTypedAttribute <StringAttribute>("comments");

	const M44fAttribute *cameraTransform =
		file.header().findTypedAttribute <M44fAttribute>("cameraTransform");

	if (comments)
		cout << "comments\n   " << comments->value() << endl;

	if (cameraTransform)
		cout << "cameraTransform\n" << cameraTransform->value() << flush;
}

void DepthMap::Downsample(float downsample_factor) {
	bool debug = false;

	if (debug) {
		cout << "Depth map before downsampling" << endl;
		DisplayDepthImage(&depth_map_);
	}

	Matrix<bool, Dynamic, 1> maskcurr(height_*width_, 1);
	bool *pM = maskcurr.data();
	float *pD = depth_map_.data();
	for (int c = 0; c < width_; c++) {
		for (int r = 0; r < height_; r++) {
			if (*pD++ != 0.)
				*pM = true;
			pM++;
		}
	}

	MatrixXf depth_map_tmp(height_, width_);
	depth_map_tmp = depth_map_;
	depth_map_tmp.resize(height_*width_, 1);
	Matrix<float, Dynamic, 1> depth_map_curr = depth_map_tmp;

	int target_height = round(static_cast<float>(height_)* downsample_factor);
	int target_width = round(static_cast<float>(width_)* downsample_factor);

	Matrix<float, Dynamic, 1> coordsnewX(target_height*target_width, 1);
	Matrix<float, Dynamic, 1> coordsnewY(target_height*target_width, 1);
	float *pCx = coordsnewX.data();
	float *pCy = coordsnewY.data();
	for (int c = 0; c < target_width; c++) {
		for (int r = 0; r < target_height; r++) {
			*pCx++ = c / downsample_factor;
			*pCy++ = r / downsample_factor;

		}
	}

	Matrix<float, Dynamic, 1> depth_map_new(target_height*target_width, 1);
	float oobv = 0.;
	Interpolation::Interpolate(width_, height_, &depth_map_curr, &coordsnewX, &coordsnewY, &maskcurr, oobv, &depth_map_new);
	
	if (debug) {
		DebugPrintMatrix(&coordsnewX, "coordsnewX");
		DebugPrintMatrix(&coordsnewY, "coordsnewY");
		DisplayImages::DisplayGrayscaleImage(&maskcurr, height_, width_);
		DisplayImages::DisplayGrayscaleImage(&depth_map_curr, height_, width_);
	}

	depth_map_.resize(target_height*target_width, 1);
	depth_map_ = depth_map_new;
	depth_map_.resize(target_height, target_width);

	height_ = target_height;
	width_ = target_width;
	calib_.RecalibrateNewSS(cv::Size(width_, height_));

	if (debug) {
		cout << "Depth map after downsampling" << endl;
		DisplayDepthImage(&depth_map_);
	}
}

// debug printing
void DepthMap::Print() {
	cout << "Depth Map" << endl;
	cout << "Camera ID " << cam_id_ << endl;
	cout << "Width, height " << width_ << ", " << height_ << endl;
	cout << "Maximum depth " << max_depth_ << endl;
	cout << "Minimum non-zero depth " << min_nonzero_depth_ << endl;
	calib_.Print();
	DisplayDepthImage(&depth_map_);
	cout << endl;
}