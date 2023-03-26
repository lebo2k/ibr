#include "DepthMap.h"

DepthMap::DepthMap() {
	fn_ = "";
	cam_id_ = -1;
	min_nonzero_depth_ = 0.;
	max_depth_ = 0.;
	width_ = 0;
	height_ = 0;
}

DepthMap::~DepthMap() {
}

// set member values according to data in node argument from Agisoft doc.xml file
// agisoft_to_world_scale_ is the scale factor associated with the change from Agisoft's space to our world space, given by the chunk transform in the xml file from Agisoft
// depth downscale is the downward scale factor as given by Agisoft for the depth map from the original image
void DepthMap::Init(xml_node<> *depthmap_node, double agisoft_to_world_scale, int depth_downscale) {
	bool debug = false;

	assert(strcmp(depthmap_node->name(), "depth_map") == 0, "DepthMap::DepthMap() wrong arg node type passed");

	depth_downscale_ = depth_downscale;
	agisoft_to_world_scale_ = agisoft_to_world_scale;

	std::string s;
	xml_node<> *curr_node;

	s = depthmap_node->first_attribute("camera_id")->value();
	if (!s.empty()) cam_id_ = convert_string_to_int(s);

	fn_ = depthmap_node->first_attribute("path")->value();

	curr_node = depthmap_node->first_node("calibration");
	calib_.Init(curr_node);
	
	// load depth map .exr image into array
	Array2D<float> zPixels;
	char* fn = convert_string_to_chars(GLOBAL_FILEPATH_INPUT + fn_);
	//readHeader(fn);
	readGZ1(fn, zPixels, width_, height_);
	delete[] fn;
	assert(height_ > 0 && width_ > 0, "DepthMap::Init() image not found");
	assert(height_ == calib_.height_ && width_ == calib_.width_, "DepthMap::Init() size of loaded image does not match size given by Agisoft doc.xml");
	
	// load depth map values into imgD_ and find minimum and maximum depth values
	imgD_ = Mat(height_, width_, CV_32F, Scalar(0));
	max_depth_ = 0;
	min_nonzero_depth_ = 0;
	for (int r = 0; r < height_; r++) {
		for (int c = 0; c < width_; c++) {
			float val = (float)zPixels[r][c];
			imgD_.at<float>(r, c) = val;
			if (val > 0) {
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
		cout << endl << endl;
	}

	if (debug) DisplayDepthImage(&imgD_);
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

// retrieves a depth value from array zPixels at location (x,y)
float DepthMap::DepthVal(Array2D<float> zPixels, int x, int y) {
	return (float)zPixels[y][x];
}

// returns the depth map image converted to world space (since is stored in Agisoft space to make reprojection faster and easier)
Mat DepthMap::GetDepthMapInWorldSpace() {
	return agisoft_to_world_scale_ * imgD_;
}

// displays a grayscale visualization of the depth map image given in the argument (Mat of type CV_32F)
void DepthMap::DisplayDepthImage(Mat *img32F) {
	float* p;

	// find max and min depths in image
	float max_depth = 0;
	float min_nonzero_depth = 0;
	float val;
	for (int r = 0; r < img32F->rows; r++) {
		p = img32F->ptr<float>(r);
		for (int c = 0; c < img32F->cols; c++) {
			val = p[c];
			if (val > 0) {
				if (val > max_depth)
					max_depth = val;
				if ((val < min_nonzero_depth) ||
					(min_nonzero_depth == 0))
					min_nonzero_depth = val;
			}
		}
	}

	Mat imgGray = Mat(img32F->rows, img32F->cols, CV_8UC1, Scalar(0));
	float rangeDepth = max_depth - min_nonzero_depth;

	
	for (int r = 0; r < img32F->rows; r++) {
		p = img32F->ptr<float>(r);
		for (int c = 0; c < img32F->cols; c++) {
			val = p[c];
			if (val > 0) {
				float scaledVal = (max_depth - val) / rangeDepth;
				float grayVal = round(scaledVal * 255.0);
				if (grayVal < 0) grayVal = 0;
				if (grayVal > 255) grayVal = 255;
				imgGray.at<uchar>(r, c) = (uchar)grayVal;
			}
		}
	}

	display_mat(&imgGray, "Depth Map");
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

// debug printing
void DepthMap::Print() {
	cout << "Depth Map" << endl;
	cout << "Camera ID " << cam_id_ << endl;
	cout << "Width, height " << width_ << ", " << height_ << endl;
	cout << "Maximum depth " << max_depth_ << endl;
	cout << "Minimum non-zero depth " << min_nonzero_depth_ << endl;
	calib_.Print();
	DisplayDepthImage(&imgD_);
	cout << endl;
}