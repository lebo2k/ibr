#include "Globals.h"
#include "Scene.h"
#include "StereoReconstruction.h"
#include "NVS.h"
#include "ArcBall.h"

// User Defined Variables
cv::Size sz;
Mat imgT;
NVS *nvs;
Calibration calib;
std::vector<int> exclude_cam_ids;
Mat RT = cv::Mat::zeros(4, 4, CV_64F);
Mat RTinv = cv::Mat::zeros(4, 4, CV_64F);

const float PI2 = 2.0*3.1415926535f;								// PI Squared

Matrix4fT   TransformAB = { 1.0f, 0.0f, 0.0f, 0.0f,	 // ArcBall's RT in ArcBall space (Z in, Y down, X right) in ArcBall format
0.0f, 1.0f, 0.0f, 0.0f,
0.0f, 0.0f, 1.0f, 0.0f,
0.0f, 0.0f, 0.0f, 1.0f };

Matrix3fT   LastRot = { 1.0f, 0.0f, 0.0f,					// NEW: Last Rotation
0.0f, 1.0f, 0.0f,
0.0f, 0.0f, 1.0f };

Matrix3fT   ThisRot = { 1.0f, 0.0f, 0.0f,					// NEW: This Rotation
0.0f, 1.0f, 0.0f,
0.0f, 0.0f, 1.0f };

ArcBallT    ArcBall(640.0f, 480.0f);				                // NEW: ArcBall Instance
Point2fT	MousePtLastRightButtonDown;								// Mouse Point at last right mouse button down event
Point2fT    MousePt;												// NEW: Current Mouse Point
bool isDragging = false;
bool isZooming = false;

// ArcBall space: X right, Y down, Z out
// Camera space: X right, Y down, Z in

Mat RTzoom_in_AB = cv::Mat::eye(4, 4, CV_64F); // zoom along Z-axis in ArcBall space
Mat WScamaligned_from_AB = cv::Mat::eye(4, 4, CV_64F); // transform from ArcBall space to WS origin location with axes aligned with camera space
Mat AB_from_WScamaligned = cv::Mat::eye(4, 4, CV_64F); // transform from WS origin location with axes aligned with camera space to ArcBall space
Mat WScamaligned_from_WS = cv::Mat::eye(4, 4, CV_64F); // transform from WS to WS origin location with axes aligned with camera space
Mat WS_from_WScamaligned = cv::Mat::eye(4, 4, CV_64F); // transform from WS origin location with axes aligned with camera space to WS
Mat RTab; // ArcBall's RT in ArcBall space (Z in, Y down, X right) in OpenCV format
Mat WSoffset_from_WS = cv::Mat::eye(4, 4, CV_64F); // transform from a world space with the axis aligned to camera space rotation to the same but translated so that the origin is at half the z of the bounding volume of the point cloud in world space so we can rotate around the center of the product instead of its base (the product is already centered about x and y and positioned so that its bottom is at z = 0)
Mat WS_from_WSoffset = cv::Mat::eye(4, 4, CV_64F); // inverse of WSoffset_from_WS

int nvs_numcams = 1;//15

double ws_z_offset; // world space Z offset, set to half of scene height, used to set rotation around center of scene rather than origin (scene is already centered around X and Y axes)

void FileIOTest() {
	// init vars
	int cols = 1404 * 960;
	int num_el = 4 * cols;
	Matrix<float, 4, Dynamic> M(4, cols);
	
	M.setRandom();

	char* fn = "C:\\Users\\lebo\\Documents\\Adornably\\Engineering\\IBR\\Data\\file.adf";
	//cout << "M before" << endl << M << endl << endl;

	/// write
	cout << endl << "writing M" << endl << endl;
	FILE* pFile = fopen(fn, "wb"); // write binary mode
	fwrite((void*)M.data(), sizeof(float), num_el, pFile);
	fclose(pFile);

	// overwrite and wipe M to ensure new data is coming in
	cout << endl << "overwrite and wipe M to ensure new data is coming in" << endl << endl;
	M.setRandom();
	//cout << "M overwritten with random data" << endl << M << endl << endl;
	//M.resize(2, 0);
	//cout << "M wiped" << endl << M << endl << endl;
	//M = Matrix<float, 2, Dynamic>(2, cols);

	// read
	cout << endl << "reading" << endl << endl;
	pFile = fopen(fn, "rb"); // read binary mode
	if (pFile == NULL) cerr << "Camera::LoadRenderingData() file not found" << endl;
	float *data = new float[4*cols]; // do not delete it later if reusing allocated memory using placement new
	fread((void*)data, sizeof(float), num_el, pFile);
	fclose(pFile);

	// remap data
	new (&M) Map<Matrix<float, 4, Dynamic>>((float*)data, 4, 3); // new (&M) places an object on memory that's already been allocated rather than copying the memory over to a new location
	Map<Matrix<float, 4, Dynamic>> X((float*)data, 4, 3); // maps to same data as M, though

	// display outcome
	cout << "M after" << endl << M << endl << "# cols: " << M.cols() << endl << endl;
	cout << "X after" << endl << X << endl << endl;
	cin.ignore();
	
}

void test_eigen_ptr_mod(Eigen::Matrix<float, Dynamic, Dynamic> *x, const Eigen::Matrix<float, Dynamic, Dynamic> *y) {
	bool timing = true; double t1, t2;

	x->setConstant(5.);
	(*x)(0, 0) = 7.;
	x->coeffRef(1, 1) = 9.;
	(*x)(2, 2) = (*y)(2, 2);

	if (timing) t1 = (double)getTickCount();
	(*x) = (*y);
	if (timing) {
		t1 = (double)getTickCount() - t1;
		cout << "test_eigen_ptr_mod() non-block version running time = " << t1*1000. / getTickFrequency() << " ms" << endl;
	}

	if (timing) t2 = (double)getTickCount();
	x->block(0, 0, x->rows(), x->cols()) = y->block(0, 0, y->rows(), y->cols());
	if (timing) {
		t2 = (double)getTickCount() - t2;
		cout << "test_eigen_ptr_mod() block version running time = " << t2*1000. / getTickFrequency() << " ms" << endl;
	}
}

void test_read_ppmANDpgm() {
	std::string sfn = GLOBAL_FILEPATH_DATA + "/cones/im0.ppm";
	const char* fncolor = sfn.c_str();
	cv::Mat imgcolor = readPPM(fncolor);

	sfn = GLOBAL_FILEPATH_DATA + "/cones/disp2.pgm";
	const char* fngray = sfn.c_str();
	Eigen::MatrixXf dm = readPGM(fngray);
}

void test_ET_segmentation(Scene *scene) {
	Mat img_start_gray = cv::Mat(scene->cameras_[1]->imgT_.size(), CV_8UC1, Scalar(0));
	cvtColor(scene->cameras_[1]->imgT_, img_start_gray, CV_BGR2GRAY); //convert the color space

	//Mat img_start_gray_ET = DetermineEigenTransform(&img_start_gray);
}

void test_Middlebury() {
	// according to ojw_script.m
	int cid_out = 2;

	StereoReconstruction *sr = new StereoReconstruction("MiddleburyCones");
	sr->Init_MiddleburyCones(cid_out);

	sr->Stereo(cid_out);

	std::string mat_name = sr->scene_name_ + "_Dresult_out_" + to_string(cid_out);
	SaveEigenMatrix(mat_name, TYPE_FLOAT, sr->sd_->depth_maps_[cid_out]);
	DisplayImages::DisplayGrayscaleImage(&sr->sd_->depth_maps_[cid_out], sr->sd_->heights_[sr->cid_out_], sr->sd_->widths_[sr->cid_out_]);

	delete sr;
}

void testMatrixIO() {
	Eigen::Matrix<double, 5, 5> test;
	test << 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25;
	std::string filename = "test";
	SaveEigenMatrix(filename, TYPE_DOUBLE, test);
	cout << "test" << endl << test << endl << endl;
	Eigen::Matrix<double, 5, 5> test2;
	LoadEigenMatrix(filename, TYPE_DOUBLE, test2);
	cout << "test2" << endl << test2 << endl << endl;
	cin.ignore();
}

// visualize reprojections for testing
void test_visualize() {
	Scene *scene = new Scene();
	scene->Init("doc.xml");

	int cid_out2;
	bool out_set = false;
	for (std::map<int, Camera*>::iterator it_in = scene->cameras_.begin(); it_in != scene->cameras_.end(); ++it_in) {
		if (!out_set) {
			cid_out2 = (*it_in).first;
			out_set = true;
			//continue;
		}
		int cid = (*it_in).first;
		if (cid == cid_out2) continue;
		cv::Size view_size((*it_in).second->width_, (*it_in).second->height_);
		Mat imgT = cv::Mat::zeros(view_size, CV_8UC3);
		Matrix<float, Dynamic, Dynamic> depth_map(view_size.height, view_size.width);
		Mat imgMask = cv::Mat::zeros(view_size, CV_8UC1);
		scene->cameras_[cid]->Reproject(&scene->cameras_[cid_out2]->P_, &scene->cameras_[cid_out2]->RT_, &imgT, &depth_map, &imgMask);
		display_mat(&imgT, "imgT", scene->cameras_[cid]->orientation_);
		DepthMap::DisplayDepthImage(&depth_map);
		display_mat(&imgMask, "imgMask", scene->cameras_[cid]->orientation_);
	}
}

// visualize reprojections for testing
void test_visualize_specific() {
	Scene *scene = new Scene();
	scene->Init("doc.xml");

	int cid_out = 9;
	int cid_in = 7;
	DepthMap::DisplayDepthImage(&scene->cameras_[cid_in]->dm_->depth_map_);

	cv::Size view_size(scene->cameras_[cid_in]->width_, scene->cameras_[cid_in]->height_);
	Mat imgT = cv::Mat::zeros(view_size, CV_8UC3);
	Matrix<float, Dynamic, Dynamic> depth_map(view_size.height, view_size.width);
	Mat imgMask = cv::Mat::zeros(view_size, CV_8UC1);
	scene->cameras_[cid_in]->Reproject(&scene->cameras_[cid_out]->P_, &scene->cameras_[cid_out]->RT_, &imgT, &depth_map, &imgMask);
	display_mat(&imgT, "imgT", scene->cameras_[cid_in]->orientation_);
	DepthMap::DisplayDepthImage(&depth_map, scene->cameras_[cid_in]->orientation_);
	display_mat(&imgMask, "imgMask", scene->cameras_[cid_in]->orientation_);
}

void test_index_computations() {
	int ht = 3;
	int wd = 4;
	Matrix<float, Dynamic, Dynamic> A(ht, wd);
	A << 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.;
	Matrix<float, Dynamic, Dynamic, RowMajor> B(ht, wd);
	B = A;
	B.resize(1, ht*wd);

	for (int i = 0; i < ht*wd; i++) {
		Point p = PixIndexBwdRM(i, wd);

		cout << "A (" << p.x << ", " << p.y << ")" << endl;
		cout << A(p.y, p.x) << endl;
	}
	cout << endl << endl;
	for (int i = 0; i < ht*wd; i++) {
		Point p = PixIndexBwdRM(i, ht);

		cout << "B (" << p.x << ", " << p.y << ")" << endl;
		cout << B(0, i) << endl;
	}
	cin.ignore();
}

void test_memory() {
	unsigned long long asize = 1618501248;
	unsigned long long bsize = 4171211648;

	long* a = (long*)malloc(asize);
	long* b = (long*)malloc(bsize*5);

	memset(b, 0, bsize * 5);

	cout << "success" << endl;
	cin.ignore();
}

void test_ptrloop() {
	double t;
	int height = 10000;
	int width = 10000;
	Matrix<double, Dynamic, Dynamic> A(height, width);
	A.setConstant(1.5);

	t = (double)getTickCount();
	
	A *= 5.;
	A = A.array().exp();
	A = A.array() + 1;
	A = A.array().log();
	A = -1 * A.array() + log(2);

	t = (double)getTickCount() - t;
	cout << "eigen time = " << t*1000. / getTickFrequency() << " ms" << endl;

	DebugPrintMatrix(&A, "A 1");

	A.setConstant(1.5);
	double *p = A.data();

	t = (double)getTickCount();
	
	// hand-coded version
	double log2 = log(2);
	for (int c = 0; c < width; c++) {
		for (int r = 0; r < height; r++) {
			*p = -1 * log(exp((*p)*5.) + 1) + log2;
			p++;
		}
	}

	t = (double)getTickCount() - t;
	cout << "hand-coded time = " << t*1000. / getTickFrequency() << " ms" << endl;

	DebugPrintMatrix(&A, "A 2");
}

void onMouse(int event, int x, int y, int flags, void*)
{
	bool debug = false;
	
	MousePt.s.X = (float)x;
	MousePt.s.Y = (float)y;

	if (event == CV_EVENT_LBUTTONUP)
	{
		if (debug) cout << "Mouse left button up at location (" << MousePt.s.X << ", " << MousePt.s.Y << ")" << endl;
		isDragging = false;

		imgT = nvs->SynthesizeView(&RT, exclude_cam_ids, nvs_numcams); // synthesize a new view
		cv::imshow("image", imgT);
	}
	else if (event == CV_EVENT_RBUTTONUP)
	{
		if (debug) cout << "Mouse right button up at location (" << MousePt.s.X << ", " << MousePt.s.Y << ")" << endl;
		isZooming = false;
	}
	else if ((event == CV_EVENT_LBUTTONDOWN) && (!isZooming))
	{
		if (debug) cout << "Mouse left button down at location (" << MousePt.s.X << ", " << MousePt.s.Y << ")" << endl;
		isDragging = true;										// Prepare For Dragging
		LastRot = ThisRot;										// Set Last Static Rotation To Last Dynamic One
		ArcBall.click(&MousePt);								// Update Start Vector And Prepare For Dragging
	}
	else if ((event == CV_EVENT_RBUTTONDOWN) && (!isDragging))
	{
		if (debug) cout << "Mouse right button down at location (" << MousePt.s.X << ", " << MousePt.s.Y << ")" << endl;
		isZooming = true;										// Prepare For Zooming
		LastRot = ThisRot;										// Set Last Static Rotation To Last Dynamic One
		ArcBall.click(&MousePt);								// Update Start Vector And Prepare For Zooming
		MousePtLastRightButtonDown.s.X = x;
		MousePtLastRightButtonDown.s.Y = y;
	}
	else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON) && (!isZooming))
	{
		if (debug) cout << "Dragging mouse to location (" << MousePt.s.X << ", " << MousePt.s.Y << ")" << endl;

		Quat4fT     ThisQuat;

		ArcBall.drag(&MousePt, &ThisQuat);						// Update End Vector And Get Rotation As Quaternion
		Matrix3fSetRotationFromQuat4f(&ThisRot, &ThisQuat);		// Convert Quaternion Into Matrix3fT
		Matrix4fSetRotationFromMatrix3f(&TransformAB, &ThisRot);	// Set Our Final Transform's Rotation From This One

		RTab = Arc4fTToCV(&TransformAB); // converts ArcBall RT to OpenCV format, but still in ArcBall space

		// WS_from_WScamaligned transforms from WS origin location with axes aligned with camera space to WS.  WScamaligned_from_WS does the inverse.  Our camera space coordinate system is z in, x right, y up.
		RTinv.copyTo(WS_from_WScamaligned);
		WS_from_WScamaligned.at<double>(0, 3) = 0.;
		WS_from_WScamaligned.at<double>(1, 3) = 0.;
		WS_from_WScamaligned.at<double>(2, 3) = 0.;
		WScamaligned_from_WS = WS_from_WScamaligned.inv();

		/*
		// enforce minimum height
		// if resulting camera position in offset world space is below (GLOBAL_MIN_VIEWHEIGHT_WORLDSPACE - ws_z_offset) in world space, rotate it around the axis perpendicular both to camera Z and world space Z (use cross product) that puts its world space Z at the minimum (GLOBAL_MIN_VIEWHEIGHT_WORLDSPACE - ws_z_offset)
		Mat RT_offset = WS_from_WSoffset * WS_from_WScamaligned * WScamaligned_from_AB * RTab * AB_from_WScamaligned * WScamaligned_from_WS * WSoffset_from_WS;
		Mat RTinv_offset = RT_offset.inv();

		if (RTinv_offset.at<double>(2, 3) < GLOBAL_MIN_VIEWHEIGHT_WORLDSPACE - ws_z_offset) {
		Mat Z_cs = cv::Mat::eye(4, 4, CV_64F);
		Mat Z_ws =
		}
		*/

		RT = RT * WS_from_WSoffset * WS_from_WScamaligned * WScamaligned_from_AB * RTab * AB_from_WScamaligned * WScamaligned_from_WS * WSoffset_from_WS;

		// else if zooming: RT = RT * RTinv* RTzoom_incameraspace * RT; // RTzoom_incameraspace is an eye matrix with a certain value in (2,3) for translation in/out

		RTinv = RT.inv();

		if (debug) cout << endl << "RT" << endl << RT << endl << endl;

		//imgT = nvs->SynthesizeView(&RT, exclude_cam_ids, nvs_numcams); // synthesize a new view
		//cv::imshow("image", imgT);
	}
	else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_RBUTTON) && (!isDragging))
	{
		if (debug) cout << "Zooming mouse to location (" << MousePt.s.X << ", " << MousePt.s.Y << ")" << endl;

		float ydist = MousePt.s.Y - MousePtLastRightButtonDown.s.Y;
		float ab_z_offset = ydist * GLOBAL_YPIXELDIST_TO_ABZDIST;
		RTzoom_in_AB.at<double>(2, 3) = -1. * ab_z_offset;

		// WS_from_WScamaligned transforms from WS origin location with axes aligned with camera space to WS.  WScamaligned_from_WS does the inverse.  Our camera space coordinate system is z in, x right, y up.
		RTinv.copyTo(WS_from_WScamaligned);
		WS_from_WScamaligned.at<double>(0, 3) = 0.;
		WS_from_WScamaligned.at<double>(1, 3) = 0.;
		WS_from_WScamaligned.at<double>(2, 3) = 0.;
		WScamaligned_from_WS = WS_from_WScamaligned.inv();

		/*
		// enforce minimum height
		// if resulting camera position in offset world space is below (GLOBAL_MIN_VIEWHEIGHT_WORLDSPACE - ws_z_offset) in world space, rotate it around the axis perpendicular both to camera Z and world space Z (use cross product) that puts its world space Z at the minimum (GLOBAL_MIN_VIEWHEIGHT_WORLDSPACE - ws_z_offset)
		Mat RT_offset = WS_from_WSoffset * WS_from_WScamaligned * WScamaligned_from_AB * RTab * AB_from_WScamaligned * WScamaligned_from_WS * WSoffset_from_WS;
		Mat RTinv_offset = RT_offset.inv();

		if (RTinv_offset.at<double>(2, 3) < GLOBAL_MIN_VIEWHEIGHT_WORLDSPACE - ws_z_offset) {
		Mat Z_cs = cv::Mat::eye(4, 4, CV_64F);
		Mat Z_ws =
		}
		*/

		RT = RT * WS_from_WSoffset * WS_from_WScamaligned * WScamaligned_from_AB * RTzoom_in_AB * AB_from_WScamaligned * WScamaligned_from_WS * WSoffset_from_WS;

		// else if zooming: RT = RT * RTinv* RTzoom_incameraspace * RT; // RTzoom_incameraspace is an eye matrix with a certain value in (2,3) for translation in/out

		RTinv = RT.inv();

		if (debug) cout << endl << "RT" << endl << RT << endl << endl;

		imgT = nvs->SynthesizeView(&RT, exclude_cam_ids, nvs_numcams); // synthesize a new view
		cv::imshow("image", imgT);
	}
}

void viewer(Scene *scene, StereoData *sd) {
	for (std::map<int, Camera*>::iterator it_in = scene->cameras_.begin(); it_in != scene->cameras_.end(); ++it_in) {
		int cid = (*it_in).first;
		if ((!(*it_in).second->enabled_) ||
			(!(*it_in).second->posed_) ||
			(!(*it_in).second->has_depth_map_)) continue; // disqualify cameras for which we are missing important information
		scene->cameras_[cid]->BuildMesh();
		scene->CleanFacesAgainstMasks(cid);
	}

	// prep data to initialize NVS
	int cid = scene->cameras_.begin()->first; // select a camera view to recreate
	Size sz(scene->cameras_[cid]->width_ * 0.5, scene->cameras_[cid]->height_ * 0.5);
	
	// create and initialize NVS
	nvs = new NVS();
	nvs->Init(scene, sd, &scene->cameras_[cid]->calib_, sz);
	
	// prep data for synthesis
	//exclude_cam_ids.push_back(cid);
	EigenOpenCV::eigen2cv(nvs->scene_->cameras_[cid]->RT_, RT); // starting position of camera
	RTinv = RT.inv();

	// synthesize a new view
	Mat imgT = nvs->SynthesizeView(&RT, exclude_cam_ids, nvs_numcams);

	// prep matrices for ArcBall
	ws_z_offset = scene->bv_max_.z / 2.;
	WSoffset_from_WS.at<double>(2, 3) = -1. * ws_z_offset;
	WS_from_WSoffset.at<double>(2, 3) = ws_z_offset;

	// Set up ArcBall
	ArcBall.setBounds(sz.width, sz.height);
	AB_from_WScamaligned.at<double>(2, 2) = -1.; // ArcBall coordinate system is rotationally aligned with our camera space coordinate system in x and y (x right, y up) but ArcBall is z out while camera space is z in
	WScamaligned_from_AB = AB_from_WScamaligned.inv();

	// Interactivity with user
	cv::namedWindow("image", 1);
	cv::imshow("image", imgT);
	cv::setMouseCallback("image", onMouse, 0);
	for (;;)
	{
		int c = waitKey(0);

		if ((char)c == 27) // exit;
			break;
		else if ((char)c == 'r') { // reset view
			EigenOpenCV::eigen2cv(nvs->scene_->cameras_[cid]->RT_, RT); // starting position of camera
			RTinv = RT.inv();
			imgT = nvs->SynthesizeView(&RT, exclude_cam_ids, nvs_numcams); // synthesize a new view
			cv::imshow("image", imgT);
		}
	}

	delete nvs;
}

void readfile() {

	std::string fn = "C:\\Users\\lebo\\Documents\\Adornably\\Engineering\\IBR\\Input\\ST-302-25-74\\points0.ply";
	FILE* pFile = fopen(fn.c_str(), "rb");

	/*
	ply
	format binary_little_endian 1.0
	element vertex 162486
	property float x
	property float y
	property float z
	property uchar red
	property uchar green
	property uchar blue
	property uint frame
	property uint flags
	end_header
	*/

	char asset_chars[220];
	std::fread(asset_chars, sizeof(char), 220, pFile);
	
	string s = convert_chars_to_string(asset_chars);
	cout << s << endl;

	char tmp_chars[1];
	/*
	std::fread(asset_chars, sizeof(char), 32, pFile);
	std::fread(asset_chars, sizeof(char), 22, pFile);
	std::fread(asset_chars, sizeof(char), 17, pFile);
	std::fread(asset_chars, sizeof(char), 17, pFile);
	std::fread(asset_chars, sizeof(char), 17, pFile);
	std::fread(asset_chars, sizeof(char), 19, pFile);
	std::fread(asset_chars, sizeof(char), 21, pFile);
	std::fread(asset_chars, sizeof(char), 20, pFile);
	std::fread(asset_chars, sizeof(char), 20, pFile);
	std::fread(asset_chars, sizeof(char), 20, pFile);
	std::fread(asset_chars, sizeof(char), 11, pFile);
	*/
	for (int c = 0; c < 4000; c++) {
		float x, y, z;
		uchar r, g, b;
		unsigned int frame, flags;
		std::fread((void*)&x, sizeof(float), 1, pFile);
		std::fread(tmp_chars, sizeof(char), 1, pFile);
		std::fread((void*)&y, sizeof(float), 1, pFile);
		std::fread(tmp_chars, sizeof(char), 1, pFile);
		std::fread((void*)&z, sizeof(float), 1, pFile);
		std::fread(tmp_chars, sizeof(char), 1, pFile);

		std::fread((void*)&r, sizeof(uchar), 1, pFile);
		std::fread(tmp_chars, sizeof(char), 1, pFile);
		std::fread((void*)&g, sizeof(uchar), 1, pFile);
		std::fread(tmp_chars, sizeof(char), 1, pFile);
		std::fread((void*)&b, sizeof(uchar), 1, pFile);
		std::fread(tmp_chars, sizeof(char), 1, pFile);

		std::fread((void*)&frame, sizeof(unsigned int), 1, pFile);
		std::fread(tmp_chars, sizeof(char), 1, pFile);
		std::fread((void*)&flags, sizeof(unsigned int), 1, pFile);
		std::fread(tmp_chars, sizeof(char), 1, pFile);
	}

	std::fclose(pFile);

	/*
	std::string fn = "C:\\Users\\lebo\\Documents\\Adornably\\Engineering\\IBR\\tmp\\projections21.ply";
	FILE* pFile = fopen(fn.c_str(), "rb");

	
	//ply
	//format binary_little_endian 1.0
	//element vertex 7765
	//property float x
	//property float y
	//property int id
	//end_header
	

	char asset_chars[50];
	std::fread(asset_chars, sizeof(char), 4, pFile);
	std::fread(asset_chars, sizeof(char), 32, pFile);
	std::fread(asset_chars, sizeof(char), 20, pFile);
	std::fread(asset_chars, sizeof(char), 17, pFile);
	std::fread(asset_chars, sizeof(char), 17, pFile);
	std::fread(asset_chars, sizeof(char), 16, pFile);
	std::fread(asset_chars, sizeof(char), 11, pFile);

	for (int c = 0; c < 4000; c++) {
		float x;
		int y;
		std::fread((void*)&x, sizeof(float), 1, pFile);
		std::fread(asset_chars, sizeof(char), 1, pFile);
		std::fread((void*)&x, sizeof(float), 1, pFile);
		std::fread(asset_chars, sizeof(char), 1, pFile);
		std::fread((void*)&y, sizeof(int), 1, pFile);
		std::fread(asset_chars, sizeof(char), 1, pFile);
	}

	std::fclose(pFile);
	*/
}

void test_meshreduction() {
	int id = 43;
	
	string fn = "test\\WH-230-7057_mesh_cam" + to_string(id) + ".obj";
	double dec_ratio = 0.1;
	string command = "cd " + GLOBAL_FILEPATH_BLENDER_EXECUTABLE + " && blender --background --python \"" + GLOBAL_FILEPATH_SOURCE + "decimate.py\" > NUL 2>&1 -- \"" + GLOBAL_FILEPATH_DATA + fn + "\" \"" + GLOBAL_FILEPATH_DATA + fn + "\" " + to_string(dec_ratio);
	
	system(command.c_str());
	cin.ignore();
}

//#include "perf.cu"
int main(int argc, char *argv[]) {

	//test_thrust();
	//return 0;

	bool debug = false;
	bool debug_save_camera_as_computed = true; // if true, saves mesh and masked image camera by camera as they are computed in ReconstructAll() instead of all at once after it finishes running

	srand(static_cast <unsigned> (time(0))); // initialize random seed - should be only called once per run, at its start

	Scene *scene = new Scene();
	//scene->Init("WH-230-7057");
	//scene->Init("HK-3005-75330");
	scene->Init("TA-5105-248");
	bool all_max = false;

	// stereo reconstruction initialization
	std::map<int, Mat> imgsT;
	std::map<int, Mat> imgMasks;
	std::map<int, Mat> imgMasks_valid;
	std::map<int, Eigen::MatrixXf> depth_maps;
	std::map<int, Eigen::Matrix3d> Ks;
	std::map<int, Eigen::Matrix3d> Kinvs;
	std::map<int, Eigen::Matrix4d> RTs;
	std::map<int, Eigen::Matrix4d> RTinvs;
	std::map<int, Matrix<double, 3, 4>> Ps;
	std::map<int, Matrix<double, 4, 3>> Pinvs;
	std::map<int, float> agisoft_to_world_scales;
	std::vector<int> cids; // track cids so can use them later after delete scene
	for (std::map<int, Camera*>::iterator it_in = scene->cameras_.begin(); it_in != scene->cameras_.end(); ++it_in) {
		int cid = (*it_in).first;
		if ((!(*it_in).second->enabled_) ||
			(!(*it_in).second->posed_) ||
			(!(*it_in).second->has_depth_map_)) continue; // disqualify cameras for which we are missing important information

		cids.push_back(cid);

		// set up images and matrices
		Mat imgTm = (*it_in).second->MaskedImgT();
		imgsT[cid] = imgTm;
		imgMasks[cid] = (*it_in).second->imgMask_;
		imgMasks_valid[cid] = (*it_in).second->imgMask_valid_;
		depth_maps[cid] = (*it_in).second->dm_->depth_map_;
		Ks[cid] = (*it_in).second->calib_.K_;
		Kinvs[cid] = (*it_in).second->calib_.Kinv_;
		RTs[cid] = (*it_in).second->RT_;
		RTinvs[cid] = (*it_in).second->RTinv_;
		Ps[cid] = (*it_in).second->P_;
		Pinvs[cid] = (*it_in).second->Pinv_;
		agisoft_to_world_scales[cid] = (*it_in).second->dm_->agisoft_to_world_scale_;
	}
	scene->UpdateCSMinMaxDepths();

	// reconstruct depth maps
 	StereoReconstruction *sr = new StereoReconstruction(scene->name_);
	sr->Init(imgsT, imgMasks, imgMasks_valid, depth_maps, Ks, Kinvs, RTs, RTinvs, Ps, Pinvs, scene->min_depths_, scene->max_depths_, scene->unknown_segs_, agisoft_to_world_scales, scene->AgisoftToWorld_, scene->WorldToAgisoft_, exclude_cam_ids, GLOBAL_MAX_RECONSTRUCTION_CAMERAS, scene);


	/*
	// test loading in a mesh and removing small connected components by segment
	int tcid = 0;
	sr->sd_->LoadMeshes(scene->name_, tcid);
	sr->sd_->RemoveIsolatedFacesFromMeshes(tcid);
	sr->sd_->BuildTextureCoordinates(tcid);
	sr->sd_->SaveMesh(scene->name_, tcid);
	cin.ignore();
	*/





	std::vector<int> exclude_cam_ids;
	sr->ReconstructAll(cids, scene, debug_save_camera_as_computed, all_max);

	if (!debug_save_camera_as_computed) { // otherwise, taken care of in sr->ReconstructAll()
		sr->sd_->SaveMeshes(sr->scene_name_);
		scene->ExportMaskedCameraImages(); // exports each camera's image after masking it
	}

	scene->ExportSceneInfo(); // writes a file that, for each camera, exports camera ID, WS position, WS view direction, and WS up direction

	delete scene;
	delete sr;

	return 0;
}