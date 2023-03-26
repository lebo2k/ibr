#ifndef Camera_H
#define Camera_H

#include "Globals.h"
#include "Sensor.h"
#include "Calibration.h"
#include "DepthMap.h"

class Camera {

private:

	// Members
	Mat imgDdisc_; // type CV_8UC1 binary image of size width_, height_ corresponding to img_, with val=255 for pixels where there is a high depth discontinuity and 0 everywhere else

	// Initialization
	void ParseAgisoftCameraExtrinsics(std::string s, Matrix4d AgisoftToWorld_, Matrix4d AgisoftToWorldinv_); // parse string of 16 ordered doubles (in col then row order) into R_ and T_ matrices for camera extrinsics; it expects the transform being parsed is a camera location in world space, and therefore represents RTinv; see Scene.h for description of AgisoftToWorld
	static Matrix<float, 3, Dynamic> ConstructSSCoordsRM(int ss_w, int ss_h); // returns 3 by (ss_w*ss_h) data structure with homogeneous pixel positions for a screen space of pixel dimensions ss_w*ss_h assuming row-major order of indices
	void InitDepthMapDiscontinuities(); // called by InitDepthMap() to initialize imgDMDisc_
	bool GetNearestDepth(Point p, Point dir, int &dist, float &depth); // used by InpaintDepthMap(); traverses depth image imgD_ from p in direction step until encounters a pixel with a non-zero depth value that is not masked out in imgMask
	void InpaintDepthMap(); // inpaints depth map by searching from each missing pixel (u,v) in 8 canonical directions until reach a pixel with a depth value (ud, pd), then perform a weighted interpolation
	bool DetermineDepthMapDiscontinuity(Point p); // tests pixel at point p for high depth map discontinuity and returns result boolean
	void SegmentImage(int height, int width);

	// Update functions
	void UpdateCameraMatrices(); // updates P_ and Pinv_; requires that calib.K_, RT_, and RTinv_ are set
	void UpdatePos(); // updates member pos_
	void UpdateViewDir(); // updates member view_dir_

	// Convenience functions
	
	bool GetTransformedPosSS(Matrix<float, 3, Dynamic> *PosSS, Point pt_ss, cv::Size sizeTargetSS, Point &pt_ss_transformed); // given pointer to 3xn matrix PosSS that holds tranformed screen space positions where n is the number of pixels in width_*height_ and the order is column then row, updates the rounded transformed screen space coordinates for a given screen space pixel position and returns boolean whether is inside the target screen space or not
	Point3d Camera::GetTransformedPosWS(Matrix<float, 4, Dynamic> *PosWS, Point pt_ss); // given pointer to 4xn matrix PosWS that holds tranformed world space positions where n is the number of pixels in width_*height_ and the order is column then row, returns the transformed world space coordinates for a given screen space pixel position

	// I/O
	char* GetFilename(std::string filepath, std::string scene_name);
	char* GetFilenameMatrix(std::string filepath, std::string scene_name, char* fn_mat_chars);
	void RLE_WriteCount(int contig_pixel_count, std::vector<unsigned short> *rls);

	void CleanFacesAgainstMasks();

public:

	// Members
	int id_; // a unique identifier for this camera
	std::string fn_; // image filename
	std::string fn_mask_; // image mask filename
	int sensor_id_; // ID of associated sensor
	Mat imgT_; // this photo; type CV_8UC3
	Mat imgMask_color_;
	Mat imgMask_; // binary image mask of type CV_8UC1 ("opaque" foreground with value 255, and "transparent" background with value 0)
	Mat imgMask_valid_; // binary image mask of type CV_8UC1 ("opaque" foreground with value 255, and "transparent" background with value 0) to be used for determining valid ranges
	Matrix4d RT_; // 4x4 camera extrinsics matrix; type CV_64F; [R | t] -- [0 | 1]
	Matrix4d RTinv_; // 4x4 inverse camera extrinsics matrix; type CV_64F
	Matrix<double, 3, 4> P_; // 3x4 projection matrix, including intrinsics and extrinsics, where P=K[R|T]; type CV_64F; converts WS to SS
	Matrix<double, 4, 3> Pinv_; // 4x3 inverse projection matrix; type CV_64F; converts SS to WS
	Point3d pos_; // position of camera in world space
	Point3d view_dir_; // view direction of camera in world space
	int width_, height_; // image width and height in pixels
	Calibration calib_;
	GLOBAL_AGI_CAMERA_ORIENTATION orientation_;
	bool enabled_; // whether the camera is enabled in the Agisoft file; it is expected that any disabled camera has no depth map, but we do not rely on that fact in the code
	DepthMap *dm_;
	bool posed_; // true if pose has been calculated, false otherwise
	bool has_depth_map_; // true if the camera has a depth map, false otherwise
	Matrix<float, 4, Dynamic> Iws_; // ordered, columnated, world space homogeneous locations (x,y,z,1) of screen space points; the order of points in Iws_ corresponds to the related pixel, where pixel order is row-major, given by index = (row * width) + col
	Matrix<unsigned int, Dynamic, Dynamic> seg_; // image segmentation based on lines in mask
	map<unsigned int, int> seglabel_counts_; // map of segmentation label => count of pixels with that label present in seg_

	bool closeup_xmin_, closeup_xmax_, closeup_ymin_, closeup_ymax_; // true in cases where photo is a close-up that doesn't fully capture the object within the screen space on the indicated side (value assigned by testing for valid masked-in pixels along the appropriate screen space side's edge)

	Point3d bv_min_, bv_max_; // minimum and maximum bounding volume coordinates for the point cloud captured by this camera in world space ... used to determine minimum and maximum depth bound in camera space

	map<int, Point3d> mesh_vertices_; // map of vertex index => x,y,z world space position of vertex; vertex indices are indices in row-major order for the image
	map<int, Vec3i> mesh_faces_; // map of face index => counter-clockwise ordered pixel indices in the face (triangles only)
	map<int, Point3d> mesh_normals_; // map of face index => normal vector

	// Constructors / destructor
	Camera();
	~Camera();

	// Initialization
	void Init(string scene_name, xml_node<> *camera_node, Matrix4d AgisoftToWorld_, Matrix4d AgisoftToWorldinv_); // see Scene.h for description of AgisoftToWorld
	void Camera::InitSensor(Sensor *sensor); // initializes projection matrices P and Pinv using the camera intrinsics matrix arg K
	void InitDepthMap(string scene_name, xml_node<> *depthmap_node, double agisoft_to_world_scale_, int depth_downscale); // initializes imgDM_ and calls InitDepthMapDiscontinuities(); depth downscale is the downward scale factor as given by Agisoft for the depth map from the original image; agisoft_to_world_scale_ is the scale factor associated with the change from Agisoft's space to our world space, given by the chunk transform in the xml file from Agisoft
	void InitWorldSpaceProjection(); // initializes Iws_ by computing the world space location that corresponds to each pixel in camera screen space
	void DownsampleToMatchDepthMap(); // downsamples images and updates resolution info, camera intrinsics, and projection matrices to match depth map size
	void InitCloseup(); // initializes values for closeup_xmin_, closeup_xmax_, closeup_ymin_, closeup_ymax_ using valid mask data


	void DownsampleAll(float downsample_factor);
	void UndistortPhotos(); // undistorts imgT_, imgD_, and imgMask_ images to correct for radial and tangential distortion
	Mat MaskedImgT(); // returns a cv::Mat of type CV_8UC3 that is imgT after applying mask imgMask_

	Matrix<float, 1, Dynamic> GetWSPriorities(); // returns 1x(ss_width*ss_height) matrix of pixel priority information, indexed in the same order as Iws_

	// Warping
	static void InverseProjectSStoWS(int ss_width, int ss_height, Matrix<float, Dynamic, Dynamic> *depth_map, Matrix3d *Kinv, Matrix4d *RTinv, Matrix<float, 4, Dynamic> *Iws); // inverse projects screen space points (screen space dimensions ss_width x ss_height) with depths given by imgD from screen space to world space using Kinv and RTinv, updating a 4xn matrix of type float of the corresponding points in world space
	void Reproject(Matrix<double, 3, 4> *P_dest, Matrix4d *RT_dest, Mat *imgT, Matrix<float, Dynamic, Dynamic> *depth_map, Mat *imgMask); // reprojects the camera view into a new camera with projection matrix P_dest; only reprojects pixels for which there is depth info; imgT is modified to include texture, imgD to include depth values in this camera's camera space (not projection's camera space), and imgMask to include binary mask values (255 pixel is opaque and 0 pixel is transparent), and all must be same size and types

	// Static convenience functions
	static Point RoundSSPoint(Point2d ptd, int width, int height); // rounds the position of a sub-pixel point in screen space to an integer pixel point in screen space
	static Point3d GetCameraPositionWS(Matrix4d *RTinv); // returns camera position in world space using RTinv inverse extrinsics matrix from argument
	static Point3f GetCameraViewDirectionWS(Matrix4f *RTinv); // // returns camera view direction in world space using RTinv inverse extrinsics matrix from argument
	static Point3d GetCameraViewDirectionWS(Matrix4d *RTinv); // // returns camera view direction in world space using RTinv inverse extrinsics matrix from argument
	static Point3f GetCameraUpDirectionWS(Matrix4f *RTinv); // returns camera up direction in world space using RTinv inverse extrinsics matrix from argument
	static Point3d GetCameraUpDirectionWS(Matrix4d *RTinv); // returns camera up direction in world space using RTinv inverse extrinsics matrix from argument
	static Matrix<double, 3, 4> Extend_K(Matrix3d *K); // converts 3x3 camera intrinsics matrix to 3x4 version with right column of [0 0 0]T
	static Matrix<double, 4, 3> Extend_Kinv(Matrix3d *Kinv); // converts 3x3 inverse camera intrinsics matrix to 4x3 version with bottom row [0 0 1]
	static void ComputeProjectionMatrices(Matrix3d *K, Matrix3d *Kinv, Matrix4d *RT, Matrix4d *RTinv, Matrix<double, 3, 4> *P, Matrix<double, 4, 3> *Pinv); // updates P and Pinv to be 4x4 projection and inverse projection matrices, respectively, from camera intrinsics K and extrinsics RT
	void UpdatePointCloudBoundingVolume(); // updates world space bounding volume in bv_min_ and bv_max around the world space point cloud Iws_

	// I/O
	void SaveRenderingDataRLE(std::string scene_name, float min_disp, float max_disp, float disp_step);
	void LoadRenderingDataRLE(std::string scene_name, float &min_disp, float &max_disp, float &disp_step);
	void SaveMaskedImage(std::string scene_name); // saves a separate image file after masking using imgMask_ and excluding any pixels for which depth values aren't available
	void Save_K(std::string scene_name);
	void Save_RT(std::string scene_name);

	// Debugging
	void Print(); // debug printing
	void SavePointCloud(string scene_name);


	void BuildMesh();
	void ReprojectMesh(Point3d view_dir, Matrix<double, 3, 4> *P_dest, Matrix4d *RT_dest, Mat *imgT, Matrix<float, Dynamic, Dynamic> *depth_map, Mat *imgMask); // reprojects the camera view into a new camera with projection matrix P_dest; only reprojects pixels for which there is depth info; imgT is modified to include texture, imgD to include depth values in this camera's camera space (not projection's camera space), and imgMask to include binary mask values (255 pixel is opaque and 0 pixel is transparent), and all must be same size and types
	void ReprojectMeshDepths(Point3d view_dir, Matrix<double, 3, 4> *P_dest, Matrix4d *RT_dest, Matrix<float, Dynamic, Dynamic> *depth_map, Matrix<bool, Dynamic, 1> *change_map); // like ReprojectMesh() but only computes depths; updates depth_map



	inline void SaveTest(std::string scene_name, float min_disp, float max_disp, float disp_step) {

		std::string fn = GLOBAL_FILEPATH_DATA + scene_name + "\\" + scene_name + "_camtest" + to_string(id_) + ".adf";
		FILE* pFile = fopen(fn.c_str(), "wb"); // write binary mode
	
		unsigned int num_pixels_used = 0;
		unsigned int contig_pixel_count = 0;
		std::vector<unsigned short> rls; // run lengths
		bool last_rl_used; // boolean denoting whether the last run-length was a count of used pixels (true) or unused pixels (false)
		uchar* pM;
		last_rl_used = false;
		for (int r = 0; r < imgMask_.rows; r++) {
			pM = imgMask_.ptr<uchar>(r);
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
				}
			}
		}

		// write data
		int num_rl_ushorts = rls.size();
		std::fwrite((void*)&num_pixels_used, sizeof(unsigned int), 1, pFile);
		std::fwrite((void*)&num_rl_ushorts, sizeof(unsigned int), 1, pFile);
		//std::fwrite((void*)&unused_rows_top, sizeof(unsigned int), 1, pFile);
		int pc;
		for (std::vector<unsigned short>::iterator it = rls.begin(); it != rls.end(); ++it) {
			pc = (*it);
			std::fwrite((void*)&pc, sizeof(unsigned short), 1, pFile);
		}

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

		std::fclose(pFile);
	}
	
	inline void LoadTest(std::string scene_name, float &min_disp, float &max_disp, float &disp_step) {

		std::string fn = GLOBAL_FILEPATH_DATA + scene_name + "\\" + scene_name + "_camtest" + to_string(id_) + ".adf";
		FILE* pFile = fopen(fn.c_str(), "rb"); // read binary mode

		if (pFile == NULL) {
			cerr << "Camera::LoadRenderingData() file not found" << endl;
			return;
		}

		// 7. int number of used pixels, int number of run length unsigned shorts, int number of unused rows at top
		unsigned int num_pixels_used, num_rl_ushorts;
		std::fread((void*)&num_pixels_used, sizeof(unsigned int), 1, pFile);
		std::fread((void*)&num_rl_ushorts, sizeof(unsigned int), 1, pFile);
		
		unsigned short* rls = new unsigned short[num_rl_ushorts]; // run lengths
		std::fread((void*)&rls[0], sizeof(unsigned short), num_rl_ushorts, pFile);

		// 9. sparse pixel quantized disparities, each as a short; they are in raster-scan order, skipping any unused pixels
		unsigned short* disp_labels = new unsigned short[num_pixels_used];
		std::fread((void*)&disp_labels[0], sizeof(unsigned short), num_pixels_used, pFile);
		/*
		for (int i = 0; i < num_pixels_used; i++) {
			std::fread((void*)&disp_labels[i], sizeof(unsigned short), 1, pFile);
		}
		*/

		dm_->depth_map_.setZero(); // clear before loading so unused pixels are blank
		int idx_used = 0; // index into used pixels for disparity labels (disp_quant)
		int idx_all = 0;
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

		std::fclose(pFile);
	}
};

#endif