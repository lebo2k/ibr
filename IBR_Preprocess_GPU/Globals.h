#ifndef GLOBALS_H
#define GLOBALS_H

/*

Mask Colors

White: Masked-in
Black: Masked-out
Grays: Segmentation lines (treated as masked-in)
Blue: denotes valid-max disparity area
Red: denotes area occluded by background object.  Treat as masked-out except for purposes of computing valid disparity ranges, wherein it is masked-in.

*/

//Only support assertions in debug builds
#ifdef _DEBUG
# include "assert.h"
#else
# define assert(x) { }
#endif

// Python - since Python may define some pre-processor definitions which affect the standard headers on some systems, you must include Python.h before any standard headers are included.
//#include "C:/Python34/include/Python.h" // all function, type and macro definitions needed to use the Python/C API are included in your code by this line

// Standard
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <algorithm>

// cvblob
#include <cvblob.h>

// OpenCV
//#include "cv.h" // deprecated header file
#include "highgui.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/photo/photo_c.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ocl/ocl.hpp"
#include "opencv2/ocl/matrix_operations.hpp"
//#include "opencv2/core/eigen.hpp" // reinstate to use Eigen, but need to get past error at compilation when do so

// Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>

// coefficient access methods (e.g. matrix(i,j)) in Eigen have assertions checking the ranges. So if you do a lot of coefficient access, these assertions can have an important cost. If you want to save cost, define EIGEN_NO_DEBUG, and it won't check assertions
#ifndef EIGEN_NO_DEBUG
#define EIGEN_NO_DEBUG
#endif

// Eigen typedefs ... update this section using comments in http://stackoverflow.com/questions/14783329/opencv-cvmat-and-eigenmatrix

// RapidXML includes
#include "rapidxml-1.13/rapidxml.hpp"
//#include "rapidxml-1.13/rapidxml_utils.hpp"
//#include "rapidxml-1.13/rapidxml_print.hpp"
//#include "rapidxml-1.13/rapidxml_iterators.hpp"
using namespace rapidxml;

#include "HelperFunctions.h"
#include "EigenOpenCV.h"
#include "EigenMatlab.h"
#include "Math3d.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace cvb;

const std::string GLOBAL_FILEPATH_DATA = "C:\\Users\\lebo\\Documents\\Adornably\\Engineering\\IBR\\Data\\";
const std::string GLOBAL_FILEPATH_BLENDER_EXECUTABLE = "C:\\Program Files\\Blender Foundation\\Blender\\"; // used for command line calls to Blender to, for example, decimate a mesh
const std::string GLOBAL_FILEPATH_SOURCE = "C:\\Users\\lebo\\Documents\\Adornably\\Engineering\\IBR\\IBR_Preprocess\\src\\"; // used to reference Python files with full path in command line calls to Blender
const string GLOBAL_FOLDER_PHOTOS = "Photos\\"; // relative folder path for camera photos
const string GLOBAL_FOLDER_MASKS = "Masks\\"; // relative folder path for camera photo masks
const string GLOBAL_FOLDER_AGISOFT = "Agisoft\\"; // relative folder path for Agisoft data

const int GLOBAL_MAX_IMAGE_DISPLAY_SIDE_LENGTH = 1024;

const int GLOBAL_NUM_PROCESSORS = 8; // number of processing cores; used in structuring code to prep for parallel processing with OpenMP

const double GLOBAL_FLOAT_ERROR = 0.00001;

const float GLOBAL_MIN_CS_DEPTH = 0.01;

const float GLOBAL_THRESHOLD_FACTOR_DEPTH_DISCONTINUITY = 4.; // threshold Td for identifying high discontinuities in depth is (max_depth-min_depth)/GLOBAL_THRESHOLD_FACTOR_DEPTH_DISCONTINUITY; threshold should equal about 25% of the maximal depth value

const double GLOBAL_MIN_VIEWHEIGHT_WORLDSPACE = 0.5; // minimum viewing height in world space in meters

const double GLOBAL_EXTEND_DEPTH_RANGE = 0.1; // decimal percentage to extend range of possible depth values from initial depth values when computing missing depth values

const int GLOBAL_JPG_WRITE_QUALITY = 98; // must range from 0 to 100, the higher the better.  OpenCV defaults to 95.

const int GLOBAL_MAX_MEAN_INTERACTIONS = 100; // maximum mean interactions for StereoReconstruction::FindInteractions()

const bool GLOBAL_QPBO_USE_LARGER_INTERNAL = false; // true if use a 64 bit internal representation for integer values, false if stay with 32 bit, which is input value size; only set true if want to use QPBO-P

// Valid ranges parameters
const int GLOBAL_MASK_DILATION = 0; // element size for morphological dilation of masks to provide leeway when constraining depth values.  Leeway important because of potential errors in camera pose that may make true depths fail mask tests from other camera angles for which there is pose error.
const float GLOBAL_RATIO_PASS_DILATED_MASKS = 0.90; // if this decimal percentage of reprojected unknown pixels from the reference camera fall within masked-in pixels of a dilated mask of an input camera in that camera's screen space, the input camera is considered to have passed the pose accuracy test
const float GLOBAL_RATIO_PASS_MASKS_TO_BE_VALID = 1;// 0.95; // // since camera poses are not exact, some disparities may fail a valid range on a camera due to pose estimation error for that camera.  This is especially common for thin areas of the mask.  To combat this issue, only require the pixel disparity to pass a certain percentage of cameras' masks, assuming it may fail some due to the error.  This variable holds the decimal percentage.

// energy minimization parameters
const double GLOBAL_LABELING_ENERGY_DISPARITY_THRESHOLD = 0.02;// 0.2;// 0.02; // ensures smoothness triple-cliques have some level of impact even if the pixels in the cliques have extremely low energy due to photoconsistency (good color match across images at current disparity); necessary because smoothness edge weighting is multiplicatively applied to existing energies
const double GLOBAL_LABELING_ENERGY_COL_THRESHOLD = 30.;// 155;// 30.; // scalar noise parameter for data likelihood
const double GLOBAL_LABELING_ENERGY_OCCLUSION_CONSTANT = 0.01; // scalar occlusion cost
const int GLOBAL_LABELING_ENERGY_LAMBDA_L = 9; // scalar smoothness prior weight for cliques crossing segmentation boundaries
const int GLOBAL_LABELING_ENERGY_LAMBDA_H = 108; // scalar smoothness prior weight for cliques not crossing segmentation boundaries

// mean-shift over-segmentation parameters for depth labeling
const int GLOBAL_MEAN_SHIFT_SEGMENTATION_SIGMAS = 12;// 4;
const double GLOBAL_MEAN_SHIFT_SEGMENTATION_SIGMAR = 5;// 5.;
const int GLOBAL_MEAN_SHIFT_SEGMENTATION_MINREGION = 0;

// optimization settings
const bool GLOBAL_OPTIMIZATION_COMPRESS_GRAPH = true; // I don't trust this at the moment because changes indices; especially don't trust it with seglabel changes to QPBO_eval()
const int GLOBAL_OPTIMIZATION_MAX_ITERS = 250; //3000 // maximum number of optimization iterations, if doesn't converge first
const double GLOBAL_OPTIMIZATION_CONVERGE = 0.005; // loop until percentage decrease in energy per loop is less than this value (so, if converge_==101, loop once); this is a %, not a decimal %, so 100 means 100%
const int GLOBAL_OPTIMIZATION_AVERAGE_OVER = 3;// 20; // number of iterations over which to average when checking convergence
const bool GLOBAL_OPTIMIZATION_DEBUG_VIEW_PLOTS = false; // true to view intermediate plots during optimization, false otherwise
const float GLOBAL_RANDDISP_MAXPERCDEPTH = 0.25; // maximum percentage of total valid range within which to generate random disparity values; used to model the fact that actual disparities tend to be near the front of the range (near min depth / max disparity)
const bool GLOBAL_TRUST_AGISOFT_DEPTHS = false; // if true, non-zero depths imported from Agisoft are considered "known" when performing stereo reconstruction and will not be altered unless and until syncing across cameras is performed; if false, they are considered "unknown" and may be altered
const int GLOBAL_NUMBER_SMOOTHS_PER_SMOOTHSTAR_ITER = 20; // number of smooth passes to perform on each SmoothStar smoothing iteration
const double GLOBAL_DEPTH_DIFF_MAX_NEIGHBORS_WITHIN_SEGMENT = 0.01; // max desired WS depth difference between neighboring pixels within the same label segment.  Used during optimization to assign Kinf_ energy to triple-cliques that include pixel neighbors with a depth difference greater than this threshold
const int GLOBAL_PIXEL_THRESHOLD_FOR_MIN_INPUT_CAMS = 400000; // limit the umber of input images participating in optimization according to the number of pixels in the segment to prevent extremely long computation times for large segments since large segments generally also need fewer input image pairwise ePhoto comparisons to arrive at the correct answer; this variable holds the the threshold minimum number of pixels at which the number of input images (excluding the reference image) is constrained to 1
const int GLOBAL_MAX_RECONSTRUCTION_CAMERAS = 10;// 3;// 8;// 10; // maximum number of other cameras to use during reconstruction for any one camera's depth values; number of cameras is further constrained by segment according to how many pixels are in the segment to reduce running time during optimization

// planar segmentation settings
const int GLOBAL_PLNSEG_WINDOW = 2; // half-size of window to use in window matching

const int GLOBAL_MIN_SEGMENT_PIXELS = 50; // minimum number of pixels in a label segment for StereoData::InitMaskSegmentation()

//const double GLOBAL_MAX_ANGLE_DEGREES_BTWN_CAM_VIEW_DIRS_FOR_POSE_TRUST = 120; // since Agisoft camera pose estimation has greater error as cameras get farther apart, set a maximum angle in degrees between two cameras' view directions in WS for which the relative pose is trusted

const enum GLOBAL_PROPOSAL_METHOD { SAME_UNI = 1, SEG_PLN = 2, SMOOTH_STAR = 3, PERC_UNI = 4 }; // stereo optimization proposal methods; SAME_UNI means random front-parallel, SEG_PLAN means prototypical segment-based stereo proposals, SMOOTH_STAR means smooth*

const enum GLOBAL_TYPE_NAME { TYPE_UNSIGNED_INT = 1, TYPE_INT = 2, TYPE_FLOAT = 3, TYPE_DOUBLE = 4, TYPE_BOOL = 5 }; // type name strings for saving and loading matrices

// Gaussian smoothing
const int GLOBAL_SMOOTH_KERNEL_SIZE = 7; // must be odd; kernel size for Gaussian smoothing of disparity maps
const int GLOBAL_SMOOTH_ITERS = 25; // number of Gaussian smoothing iterations to perform on each smoothing function call

// meshing
const double GLOBAL_MESH_EDGE_DISTANCE_MAX = 0.05; // threshold between a valid edge and two disconnected vertices for meshes constructed from screen space views with depth; any edge distance greater than or equal to this is rejected
const int GLOBAL_MESH_EDGE_DISPARITY_LABEL_SLOPE_DIFF_MAX = 3; // threshold between neighboring disparity label slopes in a triple-clique of pixels; used for image segmentation to determine edge weights within versus across boundaries in Optimization::InitEW_new()
const double GLOBAL_FACE_EXPERT_ANGLE_THRESHOLD = 45; // maximum angle between face an originating camera within which the camera is considered an "expert" on depths for the face's vertices in computing Camera::ReprojectMeshDepths()
const int GLOBAL_TARGET_MESH_FACES = 10000; // target number of faces per mesh after decimation
const int GLOBAL_TARGET_MAX_FILE_SIZE = 20000000; // target maximum number of bytes in mesh obj file; used to determine whether StereoData::DecimateMeshes() worked or Blender crashed in the process since not 100% reliable
const int GLOBAL_MIN_CONNECTED_COMPONENT_FACES = 250; // minimum number of faces required to retain a connected component of a mesh
const double GLOBAL_MESH_EDGE_SMOOTH_WEIGHT_CURR_POS = 0.2; // for mesh edge smoothing in StereoData::ConstructSmoothedMaskedInSSCoordsCM(), sets weight of current position in each iteration of the smoothing function against averaging of 4-connecting canonical edge neighbor positions
const double GLOBAL_MESH_EDGE_SMOOTH_BUFFER_HALF = 0.05; // for mesh edge smoothing in StereoData::ConstructSmoothedMaskedInSSCoordsCM(), half the buffer size between positions of neighboring pixels in the regular lattice to ensure no crossovers and degenerate meshes
const int GLOBAL_MESH_EDGE_SMOOTH_ITERS = 5; // for mesh edge smoothing in StereoData::ConstructSmoothedMaskedInSSCoordsCM(), the number of smoothing iterations to perform

const double GLOBAL_DEPTH_EXPECTED_COMPUTATION_DIST_ERROR = 0.1;

// ArcBall viewer controls
const float GLOBAL_YPIXELDIST_TO_ABZDIST = 0.005; // distance along Z axis in ArcBall space to which a pixel distance in screen space y direction corresponds; used for virtual camera zooming

// Downsampling
const float GLOBAL_DOWNSAMPLE_FACTOR = 1;// 0.5; // factor by which to downsample images before computation (to cut pixel size along each axis in half, use 0.5; use 1 for no change)

// NVS settings
const bool GLOBAL_INPAINT_VIEWS = false; // if true, inpaints any holes in new view synthesis; false otherwise; there's an issue with inpainting right now because can't be sure which blank pixels are holes and which are intentionally masked-out
const bool GLOBAL_SHARPEN = true; // if true, sharpens NVS using Fitzgibbon approach

// Line segments on masks
const int GLOBAL_MIN_MASKSEG_LINEVAL = 80; // minimum recognized uchar grayscale intensity for line segment demarcations on masks
const int GLOBAL_MAX_MASKSEG_LINEVAL = 210; // maximum recognized uchar grayscale intensity for line segment demarcations on masks
const int GLOBAL_MIN_MASK_COLOR_HURDLE = 150; // colored masking may result in anti-aliased colors at the borders; in order to still identify the pixel as masked-in, we use this value as the hurdle for a channel color being intentionally colored and masked-in

// StereoData::BuildValidRanges() assumptions
//const double GLOBAL_MAX_ANGLE_DEGREES_BTWN_CAM_VIEW_DIRS_FOR_POSE_TRUST = 120; // since Agisoft camera pose estimation has greater error as cameras get farther apart, set a maximum angle in degrees between two cameras' view directions in WS for which the relative pose is trusted

// macro debugging
const bool GLOBAL_LOAD_COMPUTED_DISPARITY_MAPS = false;

#endif