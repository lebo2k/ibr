#ifndef GLOBALS_H
#define GLOBALS_H

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
#include <functional>
#include <numeric>
#include <stdexcept>
#include <algorithm>

// OpenCV
#include "cv.h"
#include "highgui.h"
#include <cvblob.h>
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

using namespace std;
using namespace cv;
using namespace cvb;
using namespace Eigen;

const std::string GLOBAL_FILEPATH_INPUT = "C:\\Users\\lebo\\Documents\\Adornably\\Engineering\\Warp\\Input\\";
const std::string GLOBAL_FILEPATH_DATA = "C:\\Users\\lebo\\Documents\\Adornably\\Engineering\\Warp\\Data\\";
const int GLOBAL_MAX_IMAGE_DISPLAY_SIDE_LENGTH = 1024;

const enum GLOBAL_AGI_CAMERA_ORIENTATION { AGO_ORIGINAL = 1, AGO_SIDE = 8 }; // Agisoft camera orientation codes; original means matches sensor and side means on its side so width and height are switched

const double GLOBAL_MIN_VIEWHEIGHT_WORLDSPACE = 0.5; // minimum viewing height in world space in meters

const double GLOBAL_EXTEND_DEPTH_RANGE = 0.1; // decimal percentage to extend range of possible depth values from initial depth values when computing missing depth values

// energy minimization parameters for depth labeling
const double GLOBAL_LABELING_ENERGY_DISPARITY_THRESHOLD = 0.02;
const double GLOBAL_LABELING_ENERGY_COL_THRESHOLD = 30.; // scalar noise parameter for data likelihood
const double GLOBAL_LABELING_ENERGY_OCCLUSION_CONSTANT = 0.01; // scalar occlusion cost
const int GLOBAL_LABELING_ENERGY_LAMBDA_L = 9; // scalar smoothness prior weight for cliques crossing segmentation boundaries
const int GLOBAL_LABELING_ENERGY_LAMBDA_H = 108; // scalar smoothness prior weight for cliques not crossing segmentation boundaries

// mean-shift over-segmentation parameters for depth labeling
const int GLOBAL_MEAN_SHIFT_SEGMENTATION_SIGMAS = 4;
const int GLOBAL_MEAN_SHIFT_SEGMENTATION_SIGMAR = 5.;
const int GLOBAL_MEAN_SHIFT_SEGMENTATION_MINREGION = 0;


const enum GLOBAL_PROPOSAL_METHOD { SAME_UNI = 1, SEG_PLN = 2, SMOOTH_STAR = 3 }; // stereo optimization proposal methods; SAME_UNI means random front-parallel, SEG_PLAN means prototypical segment-based stereo proposals, SMOOTH_STAR means smooth*

#endif