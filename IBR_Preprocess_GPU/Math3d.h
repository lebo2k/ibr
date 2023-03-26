#ifndef MATH3D_H
#define MATH3D_H

#include "HelperFunctions.h"

// assumptions
const double CHECK_BARYCENTRIC_TOLERANCE = 0.02; // 0.001
const double CHECK_POINTONPLANE_TOLERANCE = 0.0001;
const double CHECK_ISZERO_TOLERANCE = 0.0001; // 0.000001
const double CHECK_BARYCENTRIC_ISZERO_TOLERANCE = 0.01;
const double CHECK_R3ZERODIST_TOLERANCE = 0.0001; // anything within a tenth of a millimeter of a point is assumed to be in the same place because of rounding error

class Math3d
{

private:

public:

	static inline double DistPointPlane(Point3d pt, Point3d plane_norm, double plane_dist_origin) { return plane_norm.ddot(pt) + plane_dist_origin; }; // returns distance from the point to the plane; plane_norm is the normalized normal for the plane and plane_dist_origin is the distance from the plane to the origin; here, the plane is given in Hessian Normal Form, as described in http://mathworld.wolfram.com/HessianNormalForm.html; if have plane equation as ax+by+cz+d=0, then plane_dist_origin = d / sqrt(a^2 + b^2 + c^2)
	static Point3d Centroid3d(std::map<long, Point3d> pts, int &num_points); // calculates centroid of a vector of Point3d and alters arg num_points to contain the total number
	static bool BestFitPlane(std::map<long, Point3d> pts, Point3d &p0, Point3d &norm);
	static bool BestFitLine(std::map<long, Point3d> pts, Point3d &p0, Point3d &dir);
	static std::map<long, Point3d> ProjPointsPlane(std::map<long, Point3d> pts, Point3d p0_plane, Point3d norm_plane); // projects points onto a plane
	static Point3d ProjPointPlane(Point3d pt, Point3d p0_plane, Point3d norm_plane); // projects point onto a plane
	static bool IntersectionLines(Point3d Ap0, Point3d Adir, Point3d Bp0, Point3d Bdir, Point3d &intersectPt, bool &skew, double tolerance_dist = CHECK_ISZERO_TOLERANCE); // If line A (through point Ap0 and in direction Adir) intersects line B (through point Bp0 and in direction Bdir) at a single point, returns true and alters arg intersectPt to be the point of intersection; else returns false
	static bool IntersectionLinePlane(Point3d norm_plane, Point3d p0_plane, Point3d dir_line, Point3d p0_line, Point3d &intersection); // returns true if intersect in a single point; false if don't intersect (parallel) or intersect in multiple points (line lies on plane)
	static Point3d ComputeBarycentric(Point2d x, Point2d p0, Point2d p1, Point2d p2); // Computes the respective barycentric coordinates of x wrt the triangle with vertices p0, p1, and p2
	static Point3d ComputeBarycentric(Point3d x, Point3d p0, Point3d p1, Point3d p2); // Computes the respective barycentric coordinates of x wrt the triangle with vertices p0, p1, and p2
	static bool SetBarycentricTriangle(Point2d p0, Point2d p1, Point2d p2, Point2d x, Point3d &bary); // requires arg points to be in counter-clockwise order; returns true if x is inside triangle p0,p1,p2 and puts results in bary; otherwise returns false
	static bool SetBarycentricTriangle(Point3d p0, Point3d p1, Point3d p2, Point3d x, Point3d &bary); // requires arg points to be in counter-clockwise order; returns true if x is inside triangle p0,p1,p2 and puts results in bary; otherwise returns false
	static bool CheckBarycentric(Point2d p0, Point2d p1, Point2d p2, Point2d x, Point3d baryCoords); // returns true if applying barycentric coordinates to vertices of triangle p0,p1,p2 result in x, false otherwise
	static bool CheckBarycentric(Point3d p0, Point3d p1, Point3d p2, Point3d x, Point3d baryCoords); // returns true if applying barycentric coordinates to vertices of triangle p0,p1,p2 result in x, false otherwise
	static double DistancePoints(Point3d p0, Point3d p1); // returns the distance between the two points
	static void FindPlane(Point3d p0, Point3d p1, Point3d p2, Point3d &plane_p0, Point3d &plane_norm); // requires p0, p1, p2 in counter-clockwise order
	static bool IsPointOnPlane(Point3d pt, Point3d plane_p0, Point3d plane_norm); // returns true if point is on plane, false otherwise
	static bool IntersectionRayFace(bool ignoreSkew, Point3d ray_p0, Point3d ray_dir, std::map<int, std::pair<long, Point3d>> poly, bool excludeEdge, long excludeEdgeVid1, long excludeEdgeVid2, Point3d &intersectPt, long &intersectEdgeVert1, long &intersectEdgeVert2); // poly is list of vertices of convex polygonal face in counter-clockwise order (each is a pair of ID and R3 coordinates)
	static bool IntersectionPointLineSegment(Point3d p, Point3d p1, Point3d p2); // returns true if p lies on line segment between p1 and p2 in R3, false otherwise
	static bool PointsColinear(Point3d p1, Point3d p2, Point3d p3); // returns true if points are colinear, false otherwise
	static bool PointForwardRay(Point3d p, Point3d ray_p0, Point3d ray_dir); // returns true if p is along the ray in the ray_dir direction from ray_p0, false otherwise
	static double AreaPolygon(std::map<long, Point3d> pts);
	static double AreaTriangle(Point3d p1, Point3d p2, Point3d p3);
	static double AngleBetweenVectorsRads(Point3d v1, Point3d v2);
	static double FindOrderedAngleDegrees(Point2d v1, Point2d v2, bool counterclockwise); // finds angle from v1 to v2 in either counterclockwise or clockwise direction
	static Point2d Trilateration2D(Point2d p1, double d1, Point2d p2, double d2, Point2d p3, double d3); // given the known location of 3 points (p1, p2, p3) and the distance of the current unknown location from each (d1, d2, d3), finds the current location; requires that one of the points (p1, p2, p3) is not colinear with the other two
	static Point3d CastRayThroughPixel(Point3d cam_pos, Point ss_pt, Matrix<double, 4, 3> Pinv); // given a camera position in world space, the screen space point to cast through, and its inverse projection matrix, returns the world space direction of the ray cast from the camera through the pixel
	static Matrix3f ComputeFundamentalMatrix(Point3d cam1_pos, Matrix<float, 4, 3> P1inv, Matrix<float, 3, 4> P2); // P1inv is the pseudo-inverse (since not square) projection matrix for camera 1, P2 the projection matrix for camera 2, e2 the epipole in the second image; returns the fundmental matrix for camera 1 related to camera 2; F1 = e2 x P2*P1inv where e2 = P2*cam1_pos = the projection of the first camera center into the second camera screen space = epipole; note that F2 = F1.transpose(); e2 is 3x1; P2*P1inv = 3x3; F1 and F2 are 3x3; Geometrically, F1 represents a mapping from pixels in SS1 to the pencil of epipolar lines through the epipole e2 in SS2

	/*
	// cvComputeCorrespondEpilines
	The function ComputeCorrespondEpilines computes the corresponding epiline for every input point using the basic equation of epipolar line geometry:

	If points located on first image (ImageID=1), corresponding epipolar line can be computed as:

	l2=F*p1
	where F is fundamental matrix, p1 point on first image, l2 corresponding epipolar line on second image.
	If points located on second image (ImageID=2):
	l1=FT*p2
	where F is fundamental matrix, p2 point on second image, l1 corresponding epipolar line on first image

	Each epipolar line is present by coefficients a,b,c of line equation:
	a*x + b*y + c = 0
	Also computed line normalized by a2+b2=1. It's useful if distance from point to line must be computed later.
	*/

};

#endif