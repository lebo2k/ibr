#include "Math3d.h"

/*
Plane
Normal vector: n = [a0,b0,c0]
Goes through point P0= (x0,y0,z0)
Equation for plane: a0*(x-x0) + b0*(y-y0) + c0*(z-z0) = 0
Line
Direction [a1,b1,c1]
Goes through point P1=(x1,y1,z1)
Parametric equation for line in 3D space given depth t:
x = x1 + a1 * t
y = y1 + b1 * t
z = z1 + c1 * t
Find point on line where intersects plane by finding t.  Substituting:
a0*(x1 + a1*t - x0) + b0*(y1 + b1*t - y0) + c0*(z1 + c1*t - z0) = 0
t*(a0*a1 + b0*b1 + c0*c1) + a0*x1 - a0*x0 + b0*y1 - b0*y0 + c0*z1 - c0*z0 = 0
t = (a0*x0 + b0*y0 + c0*z0 - a0*x1 - b0*y1 - c0*z1) / (a0*a1 + b0*b1 + c0*c1)

returns true if intersect in a single point; false if don't intersect (parallel) or intersect in multiple points (line lies on plane)
*/
bool Math3d::IntersectionLinePlane(Point3d norm_plane, Point3d p0_plane, Point3d dir_line, Point3d p0_line, Point3d &intersection)
{
	double denom = norm_plane.ddot(p0_line);
	if (denom == 0) return false; // don't intersect (parallel) or intersect in multiple points (line lies on plane)
	double num = norm_plane.ddot(p0_plane) - norm_plane.ddot(p0_line);
	//double t = (norm_plane.x*p0_plane.x + norm_plane.y*p0_plane.y + norm_plane.z*p0_plane.z - norm_plane.x*p0_line.x - norm_plane.y*p0_line.y - norm_plane.z*p0_line.z) / (norm_plane.x*dir_line.x + norm_plane.y*dir_line.y + norm_plane.z*dir_line.z);
	double t = num / denom;
	intersection.x = p0_line.x + dir_line.x * t;
	intersection.y = p0_line.y + dir_line.y * t;
	intersection.z = p0_line.z + dir_line.z * t;
	return true;
}

/*
calculates centroid of a vector of Point3d and alters arg num_points to contain the total number
*/
Point3d Math3d::Centroid3d(std::map<long, Point3d> pts, int &num_points)
{
	Point3d centroid(0, 0, 0);
	int num = 0;
	for (std::map<long, Point3d>::iterator it = pts.begin(); it != pts.end(); ++it)
	{
		centroid += (*it).second;
		num++;
	}
	centroid.x = centroid.x / (float)num;
	centroid.y = centroid.y / (float)num;
	centroid.z = centroid.z / (float)num;

	num_points = num;

	return centroid;
}

/*
1. Find centroid of points
2. Subtract centroid from each point
3. Form a 3xN matrix X out of resulting coordinates
4. Calculate the SVD of X
5. The normal vector of the best-fit plane is the left singular vector corresponding to the least singular value of X

returns true if successful, false if unsuccessful

Will this work in all cases?  More reliable solution offered by eigen library (http://eigen.tuxfamily.org/index.php?title=Main_Page), for which only header inclusion is necessary.  But is SVD possible for all inputs regardless of methodology?
*/
bool Math3d::BestFitPlane(std::map<long, Point3d> pts, Point3d &p0, Point3d &norm)
{

	if (pts.size() < 3) return false;

	int num_points;
	p0 = Centroid3d(pts, num_points); // 1. Find centroid of points

	Eigen::MatrixXd X(3, num_points);

	int num_point = 0;
	for (std::map<long, Point3d>::iterator it = pts.begin(); it != pts.end(); ++it)
	{
		Point3d pt = (*it).second - p0; // 2. Subtract centroid from each point

		// 3. Form a 3xN matrix X out of resulting coordinates
		X(0, num_point) = pt.x;
		X(1, num_point) = pt.y;
		X(2, num_point) = pt.z;

		num_point++;
	}

	// 4. Calculate the SVD of X
	Eigen::JacobiSVD<Eigen::MatrixXd> svd;
	svd.compute(X, Eigen::ComputeFullU);
	Eigen::MatrixXd U = svd.matrixU();
	Eigen::Vector3d svs = svd.singularValues();

	bool success = true;
	for (int r = 0; r < U.rows(); r++)
	{
		for (int c = 0; c < U.cols(); c++)
		{
			double val = U(r, c);
			if (val != val) success = false;
		}
	}
	if (!success) return false;

	// 5. The normal vector of the best-fit plane is the left singular vector corresponding to the least singular value of X (if SVD of M=UWVt where Vt is transpose of V, then columns of U contains left singular vectors and diagonal of W contains the singular values, which are the square roots of the non-zero eigenvalues of both MMt and MtM).  Singular values from eigen's svd are sorted in decreasing order.
	norm.x = U(0, U.cols() - 1);
	norm.y = U(1, U.cols() - 1);
	norm.z = U(2, U.cols() - 1);
	normalize(norm);

	return true;
}

/*
1. Find centroid of points
2. Subtract centroid from each point
3. Form a 3xN matrix X out of resulting coordinates
4. Calculate the SVD of X
5. The direction vector of the best-fit line is the left singular vector corresponding to the largest singular value of X

Will this work in all cases?  More reliable solution offered by eigen library (http://eigen.tuxfamily.org/index.php?title=Main_Page), for which only header inclusion is necessary.  But is SVD possible for all inputs regardless of methodology?
*/
bool Math3d::BestFitLine(std::map<long, Point3d> pts, Point3d &p0, Point3d &dir)
{
	if (pts.size() < 2) return false;

	int num_points;
	p0 = Centroid3d(pts, num_points); // 1. Find centroid of points

	Eigen::MatrixXd X(3, num_points);

	int num_point = 0;
	for (std::map<long, Point3d>::iterator it = pts.begin(); it != pts.end(); ++it)
	{
		Point3d pt = (*it).second - p0; // 2. Subtract centroid from each point

		// 3. Form a 3xN matrix X out of resulting coordinates
		X(0, num_point) = pt.x;
		X(1, num_point) = pt.y;
		X(2, num_point) = pt.z;

		num_point++;
	}

	// 4. Calculate the SVD of X
	Eigen::JacobiSVD<Eigen::MatrixXd> svd;
	svd.compute(X, Eigen::ComputeFullU);
	Eigen::MatrixXd U = svd.matrixU();
	Eigen::Vector3d svs = svd.singularValues();

	bool success = true;
	for (int r = 0; r < U.rows(); r++)
	{
		for (int c = 0; c < U.cols(); c++)
		{
			double val = U(r, c);
			if (val != val) success = false;
		}
	}
	if (!success) return false;

	// 5. The normal vector of the best-fit line is the left singular vector corresponding to the largest singular value of X (if SVD of M=UWVt where Vt is transpose of V, then columns of U contains left singular vectors and diagonal of W contains the singular values, which are the square roots of the non-zero eigenvalues of both MMt and MtM).  Singular values from eigen's svd are sorted in decreasing order.
	dir.x = U(0, 0);
	dir.y = U(1, 0);
	dir.z = U(2, 0);
	normalize(dir);

	return true;
}

/*
projects points onto a plane
*/
std::map<long, Point3d> Math3d::ProjPointsPlane(std::map<long, Point3d> pts, Point3d p0_plane, Point3d norm_plane)
{
	std::map<long, Point3d> ptsProj;
	for (std::map<long, Point3d>::iterator it = pts.begin(); it != pts.end(); ++it)
	{
		long id = (*it).first;
		Point3d v = (*it).second - p0_plane; // 2.a. Make a vector from your orig point to the point of interest
		double dist = v.ddot(norm_plane); // 2.b. Take the dot product of that vector with the normal vector n. (dist = scalar distance from point to plane along the normal)
		Point3d projPt = (*it).second - dist * norm_plane; // 2.c. Multiply the normal vector by the distance, and subtract that vector from your point
		ptsProj[id] = projPt;
	}
	return ptsProj;
}

/*
projects point onto a plane
*/
Point3d Math3d::ProjPointPlane(Point3d pt, Point3d p0_plane, Point3d norm_plane)
{
	Point3d v = pt - p0_plane; // 2.a. Make a vector from your orig point to the point of interest
	double dist = v.ddot(norm_plane); // 2.b. Take the dot product of that vector with the normal vector n. (dist = scalar distance from point to plane along the normal)
	Point3d projPt = pt - dist * norm_plane; // 2.c. Multiply the normal vector by the distance, and subtract that vector from your point
	return projPt;
}

/*
If line A (through point Ap0 and in direction Adir) intersects line B (through point Bp0 and in direction Bdir) at a single point, returns true and alters arg intersectPt to be the point of intersection; else returns false
If lines are skew, intersectPt is still updated and skew is set to true even though function returns false; intersectPt will be set using parametric equation from line A of inputs.
The arg tolerance_dist sets the maximum distance (exclusive) between the two lines wherein they will still be considered to intersect.  The argument defaults to CHECK_ISZERO_TOLERANCE to accomodate rounding errors.

The two lines intersect if and only if there is a solution s,t to the system of linear equations:
Ap0.x + t * Adir.x = Bp0.x + s * Bdir.x
Ap0.y + t * Adir.y = Bp0.y + s * Bdir.y
Ap0.z + t * Adir.z = Bp0.z + s * Bdir.z

Adir.x * t - Bdir.x * s = Bp0.x - Ap0.x
Adir.y * t - Bdir.y * s = Bp0.y - Ap0.y
Adir.z * t - Bdir.z * s = Bp0.z - Ap0.z

[ Adir.x -Bdir.x		[ t			[ Bp0.x - Ap0.x
Adir.y -Bdir.y	*	  s ]	=	  Bp0.y - Ap0.y
Adir.z -Bdir.z ]					  Bp0.z - Ap0.z ]


or compute parametrically


Line1                         Line2
-----                         -----
x = x1 + a1 * t1              x = x2 + a2 * t2
y = y1 + b1 * t1              y = y2 + b2 * t2
z = z1 + c1 * t1              z = z2 + c2 * t2

If we set the two x values equal, and the two y values equal we get
these two equations.

x1 + a1 * t1 = x2 + a2 * t2
y1 + b1 * t1 = y2 + b2 * t2
*/
bool Math3d::IntersectionLines(Point3d Ap0, Point3d Adir, Point3d Bp0, Point3d Bdir, Point3d &intersectPt, bool &skew, double tolerance_dist)
{
	skew = false;
	if (veclength(Adir.cross(Bdir)) == 0) return false; // parallel lines
	if (veclength(Adir) == 0) return false; // line A has no direction
	if (veclength(Bdir) == 0) return false; // line B has no direction

	double t, s;
	s = (Ap0.y - Bp0.y + Adir.y*(Bp0.x - Ap0.x) / Adir.x) / (Bdir.y - (Adir.y*Bdir.x / Adir.x));
	t = (Bp0.x - Ap0.x + Bdir.x * s) / Adir.x;

	intersectPt.x = Ap0.x + t*Adir.x;
	intersectPt.y = Ap0.y + t*Adir.y;
	intersectPt.z = Ap0.z + t*Adir.z;

	double checkZ = Bp0.z + s*Bdir.z;

	if (abs(intersectPt.z - checkZ) > tolerance_dist)
	{
		skew = true;
		return false; // lines are skew
	}
	else return true;

	/*
	// matrix approach was crashing on certain inputs, so switched to parametric substitution approach
	Mat A = Mat(3, 2, CV_64F, Scalar(0));
	Mat x = Mat(2, 1, CV_64F, Scalar(0));
	Mat B = Mat(3, 1, CV_64F, Scalar(0));

	A.at<double>(0, 0) = Adir.x;
	A.at<double>(1, 0) = Adir.y;
	A.at<double>(2, 0) = Adir.z;
	A.at<double>(0, 1) = -Bdir.x;
	A.at<double>(1, 1) = -Bdir.y;
	A.at<double>(2, 1) = -Bdir.z;

	B.at<double>(0, 0) = Bp0.x - Ap0.x;
	B.at<double>(1, 0) = Bp0.y - Ap0.y;
	B.at<double>(2, 0) = Bp0.z - Ap0.z;

	solve(A, B, x, DECOMP_SVD); // using DECOMP_SVD, the system can be over-defined, as ours is here with 2 variables in 3 equations

	double t = x.at<double>(0, 0);
	double s = x.at<double>(1, 0);

	Point3d iPt;
	intersectPt.x = Ap0.x + t * Adir.x;
	intersectPt.y = Ap0.y + t * Adir.y;
	intersectPt.z = Ap0.z + t * Adir.z;

	return true;
	*/
}

/*
Computes the respective barycentric coordinates of x wrt the triangle with vertices p0, p1, and p2
If x is inside the triangle or on an edge, all barycentric coordinates will be >=0

Vn = (p2 - p1) x (p1 - p0)
A = ||Vn||
n = Vn / A;
u = [(p2 - p1) x (x - p1)] . n / A
v = [(p0 - p2) x (x - p2)] . n / A
w = 1 - u - v
*/
Point3d Math3d::ComputeBarycentric(Point2d x, Point2d p0, Point2d p1, Point2d p2)
{
	Point2d v0 = p2 - p0;
	Point2d v1 = p1 - p0;
	Point2d v2 = x - p0;

	double dot00 = v0.ddot(v0);
	double dot01 = v0.ddot(v1);
	double dot02 = v0.ddot(v2);
	double dot11 = v1.ddot(v1);
	double dot12 = v1.ddot(v2);

	// Compute barycentric coordinates
	double u, v, w;
	if ((dot00 * dot11 - dot01 * dot01) == 0.0)
	{
		u = 1.0;
		v = 0.0;
		w = 0.0;
	}
	else
	{
		double invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
		w = (dot11 * dot02 - dot01 * dot12) * invDenom;
		v = (dot00 * dot12 - dot01 * dot02) * invDenom;
		u = 1.0f - v - w;
	}

	Point3d bary(u, v, w);
	return bary;
}

/*
Computes the respective barycentric coordinates of x wrt the triangle with vertices p0, p1, and p2
If x is inside the triangle or on an edge, all barycentric coordinates will be >=0

Vn = (p2 - p1) x (p1 - p0)
A = ||Vn||
n = Vn / A;
u = [(p2 - p1) x (x - p1)] . n / A
v = [(p0 - p2) x (x - p2)] . n / A
w = 1 - u - v
*/
Point3d Math3d::ComputeBarycentric(Point3d x, Point3d p0, Point3d p1, Point3d p2)
{
	Point3d v0 = p2 - p0;
	Point3d v1 = p1 - p0;
	Point3d v2 = x - p0;

	double dot00 = v0.ddot(v0);
	double dot01 = v0.ddot(v1);
	double dot02 = v0.ddot(v2);
	double dot11 = v1.ddot(v1);
	double dot12 = v1.ddot(v2);

	// Compute barycentric coordinates
	double u, v, w;
	if ((dot00 * dot11 - dot01 * dot01) == 0.0)
	{
		w = 1.0;
		v = 0.0;
		u = 0.0;
	}
	else
	{
		double invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
		u = (dot11 * dot02 - dot01 * dot12) * invDenom;
		v = (dot00 * dot12 - dot01 * dot02) * invDenom;
		w = 1 - u - v;
	}

	Point3d bary(w, v, u);
	return bary;
}

/*
requires arg points to be in counter-clockwise order
returns true if x is inside triangle p0,p1,p2 and puts results in bary; otherwise returns false
*/
bool Math3d::SetBarycentricTriangle(Point2d p0, Point2d p1, Point2d p2, Point2d x, Point3d &bary)
{
	// if ControlPoint is not on plane defined by p0, p1, and p2, then cannot qualify as being inside triangle face (and may screw up barycentric coordinate computation to make it seem like it is, so get these cases out of the way first); B lies on the plane through A and with normal P if and only if dotProduct(A - B, P) = 0.  Computation below assumes counter-clockwise order for points p0, p1, and p2.

	if ((p0 == p1) ||
		(p0 == p2) ||
		(p1 == p2))
		return false; // if don't catch this case, will create problems in the calculations that cause undeserving arguments to produce a response of true

	bary = Math3d::ComputeBarycentric(x, p0, p1, p2); // must be counter-clockwise

	// barycentric inside check should take into account rounding errors by not checking barycentric percentages directly, but rather R3 coordinates against a tolerance.  Otherwise, if the face is very small or narrow, we could be very close physically to the face but off by a not insignificant percentage.  Use the area of the face.
	if ((bary.x < -CHECK_ISZERO_TOLERANCE) ||
		(bary.y < -CHECK_ISZERO_TOLERANCE) ||
		(bary.z < -CHECK_ISZERO_TOLERANCE) ||
		((bary.x + bary.y + bary.z - 1) > CHECK_BARYCENTRIC_TOLERANCE)) // should all be >=0 and sum to 1 within tolerance
		return false;

	bool confirm = CheckBarycentric(p0, p1, p2, x, bary); // applying barycentric coordinates to vertices of face f should result in coordsR3 within tolerance
	if (!confirm) return false; // can occur if some iso points included in the file are not incident on faces of the surface; if enough are despite some not being so, the algorithm will still work

	return true;
}

/*
requires arg points to be in counter-clockwise order
returns true if x is inside triangle p0,p1,p2 and puts results in bary; otherwise returns false
*/
bool Math3d::SetBarycentricTriangle(Point3d p0, Point3d p1, Point3d p2, Point3d x, Point3d &bary)
{
	// if ControlPoint is not on plane defined by p0, p1, and p2, then cannot qualify as being inside triangle face (and may screw up barycentric coordinate computation to make it seem like it is, so get these cases out of the way first); B lies on the plane through A and with normal P if and only if dotProduct(A - B, P) = 0.  Computation below assumes counter-clockwise order for points p0, p1, and p2.

	if ((p0 == p1) ||
		(p0 == p2) ||
		(p1 == p2))
		return false; // if don't catch this case, will create problems in the calculations that cause undeserving arguments to produce a response of true

	Point3d vec1 = p1 - p0;
	Point3d vec2 = p2 - p0;
	Point3d normP = vec1.cross(vec2);
	normalize(normP);
	//Point3d AmB = p0 - x;
	//double testP = AmB.ddot(normP);
	//if (abs(testP) > CHECK_POINTONPLANE_TOLERANCE) return false;
	if (!Math3d::IsPointOnPlane(x, p0, normP)) return false;

	//double areaT = AreaTriangle(p0, p1, p2);

	bary = Math3d::ComputeBarycentric(x, p0, p1, p2); // must be counter-clockwise

	// barycentric inside check should take into account rounding errors by not checking barycentric percentages directly, but rather R3 coordinates against a tolerance.  Otherwise, if the face is very small or narrow, we could be very close physically to the face but off by a not insignificant percentage.  Use the area of the face.
	if ((bary.x < -CHECK_BARYCENTRIC_ISZERO_TOLERANCE) ||
		(bary.y < -CHECK_BARYCENTRIC_ISZERO_TOLERANCE) ||
		(bary.z < -CHECK_BARYCENTRIC_ISZERO_TOLERANCE) ||
		((bary.x + bary.y + bary.z - 1) > CHECK_BARYCENTRIC_TOLERANCE)) // should all be >=0 and sum to 1 within tolerance
		return false;

	bool confirm = CheckBarycentric(p0, p1, p2, x, bary); // applying barycentric coordinates to vertices of face f should result in coordsR3 within tolerance
	if (!confirm) return false; // can occur if some iso points included in the file are not incident on faces of the surface; if enough are despite some not being so, the algorithm will still work

	return true;
}

double Math3d::AreaTriangle(Point3d p1, Point3d p2, Point3d p3)
{
	Point3d v1 = p3 - p1;
	Point3d v2 = p3 - p2;
	Point3d vc = v1.cross(v2);
	double mag = veclength(vc);
	return 0.5*mag;
}

double Math3d::AreaPolygon(std::map<long, Point3d> pts)
{
	if (pts.size() < 3) return 0.0;

	Point3d plane_p0, plane_norm;
	FindPlane(pts[0], pts[1], pts[2], plane_p0, plane_norm); // requires p0, p1, p2 in counter-clockwise order

	Point3d plast, pcurr, pcross, sum;
	plast = pts[pts.rbegin()->first];
	for (std::map<long, Point3d>::iterator it = pts.begin(); it != pts.end(); ++it)
	{
		pcurr = (*it).second;
		pcross = plast.cross(pcurr);
		sum += pcross;
		plast = pcurr;
	}
	double result = sum.ddot(plane_norm);
	return abs(result / 2.0);
}

/*
returns true if applying barycentric coordinates to vertices of triangle p0,p1,p2 result in x, false otherwise
*/
bool Math3d::CheckBarycentric(Point3d p0, Point3d p1, Point3d p2, Point3d x, Point3d baryCoords)
{
	Point3d bc = (p0 * baryCoords.x) + (p1 * baryCoords.y) + (p2 * baryCoords.z);
	Point3d diff = bc - x;
	double diffMag = veclength(diff);
	if (diffMag > CHECK_BARYCENTRIC_TOLERANCE) return false;
	else return true;
}

/*
returns true if applying barycentric coordinates to vertices of triangle p0,p1,p2 result in x, false otherwise
*/
bool Math3d::CheckBarycentric(Point2d p0, Point2d p1, Point2d p2, Point2d x, Point3d baryCoords)
{
	Point2d bc = (p0 * baryCoords.x) + (p1 * baryCoords.y) + (p2 * baryCoords.z);
	Point2d diff = bc - x;
	double diffMag = veclength(diff);
	if (diffMag > CHECK_BARYCENTRIC_TOLERANCE) return false;
	else return true;
}

double Math3d::DistancePoints(Point3d p1, Point3d p2)
{
	double l = sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2) + pow(p2.z - p1.z, 2));
	return l;
}

/*
	requires p0, p1, p2 in counter-clockwise order
*/
void Math3d::FindPlane(Point3d p0, Point3d p1, Point3d p2, Point3d &plane_p0, Point3d &plane_norm)
{
	Point3d v1 = p1 - p0;
	Point3d v2 = p2 - p0;
	plane_norm = v1.cross(v2);
	normalize(plane_norm);

	std::map<long, Point3d> pts;
	pts[0] = p0;
	pts[1] = p1;
	pts[2] = p2;
	int num_points;
	plane_p0 = Centroid3d(pts, num_points);
}

/*
	returns true if point is on plane, false otherwise
*/
bool Math3d::IsPointOnPlane(Point3d pt, Point3d plane_p0, Point3d plane_norm)
{
	Point3d tmp = plane_p0 - pt;
	double dp = tmp.ddot(plane_norm);
	if (abs(dp) > CHECK_POINTONPLANE_TOLERANCE) return false;
	else return true;
}

/*
	finds angle from v1 to v2 in either counterclockwise or clockwise direction
*/
double Math3d::FindOrderedAngleDegrees(Point2d v1, Point2d v2, bool counterclockwise)
{
	/*
	double len1 = veclength(v1);
	double len2 = veclength(v2);
	double cos_theta_rads = v1.ddot(v2) / (len1 * len2);
	double theta_rads = acos(cos_theta_rads);
	double theta_degrees = theta_rads * 180 / CV_PI;
	return theta_degrees;
	*/
	
	Point2d pt1 = v1;
	Point2d pt2(0.0, 0.0);
	Point2d pt3 = v2;
	double angle1 = atan2(pt2.y - pt1.y, pt2.x - pt1.x) * 180.0 / CV_PI;
	double angle2 = atan2(pt2.y - pt3.y, pt2.x - pt3.x) * 180.0 / CV_PI;
	double angle = angle1 - angle2;
	if (angle < 0) angle += 360;
	else if (angle>360) angle -= 360;
	if (counterclockwise) angle = 360 - angle;
	return angle;
	
}

/*
	poly is list of vertices of convex polygonal face in counter-clockwise order (each is a pair of ID and R3 coordinates); indices must be incremental from 0 to size-1
	if intersects at infinite points along edge of polygon, returns only the farthest vertex along the intersecting edge and returns the vertex's OTHER edge (not the actual edge of intersection) as the edge of intersection (does it this was so can continue in same direction along surface in further steps of algorithms that use this function)
	returns true if intersected, false otherwise
	excludes edge with vertices excludeEdgeVid1 and excludeEdgeVid2 from the possible intersection edges if excludeEdge is true; ignore them if excludeEdge is false
*/
bool Math3d::IntersectionRayFace(bool ignoreSkew, Point3d ray_p0, Point3d ray_dir, std::map<int, std::pair<long, Point3d>> poly, bool excludeEdge, long excludeEdgeVid1, long excludeEdgeVid2, Point3d &intersectPt, long &intersectEdgeVert1, long &intersectEdgeVert2)
{
	bool intersect = false;
	if (veclength(ray_dir) == 0) return false; // ray has no direction

	int currPolyIdx;
	long currVid;
	Point3d currVert, edgeDir;

	/*
	// if ray_p0 is on a vertex of poly, exclude any edge that includes that vertex since can't ensure we're traveling in the right direction given ray_dir before intersection
	bool excludeVert = false;
	long excludeVid = -1;
	for (std::map<int, std::pair<long, Point3d>>::iterator it = poly.begin(); it != poly.end(); ++it)
	{
		currPolyIdx = (*it).first;
		currVid = (*it).second.first;
		currVert = (*it).second.second;

		if (vecdist(ray_p0, currVert) < CHECK_R3ZERODIST_TOLERANCE)
		{
			excludeVert = true;
			excludeVid = currVid;
		}
	}
	*/

	// for each edge of poly, find whether the ray intersects it; must be in along positive ray direction from ray_p0 and inside edge's endpoints; if find an intersection point, skip the rest
	int lastPolyIdx = poly.rbegin()->first;
	long lastVid = poly[lastPolyIdx].first;
	Point3d lastVert = poly[lastPolyIdx].second;
	for (std::map<int, std::pair<long, Point3d>>::iterator it = poly.begin(); it != poly.end(); ++it)
	{
		currPolyIdx = (*it).first;
		currVid = (*it).second.first;
		currVert = (*it).second.second;

		/*
		if (excludeVert) // if ray_p0 is on a vertex of poly, exclude any edge that includes that vertex since can't ensure we're traveling in the right direction given ray_dir before intersection
		{
			if ((currVid == excludeVid) ||
				(lastVid == excludeVid))
			{
				lastPolyIdx = currPolyIdx;
				lastVid = currVid;
				lastVert = currVert;
				continue;
			}
		}
		*/

		if (excludeEdge) // excludes edge with vertices excludeEdgeVert1 and excludeEdgeVert2 from the possible intersection edges
		{
			if (((currVid == excludeEdgeVid1) && (lastVid == excludeEdgeVid2)) ||
				((currVid == excludeEdgeVid2) && (lastVid == excludeEdgeVid1)))
			{
				lastPolyIdx = currPolyIdx;
				lastVid = currVid;
				lastVert = currVert;
				continue;
			}
		}

		edgeDir = currVert - lastVert;
		normalize(edgeDir);
		if (veclength(edgeDir) == 0)
		{
			lastPolyIdx = currPolyIdx;
			lastVid = currVid;
			lastVert = currVert;
			continue;
		}

		// first, check whether ray and edge are parallel and, if so, whether are colinear; if parallel and also colinear, then return the far vertex of the edge along ray_dir; if parallel and not colinear, there is no intersection of the ray with this edge
		if (veclength(ray_dir.cross(edgeDir)) < CHECK_ISZERO_TOLERANCE) // parallel
		{
			if (PointsColinear(ray_p0, lastVert, currVert)) // colinear (since we know the ray and edge are parallel, no need to also test a second point on the ray); if parallel and also colinear, then return the far vertex of the edge along ray_dir; also, set intersectEdgeVerts as being the vertex's OTHER edge (not the actual edge of intersection); do it this was so can continue in same direction along surface in further steps of algorithms that use this function
			{
				double dist1 = vecdist(lastVert, ray_p0);
				double dist2 = vecdist(currVert, ray_p0);
				bool lastIsForward = PointForwardRay(lastVert, ray_p0, ray_dir);
				bool currIsForward = PointForwardRay(currVert, ray_p0, ray_dir);
				if ((dist1 > dist2) && // lastVert farther from ray_p0
					(lastIsForward)) // lastVert is on the ray (is in the correct direction from ray_p0)
				{
					intersectPt = lastVert;
					int lastlastPolyIdx;
					if (lastPolyIdx == 0) lastlastPolyIdx = poly.size() - 1;
					else lastlastPolyIdx = lastPolyIdx - 1;
					intersectEdgeVert1 = poly[lastlastPolyIdx].first;
					intersectEdgeVert2 = poly[lastPolyIdx].first;
					return true;
				}
				else if (currIsForward) // currVert farther from ray_p0, or distances are equal; and currVert is on the ray (is in the correct direction from ray_p0)
				{
					intersectPt = currVert;
					int nextPolyIdx;
					if (currPolyIdx == (poly.size() - 1)) nextPolyIdx = 0;
					else nextPolyIdx = currPolyIdx + 1;
					intersectEdgeVert1 = poly[currPolyIdx].first;
					intersectEdgeVert2 = poly[nextPolyIdx].first;
					return true;
				}
				else
				{
					lastPolyIdx = currPolyIdx;
					lastVid = currVid;
					lastVert = currVert;
					continue;
				}
			}
			else
			{
				lastPolyIdx = currPolyIdx;
				lastVid = currVid;
				lastVert = currVert;
				continue; // if parallel and not colinear, there is no intersection of the ray with this edge
			}
		}
		
		// next, run line-to-line intersection and check bounds (intersection is within edge segment and along forward direction of the ray)
		bool skew;
		bool intersects = IntersectionLines(lastVert, edgeDir, ray_p0, ray_dir, intersectPt, skew);
		if ((intersects) ||
			(ignoreSkew && skew))
		{
			if ((IntersectionPointLineSegment(intersectPt, lastVert, currVert)) &&
				(PointForwardRay(intersectPt, ray_p0, ray_dir)))
			{
				intersectEdgeVert1 = lastVid;
				intersectEdgeVert2 = currVid;
				return true;
			}
		}

		lastPolyIdx = currPolyIdx;
		lastVid = currVid;
		lastVert = currVert;
	}
	return false;
}

/*
	returns true if p lies on line segment between p1 and p2 in R3, false otherwise

	if P1 and P2 are vectors giving the endpoint coordinates of the line segment
	and P the coordinates of an arbitrary point, then P lies within the line segment
	whenever

	(norm(cross(P - P1, P2 - P1)) < tol) & ...
	(dot(P - P1, P2 - P1) >= 0) & (dot(P - P2, P2 - P1) <= 0)

	is true, where 'tol' is just large enough to allow for worst case round - off error
	in performing the 'cross', 'dot', and 'norm' operations.

	where 'norm' is the vector length
*/
bool Math3d::IntersectionPointLineSegment(Point3d p, Point3d p1, Point3d p2)
{
	Point3d pp1 = p - p1;
	Point3d pp2 = p - p2;
	Point3d p2p1 = p2 - p1;

	Point3d c = pp1.cross(p2p1);
	double normc = veclength(c);
	double d1 = pp1.ddot(p2p1);
	double d2 = pp2.ddot(p2p1);

	if ((normc < CHECK_ISZERO_TOLERANCE) &&
		(d1 >= 0) &&
		(d2 <= 0)) return true;
	else return false;
}

/*
	returns true if points are colinear, false otherwise
*/
bool Math3d::PointsColinear(Point3d p1, Point3d p2, Point3d p3)
{
	Point3d n1 = p2 - p1;
	Point3d n2 = p3 - p1;

	double area = 0.5 * veclength(n1.cross(n2));
	if (area < CHECK_ISZERO_TOLERANCE) return true;
	else return false;
}

/*
	returns true if p is along the ray in the ray_dir direction from ray_p0, false otherwise

	parametric eqn for a line in 3d:
	x = x1 + a1 * t
	y = y1 + b1 * t
	z = z1 + c1 * t
*/
bool Math3d::PointForwardRay(Point3d p, Point3d ray_p0, Point3d ray_dir)
{
	if (!PointsColinear(p, ray_p0, (ray_p0 + ray_dir))) return false;

	double t = (p.x - ray_p0.x) / ray_dir.x;
	if (t > 0) return true;
	else return false;
}

double Math3d::AngleBetweenVectorsRads(Point3d v1, Point3d v2)
{
	double dot = v1.ddot(v2);
	double mag1 = veclength(v1);
	double mag2 = veclength(v2);
	return acos(dot / (mag1*mag2));
}

/*
	given the known location of 3 points (p1, p2, p3) and the distance of the current unknown location from each (d1, d2, d3), finds the current location
	requires that one of the points (p1, p2, p3) is not colinear with the other two
*/
Point2d Math3d::Trilateration2D(Point2d p1, double d1, Point2d p2, double d2, Point2d p3, double d3)
{
	double i1 = p1.x, i2 = p2.x, i3 = p3.x;
	double j1 = p1.y, j2 = p2.y, j3 = p3.y;
	double x, y;
	
	x = (((2 * j3 - 2 * j2)*((d1*d1 - d2*d2) + (i2*i2 - i1*i1) + (j2*j2 - j1*j1)) - (2 * j2 - 2 * j1)*((d2*d2 - d3*d3) + (i3*i3 - i2*i2) + (j3*j3 - j2*j2))) /
		((2 * i2 - 2 * i3)*(2 * j2 - 2 * j1) - (2 * i1 - 2 * i2)*(2 * j3 - 2 * j2)));
	y = ((d1*d1 - d2*d2) + (i2*i2 - i1*i1) + (j2*j2 - j1*j1) + x*(2 * i1 - 2 * i2)) / (2 * j2 - 2 * j1);

	return Point2d(x, y);
}

/*
	given a camera position in world space, the screen space point to cast through, and its inverse projection matrix, returns the world space direction of the ray cast from the camera through the pixel
*/
Point3d Math3d::CastRayThroughPixel(Point3d cam_pos, Point ss_pt, Matrix<double, 4, 3> Pinv) {
	Matrix<double, 3, 1> pt;
	pt(0, 0) = ss_pt.x;
	pt(1, 0) = ss_pt.y;
	pt(2, 0) = 1;

	// transform SS to WS
	Matrix<double, 4, 1> ws_pt = Pinv * pt;
	// normalize
	double h = ws_pt(3, 0);
	ws_pt(0, 0) = ws_pt(0, 0) / h;
	ws_pt(1, 0) = ws_pt(1, 0) / h;
	ws_pt(2, 0) = ws_pt(2, 0) / h;

	Point3d ray_dir;
	ray_dir.x = ws_pt(0, 0) - cam_pos.x;
	ray_dir.y = ws_pt(1, 0) - cam_pos.y;
	ray_dir.z = ws_pt(2, 0) - cam_pos.z;
	
	return ray_dir;
}

// P1inv is the pseudo-inverse (since not square) projection matrix for camera 1, P2 the projection matrix for camera 2, e2 the epipole in the second image
// returns the fundmental matrix for camera 1 related to camera 2
// F1 = [e2]x * P2 * P1inv = [e2]x * Hpi
// [e2]x means a skew-symmetric matrix based on 3-vector e2 = [a1,a2,a3]T where [e2]x = [0 -a3 a2; a3 0 -a1; -a2 a1 0]
// F1*X = e2 x P2*P1inv*X = e2 x Hpi * X where X is any 3x1 point on the epiline in image 2
// e2 = P2*cam1_pos = the projection of the first camera center into the second camera screen space = epipole
// note that F2 = F1.transpose()
// e2 is 3x1
// P2*P1inv = 3x3; F1 and F2 are 3x3
// geometrically, F1 represents a mapping from pixels in SS1 to the pencil of epipolar lines through the epipole e2 in SS2
// vector x matrix : I think means vector cross each row of the matrix
Matrix3f Math3d::ComputeFundamentalMatrix(Point3d cam1_pos, Matrix<float, 4, 3> P1inv, Matrix<float, 3, 4> P2) {
	Matrix<float, 4, 1> C;
	C(0, 0) = static_cast<float>(cam1_pos.x);
	C(1, 0) = static_cast<float>(cam1_pos.y);
	C(2, 0) = static_cast<float>(cam1_pos.z);
	C(3, 0) = 1.;
	Matrix<float, 3, 1> e2 = P2 * C;
	float h;
	Point3f e2p;
	h = e2(2, 0);
	e2p.x = e2(0, 0);// / h;
	e2p.y = e2(1, 0);// / h;
	e2p.z = e2(2, 0);// 1.;

	Matrix3f e2x;
	e2x.setZero();
	e2x(0, 1) = -1 * e2p.z;
	e2x(0, 2) = e2p.y;
	e2x(1, 0) = e2p.z;
	e2x(1, 2) = -1 * e2p.x;
	e2x(2, 0) = -1 * e2p.y;
	e2x(2, 1) = e2p.x;

	Matrix3f Hpi = P2 * P1inv;

	Matrix3f F = e2x * Hpi;
	
	return F;
}