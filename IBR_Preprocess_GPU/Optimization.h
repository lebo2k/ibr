#ifndef Optimization_H
#define Optimization_H

#include "Globals.h"
#include "Interpolation.h"
#include "EigenMatlab.h"
#include "Segmentation.h"

// Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>

// 3rd party libraries
#include "igl/sortrows.h"
#include "qpbo/QPBO.h"

using namespace std;
using namespace Eigen;

/*
notes
A|B (A and B same size mxn) ==> mxn C for which each coeff is the OR of coefficients in associated positions in A and B
min(A) (A is mxn) ==> 1xn C where each coefficient is the minimum in the associated column of A
ojw_bsxfun(@eq, A, B) == bsxfun(@eq, A, B) where A and B are both mxn ==> mxn C that is true if A(r,c) == B(r,c) and false otherwise
A >= 0 where A is mxn ==> mxn C with boolean 1 if A(r,c)>=0, and 0 otherwise
*/

// objective function optimization
class Optimization {

private:
	
	Matrix<unsigned int, 3, Dynamic> SEI_; // 3xM uint32 array of smoothness clique indices; additional explanation: 3xi (i==(3*(rows)*(cols-2)+3*(rows-2)*(cols)) of output image) matrix of pixel location indices (including separate color channels in RGB order) displaying across its rows connectivity of the output image (first vertical connectivity, then horizonal connectivity); colors are num_pixels apart; form necessary for 2nd order smoothness prior; also described as: {3,4}xQ uint32 matrix, each column containing the following information on a triple clique : [node1 node2 node3 [triple_clique_energy_table_index]].If there are only 3 rows then triple_clique_energy_table_index is assumed to be the column number, therefore you must have Q == R.
	Matrix<unsigned int, 3, Dynamic> SEI_used_; // same indices as SEI_, but transformed to compact used pixel index space
	Matrix<int, Dynamic, 1> EW_; // data structure of smoothness edges that don't cross segmentation boundaries ("Edge Weighting"); assigns higher energy values to neighboring pixels in same segments and lower energy values to neighboring pixels in different segments so that attempts to smooth proposals across segmentation boundaries are less likely to succeed during optimization because would increase the energy; # rows = 8 * SEI_.cols() where the value is repeated 8 times for multiplication against each triple-clique energy value in eSmooth ... note that SEI_ and SEI_used_ have the same number of columns, but different actual indices since SEI_ uses full index representations whereas SEI_used_ uses compact used index representations; 8 is the multiple because when taking 2nd derivative approximations for SE, there are 8 combinations of 2 proposal values (Dcurr and Dnew) over 3 nodes (p, q, and r)
	Matrix<float, Dynamic, Dynamic> D_segpln_; // disparity label map structure generated by SegmentPlanar(); size num_pixels_[cid_out_] x b; disparity are given in quantized label form
	Matrix<double, Dynamic, 4> WC_partial_; // partially-filled WC for use in Optimization::FuseProposals() for each image in turn assume it's the reference image; third column will be filled with disparities at run time within each optimization iteration; only uses pixels that are not masked out and for which a disparity is not known with high confidence
	Matrix<unsigned int, 2, Dynamic> EI_partial_; // partially-filled EI for use in Optimization::FuseProposals() for each image in turn assume it's the reference image
	Matrix<int, 4, Dynamic> E_partial_; // partially filled E for use in Optimization::FuseProposals() for each image in turn assume it's the reference image


	void InitEW(); // initializes data structure EW_, which identifies smoothness edges that don't cross segmentation boundaries
	void InitEW_new(); // initializes data structure EW_, which identifies smoothness edges that don't cross segmentation boundaries; boundaries are assigned by either significant disparity differences between neighboring pixels, or significant slope differences of the disparities among neighboring pixels
	void InitSEI(); // initializes data structure SEI_
	void InitPairwiseInputs(); // pre-computes some data for WC, EI, and E data structures for use in Optimization::FuseProposals() for each image in turn assume it's the reference image; third column will be filled with disparities at run time within each optimization iteration; only uses pixels that are not masked out and for which a disparity is not known with high confidence
	void ComputeSE(const Eigen::Matrix<double, Dynamic, 1> *Dcurr, const Eigen::Matrix<double, Dynamic, 1> *Dnew, double scale_factor, Matrix<int, 8, Dynamic> *SE); // computes SE (updating arg SE), an 8xR triple clique energy table, each column containing the energies[E000 E001 E010 E011 E100 E101 E110 E111] for a given triple clique.
	Matrix<unsigned int, 2, Dynamic> Optimization::MapEIUsedToFull(Matrix<unsigned int, 2, Dynamic> *EI); // the indices in EI are used pixel indices, but for each input image, the second row's index (each image comprises 2*num_used_pixels_out_ columns) is shifted by (num_used_pixels_out_ + (num_used_pixels_out_ * 2 * img_num)) where img_num is the 0-indexed index of the image we're on in order of those already added to EI; therefore, indices must first be transformed back to the range [0, num_used_pixels_out_), then mapped to full pixel indices, then transformed back so that each image in the second row (each image still comprises 2*num_used_pixels_out_ columns) is shifted by (num_pixels_out_ + (num_pixels_out_ * 2 * img_num)); to make matters more complicated, each image's pixels indices are each in two parts: one set of num_used_pixels_out_ columns corresponding to Dcurr and another num_used_pixels_out_ columns corresponding to Dnew, and the Dnew columns are shifted by an additional num_used_pixels_out_ columns pixels; so basically, the second row starts at num_used_pixels_out_, and after every num_used_pixels_out_ columns num_used_pixels_out_ is added to the indices in the next set

	void UpdatePhotoconsistencyVisibility(int cid_in, int img_num, map<int, int> label_mappings, Matrix<double, Dynamic, 4> *WC, Matrix<double, Dynamic, 3> *T, Matrix<int, 4, Dynamic> *E); // photoconsistency costs are based on the color error if the pixel is visible, and are occl_cost_ if the pixel is not visible; so update E to take into account label segmentation and mapping so that if a pixel projects to an unassigned label in the input image (and if another label is assigned for that input image as the mapping, meaning we have some info for it to compare against), then it's considered not visible and that's reflected in photoconsistency costs in E by setting it to occl_cost_

	static void erfunc(char *err) { cerr << err << endl; }

	void FindInteractions(const Eigen::Matrix<double, Dynamic, 3> *V, double dist, Eigen::Matrix<unsigned int, 2, Dynamic> *P); // given a set of 3d image coordinates(pixel coordinates plus depth), ordered such that the x coordinates are monotonically increasing, as are the y coordinates within each block of identical x coordinates, finds any occluding / occluded pairs in this set.; assumes V is a compact used pixel representation according to mask, having skipped pixel indices for mask values of false; will return indices for a full pixel image, not a compact representation

	// our matrices are already 0-indexed, so don't subtract one from indices and bounds as ojw does in several places
	// iter is the 0-indexed number of the current iteration
	template<class INTERNAL>
	inline void QPBO_wrapper_func(Matrix<int, 2, Dynamic> *UE_mat, Eigen::Matrix<unsigned int, 2, Dynamic> *PI_mat, Matrix<int, 4, Dynamic> *PE_mat, Matrix<unsigned int, 3, Dynamic> *TI_mat, Matrix<int, 8, Dynamic> *TE_mat, Matrix<int, Dynamic, 1> *Lout, int iter, INTERNAL infinite_edge_cost, bool include_smoothness_terms = true) {
		assert(PI_mat->cols() == PE_mat->cols());
		assert(TI_mat->cols() == TE_mat->cols());

		bool debug = false;
		bool debug_tmp = false;
		bool timing = true;
		double t;

		if (debug) cout << "Optimization::QPBO_wrapper_func()" << endl;

		int Pindices = PI_mat->rows();
		int nP = PI_mat->cols();
		int nPE = PE_mat->cols();
		int nU = UE_mat->cols();
		const int32_t *U = (const int32_t *)UE_mat->data();
		const int32_t *PE = (const int32_t *)PE_mat->data();
		const int32_t *TE = (const int32_t *)TE_mat->data();
		const uint32_t *PI = (const uint32_t *)PI_mat->data();
		const uint32_t *TI = (const uint32_t *)TI_mat->data();
		int Tindices = TI_mat->rows();
		int nT = TI_mat->cols();
		int nTE = TE_mat->cols();

		// options
		int firstNNodes = Lout->rows(); // only return labels of first N nodes
		int improveMethod = 0; // method to improve unknown nodes: 0 - none, 1 - improve (assume 0 best), 2 - optimal splice
		int ContractIters = 0; // number of contract cycles to try

		if (debug_tmp) cout << "nU+nT " << (nU + nT) << ", nP + 6*nT " << (nP + 6 * nT) << endl;

		// Define the graph
		QPBO<INTERNAL> *graph = new QPBO<INTERNAL>(nU + nT, nP + 6 * nT, erfunc); // each triple-clique generates an extra node (so 4 total) and ends up with 6 edges

		// Create graph nodes
		int start_node = graph->AddNode(nU);
		code_assert(start_node == 0);

		// Enter unary terms
		for (int nodeInd = 0; nodeInd < nU; nodeInd++) {
			// Add unary potetial
			graph->AddUnaryTerm(nodeInd, (INTERNAL)U[0], (INTERNAL)U[1]);
			U += 2;
		}

		if (debug_tmp) cout << "unary terms added" << endl;

		// Enter pairwise terms
		int countIrregular = 0;
		const int32_t *INPUT_ptr = PE - 4;
		const uint32_t *UL_end = &PI[Pindices*nP];
		for (int i = 0; PI < UL_end; PI += Pindices) {

			// Calculate index of energy table
			if (Pindices == 2)
				INPUT_ptr += 4;
			else {
				if (PI[2] < 0 || PI[2] >= nPE)
					cerr << "column of PI references invalid column of PE" << endl;
				INPUT_ptr = &PE[4 * PI[2]];
			}

			// Check submodularity 
			//if (PE_ptr[0] + PE_ptr[3] > PE_ptr[1] + PE_ptr[2])
			//	countIrregular++;

			graph->AddPairwiseTerm(PI[0], PI[1], (INTERNAL)INPUT_ptr[0], (INTERNAL)INPUT_ptr[1], (INTERNAL)INPUT_ptr[2], (INTERNAL)INPUT_ptr[3]);
		}

		if (debug_tmp) cout << "pairwise terms added" << endl;

		// Enter triple clique terms
		int nNodes = nU;
		if (include_smoothness_terms) {
			if (nT) {
				INPUT_ptr = TE - 8;
				UL_end = &TI[Tindices*nT];
				for (; TI < UL_end; TI += Tindices) {

					// Calculate index of energy table
					if (Tindices == 3)
						INPUT_ptr += 8;
					else {
						if (TI[3] < 0 || TI[3] >= nTE)
							cerr << "Column of TI references invalid column of TE" << endl;
						INPUT_ptr = &TE[8 * TI[3]];
					}

					int node1 = TI[0];
					int node2 = TI[1];
					int	node3 = TI[2];

					INTERNAL A = (INTERNAL)INPUT_ptr[0]; // E000
					INTERNAL B = (INTERNAL)INPUT_ptr[1]; // E001
					INTERNAL C = (INTERNAL)INPUT_ptr[2]; // E010
					INTERNAL D = (INTERNAL)INPUT_ptr[3]; // E011
					INTERNAL E = (INTERNAL)INPUT_ptr[4]; // E100
					INTERNAL F = (INTERNAL)INPUT_ptr[5]; // E101
					INTERNAL G = (INTERNAL)INPUT_ptr[6]; // E110
					INTERNAL H = (INTERNAL)INPUT_ptr[7]; // E111

					INTERNAL pi = (A + D + F + G) - (B + C + E + H);

					if (pi >= 0) {
						graph->AddPairwiseTerm(node1, node2, 0, C - A, 0, G - E);
						//if (C-A < G-E)
						//	countIrregular++;
						graph->AddPairwiseTerm(node1, node3, 0, 0, E - A, F - B);
						//if (E-A < F-B)
						//	countIrregular++;
						graph->AddPairwiseTerm(node2, node3, 0, B - A, 0, D - C);
						//if (B-A < D-C)
						//	countIrregular++;

						if (pi > 0) {
							// Add node
							int node4 = graph->AddNode();
							graph->AddUnaryTerm(node4, A, A - pi);
							graph->AddPairwiseTerm(node1, node4, 0, pi, 0, 0);
							graph->AddPairwiseTerm(node2, node4, 0, pi, 0, 0);
							graph->AddPairwiseTerm(node3, node4, 0, pi, 0, 0);
							nNodes++;
						}
					}
					else {
						graph->AddPairwiseTerm(node1, node2, B - D, 0, F - H, 0);
						//if (F-H < B-D)
						//	countIrregular++;
						graph->AddPairwiseTerm(node1, node3, C - G, D - H, 0, 0);
						//if (D-H < C-G)
						//	countIrregular++;
						graph->AddPairwiseTerm(node2, node3, E - F, 0, G - H, 0);
						//if (G-H < E-F)
						//	countIrregular++;

						// Add node
						int node4 = graph->AddNode();
						graph->AddUnaryTerm(node4, H + pi, H);
						graph->AddPairwiseTerm(node1, node4, 0, 0, -pi, 0);
						graph->AddPairwiseTerm(node2, node4, 0, 0, -pi, 0);
						graph->AddPairwiseTerm(node3, node4, 0, 0, -pi, 0);
						nNodes++;
					}
				}
				// Merge edges
				graph->MergeParallelEdges();
			}
		}

		if (debug_tmp) cout << "triple clique terms added" << endl;

		if (timing) t = (double)getTickCount();
	
		// Solve for optimimum
		graph->Solve();

		if ((debug) &&
			(timing)) {
			t = (double)getTickCount() - t;
			cout << "Optimization::QPBO_wrapper_func() Solve running time = " << t*1000. / getTickFrequency() << " ms" << endl;
			t = (double)getTickCount();
		}

		// Label a few more nodes if there are several global minima, and clump unlabelled nodes into groups
		graph->ComputeWeakPersistencies();

		/*
		if (timing) {
			t = (double)getTickCount() - t;
			cout << "Optimization::QPBO_wrapper_func() ComputeWeakPersistencies running time = " << t*1000. / getTickFrequency() << " ms" << endl;
		}
		*/

		// create the output array
		int32_t *labelOutPtr = (int32_t *)Lout->data();

		// Read out labelling
		int countUnlabel = 0;
		int *listUnlabel = new int[firstNNodes];
		for (int nodeCount = 0; nodeCount < firstNNodes; nodeCount++) {
			labelOutPtr[nodeCount] = (int32_t)graph->GetLabel(nodeCount);
			if (labelOutPtr[nodeCount] < 0)
				listUnlabel[countUnlabel++] = nodeCount;
		}

		int countRegions = 0;
		int countUnlabelAfterProbe = countUnlabel;
		if (countUnlabel) {
			// Fix up stage
			// Initialise mapping for probe
			int *mapping = new int[nNodes];
			for (int i = 0; i < nNodes; i++)
				mapping[i] = i * 2;

			// Clump unlabelled nodes into independent regions
			int *regionMap = new int[countUnlabelAfterProbe];
			for (int nodeCount = 0; nodeCount < countUnlabelAfterProbe; nodeCount++) {
				int regionId = graph->GetRegion(mapping[listUnlabel[nodeCount]] / 2);
				// Compress the labelling to consecutive integers
				int region;
				for (region = 0; region < countRegions; region++) {
					if (regionMap[region] == regionId)
						goto skip_point;
				}
				regionMap[countRegions] = regionId;
				countRegions++;
			skip_point:
				labelOutPtr[listUnlabel[nodeCount]] = -1 - (int32_t)region;
			}
			delete regionMap;
			delete mapping;
		}
		delete listUnlabel;
		delete graph;

		// save statistics
		sd_->count_unlabelled_vals_(0, iter) = (int32_t)countUnlabel;
		sd_->count_unlabelled_regions_(0, iter) = (int32_t)countRegions;
		sd_->count_unlabelled_after_QPBOP_(0, iter) = (int32_t)countUnlabelAfterProbe;

		if (debug) cout << "unlabeled: " << countUnlabel << ", Non-sub " << countIrregular << endl;
	}

public:

	StereoData *sd_;
	int cid_out_; // ID of reference (output) camera

	// energy parameters
	float occl_val_; // scalar penalty energy cost for occluded pixels
	double scale_factor_;
	int occl_cost_;
	int Kinf_; // smaller value seems to avoid errors in QPBO
	double energy_val;
	int num_in_; // number of used input images excluding the reference(output) image - see ojw_stereo.m for where vals.I is set to exclude the reference image
	int num_pixels_out_; // total number of pixels in reference image, regardless of whether disparities are known or pixel is masked
	int num_used_pixels_out_; // number of pixels in the reference image that are not masked out
	int num_unknown_pixels_out_; // number of pixels in the reference image for which depth is unknown (excludes both masked-out pixels and pixels for which depth is known to a high degree of confidence)
	int oobv_; // used in an interpolation call below, so must be type int

	// Constructors and destructor
	Optimization();
	~Optimization();

	// Initialization
	void Init(StereoData *sd, int cid_out); // requires reference image chosen and data set for sd

	// Fuse proposals
	void FuseProposals(const Eigen::Matrix<double, Dynamic, 1> *Dcurr, const Eigen::Matrix<double, Dynamic, 1> *Dnew, Eigen::Matrix<bool, Dynamic, 1> *Dswap, const int iter, Eigen::Matrix<double, Dynamic, 1> *energy, map<int, map<int, int>> label_mappings, bool include_smoothness_terms = true); // updates Dswap to booleans denoting whether the value in Dcurr should be replaced by the value in Dnew; updates the value in energy in the row given by iter to the energy of the fused disparity map; updates timings with additional timing information for the iteration iter (iter indexes the column of timings)
	inline float FuseProposals_ePhoto(float f) { return log(2) - log(exp(pow(f, 2.) * (-1. / (GLOBAL_LABELING_ENERGY_COL_THRESHOLD*3.))) + 1); }; // compute scalar ePhoto for fuse depths; vals.ephoto = @(F)log(2) - log(exp(sum(F . ^ 2, 2)*(-1 / (col_thresh*3))) + 1); // function handle to data cost energy function
	void FuseProposals_ePhoto(Eigen::Matrix<double, Dynamic, 3> *F, Eigen::Matrix<double, Dynamic, 1> *X, Eigen::Matrix<double, Dynamic, 3> *scratch); // compute matrix ePhoto for fuse depths
	void FuseProposals_eSmooth(Matrix<double, Dynamic, Dynamic> *F); // vals.esmooth = @(F) EW .* min(abs(F), options.disp_thresh); since options.smoothness_kernel == 1 for truncated linear kernel; operates on F in-place

	// QPBO
	void QPBO_CalcVisEnergy(Matrix<bool, Dynamic, 1> *L, Matrix<int, 2, Dynamic> *U, Matrix<int, 4, Dynamic> *E, Matrix<unsigned int, 2, Dynamic> *EI, Matrix<int, 8, Dynamic> *SE, Matrix<unsigned int, 3, Dynamic> *SEI, Matrix<int, 1, Dynamic> *TE, Matrix<unsigned int, 2, Dynamic> *TEI, int num_in, Matrix<int, Dynamic, 1> *Uout, Matrix<int, Dynamic, 1> *Eout, Matrix<int, Dynamic, 1> *SEout, Matrix<bool, Dynamic, Dynamic> *Vout); // updates Mout, num_regions, Uout, Eout, SEout, Vout
	void QPBO_ChooseLabels(Matrix<int, Dynamic, 1> *M, Matrix<int, 2, Dynamic> *U, Matrix<int, 4, Dynamic> *E, Matrix<unsigned int, 2, Dynamic> *EI, Matrix<int, 8, Dynamic> *SE, Matrix<unsigned int, 3, Dynamic> *SEI, Matrix<int, 1, Dynamic> *TE, Matrix<unsigned int, 2, Dynamic> *TEI, int num_in, Matrix<int, Dynamic, 1> *Mout, int &num_regions, Matrix<int, Dynamic, 1> *Uout, Matrix<int, Dynamic, 1> *Eout, Matrix<int, Dynamic, 1> *SEout, Matrix<bool, Dynamic, Dynamic> *Vout); // updates Uout, Eout, SEout, Vout
	void QPBO_eval(Matrix<int, 2, Dynamic> *UE, Eigen::Matrix<unsigned int, 2, Dynamic> *PI, Matrix<int, 4, Dynamic> *PE, Matrix<unsigned int, 3, Dynamic> *TI, Matrix<int, 8, Dynamic> *TE, Matrix<int, Dynamic, 1> *Lout, int iter, bool include_smoothness_terms = true);
	void QPBO_CompressGraph(Matrix<int, 2, Dynamic> *U, Matrix<int, 4, Dynamic> *E, Matrix<unsigned int, 2, Dynamic> *EI, Matrix<int, 4, Dynamic> *TE, Matrix<unsigned int, 2, Dynamic> *TEI, int num_in, Matrix<unsigned int, 2, Dynamic> *EI_);

	// Display
	void InitOutputFigures();
	void CloseOutputFigures();
	void UpdateOutputFigures(const Eigen::Matrix<double, Dynamic, 1> *Dcurr, const Eigen::Matrix<double, Dynamic, 1> *Dnew, Eigen::Matrix<bool, Dynamic, 1> *Dswap, Matrix<bool, Dynamic, Dynamic> *Visibilities, Matrix<unsigned int, 2, Dynamic> *EI, Matrix<int, Dynamic, 1> *Uout, Matrix<int, Dynamic, 1> *Eout, Matrix<int, Dynamic, 1> *SEout, Matrix<bool, Dynamic, Dynamic> *Vout);
	
};

#endif