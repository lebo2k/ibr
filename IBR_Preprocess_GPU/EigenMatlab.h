#ifndef EigenMatlab_H
#define EigenMatlab_H

// Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>

// 3rd party libraries
#include "igl/sortrows.h"

using namespace std;
using namespace Eigen;

/*
	notes
		A|B (A and B same size mxn) ==> mxn C for which each coeff is the OR of coefficients in associated positions in A and B
		min(A) (A is mxn) ==> 1xn C where each coefficient is the minimum in the associated column of A
		ojw_bsxfun(@eq, A, B) == bsxfun(@eq, A, B) where A and B are both mxn ==> mxn C that is true if A(r,c) == B(r,c) and false otherwise
		A >= 0 where A is mxn ==> mxn C with boolean 1 if A(r,c)>=0, and 0 otherwise
*/

// functions to replicate Matlab commands in Eigen
class EigenMatlab {

private:

	// updates src by adding additional_cols new columns of zeros on the right, while preserving existing data
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static inline void ExtendHorizontally(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *src, unsigned int additional_cols) {
		if (src->cols() > 0) { // extend src horizontally while preserving its data
			Matrix<_Tp, Dynamic, Dynamic> tmp(src->rows(), src->cols());
			tmp = (*src);
			src->resize(src->rows(), src->cols() + additional_cols);
			src->block(0, 0, src->rows(), tmp.cols()) = tmp;
			src->block(0, tmp.cols(), src->rows(), additional_cols).setZero();
		}
		else {
			src->resize(src->rows(), additional_cols); // src has no data yet
			src->setZero();
		}
	}

	// updates src by adding additional_rows new columns of zeros on the bottom, while preserving existing data
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static inline void ExtendVertically(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *src, unsigned int additional_rows) {
		if (src->rows() > 0) { // extend src vertically while preserving its data
			Matrix<_Tp, Dynamic, Dynamic> tmp(src->rows(), src->cols());
			tmp = (*src);
			src->resize(src->rows() + additional_rows, src->cols());
			src->block(0, 0, tmp.rows(), src->cols()) = tmp;
			src->block(tmp.rows(), 0, additional_rows, src->cols()).setZero();
		}
		else {
			src->resize(additional_rows, src->cols()); // src has no data yet
			src->setZero();
		}
	}

public:

	// A = [A B]
	// updates A by concatenating B to the right of it
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, int _cols2, int _maxCols2, int _options2>
	static inline void ConcatHorizontally(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mdest, const Matrix<_Tp, _rows, _cols2, _options2, _maxRows, _maxCols2> *Maddl) {
		assert(Mdest->rows() == Maddl->rows());

		int cols_orig = Mdest->cols();
		ExtendHorizontally(Mdest, Maddl->cols());
		Mdest->block(0, cols_orig, Mdest->rows(), Maddl->cols()) = (*Maddl);
	}

	// A = [A; B]
	// updates A by concatenating B below it
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, int _rows2, int maxRows2, int _options2>
	static inline void ConcatVertically(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mdest, const Matrix<_Tp, _rows2, _cols, _options2, maxRows2, _maxCols> *Maddl) {
		assert(Mdest->cols() == Maddl->cols());

		int rows_orig = Mdest->rows();
		ExtendVertically(Mdest, Maddl->rows());
		Mdest->block(rows_orig, 0, Maddl->rows(), Mdest->cols()) = (*Maddl);
	}

	// Mvals(:, Mindices) = []
	// returns matrix Mret of size Mvals.rows() x (Mvals.cols() - Mindices->rows()*Mindices.cols()) where each position's associated index in Mindices is used to look up the column in Mvals to be truncated
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, typename _Tp2, int _rows2, int _cols2, int _options2, int _maxRows2, int _maxCols2>
	static inline Matrix<_Tp, _rows, Dynamic> TruncateByIndicesColumns(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mvals, const Matrix<_Tp2, _rows2, _cols2, _options2, _maxRows2, _maxCols2> *Mindices) {

		Matrix<bool, 1, Dynamic> Mbools(1, Mvals->cols());
		Mbools.setConstant(true);

		_Tp2 idx;
		const _Tp2 *mi = Mindices->data();
		int size = Mindices->cols()*Mindices->rows();
		for (int j = 0; j < size; j++) {
			idx = *mi++;
			Mbools(idx) = false;
		}

		Matrix<_Tp, _rows, Dynamic> Mret = TruncateByBooleansColumns(Mvals, &Mbools);

		return Mret;
	}

	// Mvals(:, Mindices)
	// returns matrix Mret of size Mvals.rows() x Mindices->rows()*Mindices.cols() where each position's associated index in Mindices is used to look up the value for Mret from Mvals
	// Mindices indexes into the columns of Mvals and returns all associated rows, where columns are strung out in the column-major order of Mindices, so there are Mvals.rows() rows and Mindices.rows()
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, typename _Tp2, int _rows2, int _cols2, int _options2, int _maxRows2, int _maxCols2>
	static inline Matrix<_Tp, _rows, Dynamic> AccessByIndicesColumns(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mvals, const Matrix<_Tp2, _rows2, _cols2, _options2, _maxRows2, _maxCols2> *Mindices) {

		int nrows = Mvals->rows();
		int ncols = Mindices->rows() * Mindices->cols();
		Matrix<_Tp, _rows, Dynamic> Mret(nrows, ncols);

		_Tp2 idx;
		const _Tp2 *mi = Mindices->data();
		int size = Mindices->cols()*Mindices->rows();
		int i = 0;
		for (int j = 0; j < size; j++) {
			idx = *mi++;
			Mret.col(i) = Mvals->col(idx);
			i++;
		}

		return Mret;
	}

	// Mvals(Mindices, :)
	// returns matrix Mret of size Mindices.rows()*Mindices->cols() x Mvals.cols() where each position's associated index in Mindices is used to look up the value for Mret from Mvals
	// Mindices indexes into the rows of Mvals and returns all associated columns, where rows are strung out in the column-major order of Mindices, so there are Mvals.rows() columns and Mindices.rows()*Mindices*cols() rows
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, typename _Tp2, int _rows2, int _cols2, int _options2, int _maxRows2, int _maxCols2>
	static inline Matrix<_Tp, Dynamic, _cols> AccessByIndicesRows(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mvals, const Matrix<_Tp2, _rows2, _cols2, _options2, _maxRows2, _maxCols2> *Mindices) {

		Matrix<_Tp, Dynamic, _cols> Mret(Mindices->rows()*Mindices->cols(), Mvals->cols());

		_Tp2 idx;
		const _Tp2 *mi = Mindices->data();
		int size = Mindices->cols()*Mindices->rows();
		int i = 0;
		for (int j = 0; j < size; j++) {
			idx = *mi++;
			Mret.row(i) = Mvals->row(idx);
			i++;
		}

		return Mret;
	}

	// Mvals(Mindices)
	// returns matrix Mret where each position's associated index in Mindices is used to look up the value for Mret from Mvals
	// Mret's size is governed by Mindices; however, note that Matlab doesn't always work this way - for example, if Mindices is a row and Mvals is a column, Mret according to Matlab would be columnar
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, typename _Tp2, int _rows2, int _cols2, int _options2, int _maxRows2, int _maxCols2>
	static inline Matrix<_Tp, _rows2, _cols2> AccessByIndices(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mvals, const Matrix<_Tp2, _rows2, _cols2, _options2, _maxRows2, _maxCols2> *Mindices) {
		//assert((Mindices->maxCoeff() < Mvals->rows() * Mvals->cols()) && (Mindices->minCoeff() >= 0), "EigenMatlab::AccessByIndices() Mindices coefficients must be within index ranges for Mvals"); // correct assertion but too costly in terms of computation time

		Matrix<_Tp, _rows2, _cols2> Mret(Mindices->rows(), Mindices->cols());
		_Tp *mv = Mvals->data();
		const _Tp2 *mi = Mindices->data();
		_Tp *mr = Mret.data();
		_Tp2 idx;
		int size = Mindices->cols()*Mindices->rows();
		for (int i = 0; i < size; i++) {
			idx = *mi++;
			*mr++ = mv[idx];
		}

		return Mret;
	}

	// Mdest(Mindices) = Mvals
	// updates Mdest as follows: for each coefficient in Mindices, use it as an index into Mdest for which the value in Mvals associated with the position in Mindices is substituted in the indexed position of Mdest
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, typename _Tp2, int _rows2, int _cols2, int _options2, int _maxRows2, int _maxCols2, int _rows3, int _cols3, int _options3, int _maxRows3, int _maxCols3>
	static inline void AssignByIndices(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mdest, const Matrix<_Tp2, _rows2, _cols2, _options2, _maxRows2, _maxCols2> *Mindices, const Matrix<_Tp, _rows3, _cols3, _options3, _maxRows3, _maxCols3> *Mvals) {
		assert(Mvals->rows()*Mvals->cols() == Mindices->rows()*Mindices->cols());
		//assert(Mindices->minCoeff() >= 0 && Mindices->maxCoeff() < (Mdest->rows()*Mdest->cols()), "AssignByIndices() indices in Mindices must references positions in Mdest"); // correct assertion but too costly in terms of computation time

		_Tp *md = Mdest->data();
		const _Tp2  *mi = Mindices->data();
		const _Tp *mv = Mvals->data();
		_Tp2 idx;
		int size = Mindices->cols()*Mindices->rows();
		for (int i = 0; i < size; i++) {
			idx = *mi++;
			md[idx] = *mv++;
		}
	}

	// Mdest(Mindices, :) = Mvals
	// updates Mdest as follows: for each coefficient in Mindices, use it as an index into Mdest for which the value in Mvals associated with the position in Mindices is substituted in the indexed position of Mdest
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, typename _Tp2, int _rows2, int _cols2, int _options2, int _maxRows2, int _maxCols2, int _rows3, int _options3, int _maxRows3>
	static inline void AssignByIndicesRows(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mdest, const Matrix<_Tp2, _rows2, _cols2, _options2, _maxRows2, _maxCols2> *Mindices, const Matrix<_Tp, _rows3, _cols, _options3, _maxRows3, _maxCols> *Mvals) {
		assert(Mvals->rows() == Mindices->rows()*Mindices->cols());
		assert(Mdest->cols() == Mvals->cols());
		//assert(Mindices->minCoeff() >= 0 && Mindices->maxCoeff() < (Mdest->rows()*Mdest->cols()), "AssignByIndices() indices in Mindices must references positions in Mdest"); // correct assertion but too costly in terms of computation time

		const _Tp2  *mi = Mindices->data();
		_Tp2 idx;
		int size = Mindices->cols()*Mindices->rows();
		for (int i = 0; i < size; i++) {
			idx = *mi++;
			Mdest->row(idx) = Mvals->row(i);
		}
	}

	// Mdest(:, Mindices) = Mvals
	// updates Mdest as follows: for each coefficient in Mindices, use it as an index into Mdest for which the value in Mvals associated with the position in Mindices is substituted in the indexed position of Mdest
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, typename _Tp2, int _rows2, int _cols2, int _options2, int _maxRows2, int _maxCols2, int _cols3, int _options3, int _maxCols3>
	static inline void AssignByIndicesCols(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mdest, const Matrix<_Tp2, _rows2, _cols2, _options2, _maxRows2, _maxCols2> *Mindices, const Matrix<_Tp, _rows, _cols3, _options3, _maxRows, _maxCols3> *Mvals) {
		assert(Mvals->cols() == Mindices->rows()*Mindices->cols());
		assert(Mdest->rows() == Mvals->rows());
		//assert(Mindices->minCoeff() >= 0 && Mindices->maxCoeff() < (Mdest->rows()*Mdest->cols()), "AssignByIndices() indices in Mindices must references positions in Mdest"); // correct assertion but too costly in terms of computation time

		const _Tp2  *mi = Mindices->data();
		_Tp2 idx;
		int size = Mindices->cols()*Mindices->rows();
		for (int i = 0; i < size; i++) {
			idx = *mi++;
			Mdest->col(idx) = Mvals->col(i);
		}
	}

	// Mdest(Mindices) = val
	// updates Mdest as follows: for each coefficient in Mindices, use it as an index into Mdest for which the value val is substituted in the indexed position of Mdest
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, typename _Tp2, int _rows2, int _cols2, int _options2, int _maxRows2, int _maxCols2>
	static inline void AssignByIndices(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mdest, const Matrix<_Tp2, _rows2, _cols2, _options2, _maxRows2, _maxCols2> *Mindices, const _Tp val) {
		//assert(Mindices->minCoeff() >= 0 && Mindices->maxCoeff() < (Mdest->rows()*Mdest->cols()), "AssignByIndices() indices in Mindices must references positions in Mdest"); // correct assertion but too costly in terms of computation time

		_Tp *md = Mdest->data();
		const _Tp2 *mi = Mindices->data();
		_Tp2 idx;
		int size = Mindices->cols()*Mindices->rows();
		for (int i = 0; i < size; i++) {
			idx = *mi++;
			md[idx] = val;
		}
	}

	// Mdest(Mbooleans) = val
	// updates Mdest as follows: for each boolean in Mbooleans, if it is true, update the associated position in Mdest with the value val
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static inline void AssignByBooleans(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mdest, const Matrix<bool, _rows, _cols, _options, _maxRows, _maxCols> *Mbooleans, const _Tp val) {
		assert(Mdest->rows() == Mbooleans->rows() && Mdest->cols() == Mbooleans->cols());
		const bool *mb = Mbooleans->data();
		_Tp *md = Mdest->data();
		int size = Mbooleans->cols()*Mbooleans->rows();
		for (int i = 0; i < size; i++) {
			if (*mb++) *md = val;
			md++;
		}
	}

	// Mdest(!Mbooleans) = val
	// updates Mdest as follows: for each boolean in Mbooleans, if it is false, update the associated position in Mdest with the value val
	// unlike AssignByBooleans() above, doesn't require Mdest and Mbooleans to have same shape
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, int _rows2, int _cols2, int _options2, int _maxRows2, int _maxCols2>
	static inline void AssignByBooleansNot(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mdest, const Matrix<bool, _rows2, _cols2, _options2, _maxRows2, _maxCols2> *Mbooleans, const _Tp val) {
		assert(Mdest->rows()*Mdest->cols() == Mbooleans->rows()*Mbooleans->cols());
		const bool *mb = Mbooleans->data();
		_Tp *md = Mdest->data();
		int size = Mbooleans->cols()*Mbooleans->rows();
		bool bv;
		for (int i = 0; i < size; i++) {
			bv = *mb++;
			if (!bv) *md = val;
			md++;
		}
	}

	// Mdest(Mbooleans) = Mvals(Mbooleans)
	// Mvals is of a size equal to Mdest and Mbooleans
	// updates Mdest as follows: for each boolean in Mbooleans, if it is true, update the associated position in Mdest with the associated value in Mvals
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, int _rows2, int _cols2, int _options2, int _maxRows2, int _maxCols2, int _rows3, int _cols3, int _options3, int _maxRows3, int _maxCols3>
	static inline void AssignByBooleans(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mdest, const Matrix<bool, _rows2, _cols2, _options2, _maxRows2, _maxCols2> *Mbooleans, const Matrix<_Tp, _rows3, _cols3, _options3, _maxRows3, _maxCols3> *Mvals) {
		assert(Mdest->rows()*Mdest->cols() == Mbooleans->rows()*Mbooleans->cols() && Mdest->rows()*Mdest->cols() == Mvals->rows()*Mvals->cols());
		_Tp *md = Mdest->data();
		const bool *mb = Mbooleans->data();
		const _Tp *mv = Mvals->data();
		int size = Mbooleans->cols()*Mbooleans->rows();
		for (int i = 0; i < size; i++) {
			if (*mb++) *md = *mv;
			md++;
			mv++;
		}
	}

	// Mdest = Mvals(Mbools)
	// returns matrix Mret where each position's associated integer boolean value in Mbools is used to determine whether or not to include the value from Mvals in Mret
	// the sizes of the two matrices must be the same (though Matlab allows for differing arrangements)
	// all qualifying values are returned in Mret as a single row
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, int _rows2, int _cols2, int _options2, int _maxRows2, int _maxCols2, int _rows3, int _cols3, int _options3, int _maxRows3, int _maxCols3>
	static inline void AssignByBooleansOfVals(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mdest, const Matrix<bool, _rows2, _cols2, _options2, _maxRows2, _maxCols2> *Mbools, Matrix<_Tp, _rows3, _cols3, _options3, _maxRows3, _maxCols3> *Mvals) {
		assert(Mbools->rows() * Mbools->cols() == Mvals->rows() * Mvals->cols());
		//assert(Mbools->count() == Mdest->rows()*Mdest->cols(), "Mdest must have number of elements equal to number of true values in Mbooleans"); // correct assertion but too costly in terms of computation time

		int size = Mbools->cols()*Mbools->rows();

		const bool *mb = Mbools->data();
		_Tp *mv = Mvals->data();
		_Tp *md = Mdest->data();
		for (int idx = 0; idx < size; idx++) {
			if (*mb++) *md++ = *mv;
			mv++;
		}
	}

	// Mdest(Mbooleans) = Mvals
	// Mvals is of a truncated size compared with Mdest and Mbooleans (size equals size of Mdest(Mbooleans))
	// updates Mdest as follows: for each boolean in Mbooleans, if it is true, update the associated position in Mdest with the current value in Mvals
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, int _rows2, int _cols2, int _options2, int _maxRows2, int _maxCols2, int _rows3, int _cols3, int _options3, int _maxRows3, int _maxCols3>
	static inline void AssignByTruncatedBooleans(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mdest, const Matrix<bool, _rows3, _cols3, _options3, _maxRows3, _maxCols3> *Mbooleans, const Matrix<_Tp, _rows2, _cols2, _options2, _maxRows2, _maxCols2> *Mvals) {
		assert(Mdest->rows() * Mdest->cols() == Mbooleans->rows() * Mbooleans->cols());
		//assert(Mbooleans->count() == Mvals->rows()*Mvals->cols(), "Mvals must have number of elements equal to number of true values in Mbooleans"); // correct assertion but too costly in terms of computation time
		const bool *mb = Mbooleans->data();
		_Tp *md = Mdest->data();
		const _Tp *mv = Mvals->data();
		int size = Mbooleans->cols()*Mbooleans->rows();
		for (int i = 0; i < size; i++) {
			if (*mb++) {
				*md = *mv++;
			}
			md++;
		}
	}

	// Mvals(Mbools)
	// returns matrix Mret where each position's associated integer boolean value in Mbools is used to determine whether or not to include the value from Mvals in Mret
	// the sizes of the two matrices must be the same (though Matlab allows for differing arrangements)
	// all qualifying values are returned in Mret as a single row
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, typename _Tp2, int _rows2, int _cols2, int _options2, int _maxRows2, int _maxCols2>
	static inline Matrix<_Tp, 1, Dynamic> TruncateByBooleans(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mvals, const Matrix<_Tp2, _rows2, _cols2, _options2, _maxRows2, _maxCols2> *Mbools) {
		assert(Mbools->rows()*Mbools->cols() == Mvals->rows()*Mvals->cols());

		//Matrix<_Tp, 1, Dynamic> Mret(1, Mbools->count()); // count is too costly in terms of computation time

		int size = Mbools->cols()*Mbools->rows();
		Matrix<_Tp, 1, Dynamic> Mret_tmp(1, size);

		const _Tp2 *mb = Mbools->data();
		_Tp *mv = Mvals->data();
		_Tp *mr = Mret_tmp.data();
		int i = 0;
		for (int idx = 0; idx < size; idx++) {
			if (*mb++) {
				*mr++ = *mv;
				i++;
			}
			mv++;
		}

		Matrix<_Tp, 1, Dynamic> Mret = Mret_tmp.block(0, 0, 1, i);

		return Mret;
	}

	// Mvals(:, Mbools)
	// returns matrix Mret where each columns's associated integer boolean value in Mbools is used to determine whether or not to include the column from Mvals in Mret
	// the column counts of the two matrices must be the same (though Matlab allows for differing arrangements)
	// all qualifying columns are returned in Mret
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, typename _Tp2, int _rows2, int _options2, int _maxRows2>
	static inline Matrix<_Tp, Dynamic, Dynamic> TruncateByBooleansColumns(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mvals, const Matrix<_Tp2, _rows2, _cols, _options2, _maxRows2, _maxCols> *Mbools) {
		assert(Mbools->cols() == Mvals->cols());
		assert(Mbools->rows() == 1);
		//Matrix<_Tp, Dynamic, Dynamic> Mret(Mvals->rows(), Mbools->count()); // count is too costly in terms of computation time

		int size = Mbools->cols();
		Matrix<_Tp, Dynamic, Dynamic> Mret_tmp(Mvals->rows(), size);

		const _Tp2 *mb = Mbools->data();
		int i = 0;
		for (int c = 0; c < size; c++) {
			if (!*mb++) continue;
			Mret_tmp.col(i) = Mvals->col(c);
			i++;
		}

		Matrix<_Tp, Dynamic, Dynamic> Mret = Mret_tmp.block(0, 0, Mret_tmp.rows(), i);

		return Mret;
	}

	// Mvals(Mbools, :)
	// returns matrix Mret where each row's associated integer boolean value in Mbools is used to determine whether or not to include the row from Mvals in Mret
	// the row counts of the two matrices must be the same (though Matlab allows for differing arrangements)
	// all qualifying rows are returned in Mret
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, typename _Tp2, int _cols2, int _options2, int _maxCols2>
	static inline Matrix<_Tp, Dynamic, _cols> TruncateByBooleansRows(const Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *Mvals, const Matrix<_Tp2, _rows, _cols2, _options2, _maxRows, _maxCols2> *Mbools) {
		assert(Mbools->rows() == Mvals->rows());
		assert(Mbools->cols() == 1);
		//Matrix<_Tp, Dynamic, Dynamic> Mret(Mbools->count(), Mvals->cols()); // count is too costly in terms of computation time

		int size = Mbools->rows();
		Matrix<_Tp, Dynamic, _cols> Mret_tmp(size, Mvals->cols());

		const _Tp2 *mb = Mbools->data();
		int i = 0;
		for (int r = 0; r < size; r++) {
			if (!*mb++) continue;
			Mret_tmp.row(i) = Mvals->row(r);
			i++;
		}

		Matrix<_Tp, Dynamic, _cols> Mret = Mret_tmp.block(0, 0, i, Mret_tmp.cols());

		return Mret;
	}

	// ~dest
	// modifies dest with a coefficient-wise boolean NOT operation
	template<int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static inline void CwiseNot(Matrix<bool, _rows, _cols, _options, _maxRows, _maxCols> *dest) {
		(*dest) = dest->array() == 0;
	}

	// A|B
	// returns matrix that results from a coefficient-wise boolean OR operation between A and B
	template<int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static inline Matrix<bool, _rows, _cols, _options, _maxRows, _maxCols> CwiseOr(const Matrix<bool, _rows, _cols, _options, _maxRows, _maxCols> *A, const Matrix<bool, _rows, _cols, _options, _maxRows, _maxCols> *B) {
		Matrix<bool, _rows, _cols, _options, _maxRows, _maxCols> dest;
		Matrix<int, _rows, _cols, _options, _maxRows, _maxCols> dest_int = A->cast<int>() + B->cast<int>();
		dest = dest_int.array() > 0;
		return dest;
	}

	// A&B
	// returns matrix that results from a coefficient-wise boolean AND operation between A and B
	template<int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static inline Matrix<bool, _rows, _cols, _options, _maxRows, _maxCols> CwiseAnd(const Matrix<bool, _rows, _cols, _options, _maxRows, _maxCols> *A, const Matrix<bool, _rows, _cols, _options, _maxRows, _maxCols> *B) {
		Matrix<bool, _rows, _cols, _options, _maxRows, _maxCols> dest;
		Matrix<int, _rows, _cols, _options, _maxRows, _maxCols> dest_int = A->cast<int>() + B->cast<int>();
		dest = dest_int.array() == 2;
		return dest;
	}

	// A = accum(subs, vals, sz)
	// subs is a mx2 matrix of indices into A: row m contains the 2-dimensional index into A; the row number of subs also refers to the index into vals
	// vals is a 1xm or mx1 vector of values
	// height_out and width_out give the size for output matrix A
	// if multiple indices in subs for A are the same (i.e. multiple rows in subs have the same values), the corresponding values in vals are summed before adding to A (or summed as they are added)
	// any elements of A not indexed by subs are given the value 0
	template<typename _Tp_ind, int _rows_subs, int _cols_subs, int _options_subs, int _maxRows_subs, int _maxCols_subs, typename _Tp_vals, int _rows_vals, int _cols_vals, int _options_vals, int _maxRows_vals, int _maxCols_vals>
	static inline Matrix<_Tp_vals, Dynamic, Dynamic> Accumarray(const Matrix<_Tp_ind, _rows_subs, _cols_subs, _options_subs, _maxRows_subs, _maxCols_subs> *subs, const Matrix<_Tp_vals, _rows_vals, _cols_vals, _options_vals, _maxRows_vals, _maxCols_vals> *vals, int height_out, int width_out) {
		assert(subs->cols() <= 2);
		assert(vals->rows() == 1 || vals->cols() == 1);
		assert(vals->rows()*vals->cols() == subs->rows());
		//assert(subs->col(0).minCoeff() >= 0 && subs->col(0).maxCoeff() < height_out, "EigenMatlab::Accumarray() subs col 0 indices into output A must be within bounds (0,0) through (height_out-1, width_out-1)"); // correct assertion but too costly in terms of computation time
		//assert(subs->cols() == 1 || (subs->col(1).minCoeff() >= 0 && subs->col(1).maxCoeff() < width_out), "EigenMatlab::Accumarray() subs col 1 (if it exists) indices into output A must be within bounds (0,0) through (height_out-1, width_out-1)"); // correct assertion but too costly in terms of computation time

		Matrix<_Tp_vals, Dynamic, Dynamic> A(height_out, width_out);
		A.setZero();

		int r_out, c_out;
		int num_rows = subs->rows();
		int num_cols = subs->cols();
		const _Tp_ind* ms = subs->transpose().data();
		const _Tp_vals* mv = vals->data();
		_Tp_vals val;

		if (num_cols > 1) {
			for (int r = 0; r < num_rows; r++) {
				r_out = *ms++;
				c_out = *ms++;
				val = *mv++;
				A(r_out, c_out) = A(r_out, c_out) + val;
			}
		}
		else {
			c_out = 0;
			for (int r = 0; r < num_rows; r++) {
				r_out = *ms++;
				val = *mv++;
				A(r_out, c_out) = A(r_out, c_out) + val;
			}
		}

		return A;
	}

	// A = accum(subs, vals, sz, @num_first) where @num_first is a function that returns the negative of the number of elements in an array, but if that number is 1, returns the single value in the array instead of -1
	// subs is a mx2 matrix of indices into A: row m contains the 2-dimensional index into A; the row number of subs also refers to the index into vals
	// vals is a 1xm or mx1 vector of values
	// height_out and width_out give the size for output matrix A
	// if multiple indices in subs for A are the same (i.e. multiple rows in subs have the same values), the negative of the number of elements that share an index is used, but if that number is 1, returns the single value from vals into the array instead of -1
	// any elements of A not indexed by subs are given the value 0
	// since we're using 0-indexing here, function has been changed so that single values are incremented by 1; so positive values are the vals+1, negative values are the negative of the number of times the value appeared, and 0s mean it didn't appear
	template<typename _Tp_ind, int _rows_subs, int _cols_subs, int _options_subs, int _maxRows_subs, int _maxCols_subs, typename _Tp_vals, int _rows_vals, int _cols_vals, int _options_vals, int _maxRows_vals, int _maxCols_vals>
	static inline Matrix<_Tp_vals, Dynamic, Dynamic> Accumarray_NumFirst(const Matrix<_Tp_ind, _rows_subs, _cols_subs, _options_subs, _maxRows_subs, _maxCols_subs> *subs, const Matrix<_Tp_vals, _rows_vals, _cols_vals, _options_vals, _maxRows_vals, _maxCols_vals> *vals, int height_out, int width_out) {
		assert(subs->cols() <= 2);
		assert(vals->rows() == 1 || vals->cols() == 1);
		assert(vals->rows()*vals->cols() == subs->rows());
		//assert(subs->col(0).minCoeff() >= 0 && subs->col(0).maxCoeff() < height_out, "EigenMatlab::Accumarray() subs col 0 indices into output A must be within bounds (0,0) through (height_out-1, width_out-1)"); // correct assertion but too costly in terms of computation time
		//assert(subs->cols() == 1 || (subs->col(1).minCoeff() >= 0 && subs->col(1).maxCoeff() < width_out), "EigenMatlab::Accumarray() subs col 1 (if it exists) indices into output A must be within bounds (0,0) through (height_out-1, width_out-1)"); // correct assertion but too costly in terms of computation time

		Matrix<_Tp_vals, Dynamic, Dynamic> A(height_out, width_out);
		A.setZero();

		int r_out, c_out;
		int num_rows = subs->rows();
		int num_cols = subs->cols();
		const _Tp_ind* ms = subs->transpose().data();
		const _Tp_vals* mv = vals->data();
		_Tp_vals val;

		// first pass determines the number of elements that match
		if (num_cols > 1) {
			for (int r = 0; r < num_rows; r++) {
				r_out = *ms++;
				c_out = *ms++;
				val = *mv++;
				A(r_out, c_out) = A(r_out, c_out) - 1;
			}
		}
		else {
			c_out = 0;
			for (int r = 0; r < num_rows; r++) {
				r_out = *ms++;
				val = *mv++;
				A(r_out, c_out) = A(r_out, c_out) - 1;
			}
		}

		// second pass assigns val from Vals if only a single elements matches
		ms = subs->transpose().data();
		mv = vals->data();
		if (num_cols > 1) {
			for (int r = 0; r < num_rows; r++) {
				r_out = *ms++;
				c_out = *ms++;
				val = *mv++;
				if (A(r_out, c_out) == -1)
					A(r_out, c_out) = val + 1;
			}
		}
		else {
			c_out = 0;
			for (int r = 0; r < num_rows; r++) {
				r_out = *ms++;
				val = *mv++;
				if (A(r_out, c_out) == -1)
					A(r_out, c_out) = val + 1;
			}
		}

		return A;
	}

	// works like Matlab's diff, but requires more than 1 row and at least 1 column
	// returns a matrix of one less row and same number of columns as the input matrix, but the coefficients are the differences between the input matrix's rows
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static inline Matrix<_Tp, Dynamic, _cols> Diff(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *M) {
		assert(M->rows() > 1 && M->cols() > 0);

		Matrix<_Tp, Dynamic, _cols> Mret(M->rows() - 1, M->cols());

		int size = Mret.rows();
		for (int r = 0; r < size; r++) {
			Mret.row(r) = M->row(r + 1) - M->row(r);
		}

		return Mret;
	}

	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static inline Matrix<int, _rows, _cols> cwiseRound(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *M) {
		Matrix<int, _rows, _cols> Mret(M->rows(), M->cols());

		_Tp *m = M->data();
		int *mr = Mret.data();
		int n = M->cols() * M->rows();
		for (int i = 0; i < n; i++){
			*mr++ = round(*m++, 0);
		}

		return Mret;
	}

	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static inline Matrix<int, _rows, _cols> cwiseFloor(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *M) {
		Matrix<int, _rows, _cols> Mret(M->rows(), M->cols());

		_Tp *m = M->data();
		int *mr = Mret.data();
		int n = M->cols() * M->rows();
		for (int i = 0; i < n; i++){
			*mr++ = std::floor(*m++);
		}

		return Mret;
	}

	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static inline Matrix<int, _rows, _cols> cwiseCeiling(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *M) {
		Matrix<int, _rows, _cols> Mret(M->rows(), M->cols());

		_Tp *m = M->data();
		int *mr = Mret->data();
		int n = M->cols() * M->rows();
		for (int i = 0; i < n; i++){
			*mr++ = ceil(*m++);
		}

		return Mret;
	}

	// returns true if the two matrices are equal in both size and coefficients, false otherwise
	// requires that they are of the same type
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, int _rows2, int _cols2, int _options2, int _maxCols2>
	static inline bool TestEqual(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *A, const Matrix<_Tp, _rows2, _cols2, _options2, _maxRows, _maxCols2> *B) {
		bool debug = true;

		if ((A->rows() != B->rows()) ||
			(A->cols() != B->cols())) return false;

		bool match = true;

		const _Tp *a = A->data();
		const _Tp *b = B->data();
		int size = A->rows()*A->cols();
		for (int i = 0; i < size; i++) {
			if (*a++ != *b++) {
				if (debug) cout << "Mismatch at " << i << ": A = " << (*A)(i) << ", B = " << (*B)(i) << endl;
				match = false;
			}
		}

		return match;
	}

	// modifies A to mask it according to Mask
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
	static inline void Mask(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *A, const Matrix<bool, _rows, _cols, _options, _maxRows, _maxCols> *Mask) {
		(*A) = A->cwiseProduct(Mask->cast<_Tp>());
	}

	// modifies A to mask it row-wise according to Mask
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, int _options2, int _maxCols2>
	static inline void MaskRows(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *A, const Matrix<bool, _rows, 1, _options2, _maxRows, _maxCols2> *Mask) {
		for (int c = 0; c < A->cols(); c++) {
			A->col(c) = A->col(c).cwiseProduct(Mask->cast<_Tp>());
		}
	}

	// modifies A to mask it column-wise according to Mask
	template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols, int  _options2, int _maxRows2>
	static inline void MaskCols(Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols> *A, const Matrix<bool, 1, _cols, _options2, _maxRows2, _maxCols> *Mask) {
		for (int r = 0; r < A->rows(); r++) {
			A->row(r) = A->row(r).cwiseProduct(Mask->cast<_Tp>());
		}
	}


};

#endif