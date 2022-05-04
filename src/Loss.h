#pragma once

#include <iostream>
#include <cmath>

#include <Eigen/Dense>

// set Eigen Matrix row major (default column major)
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

//namespace py = pybind11;

/*
 * size check function for two Eigne Matrix
*/
void check_same_size(Eigen::Ref<RowMatrixXd> mat1, Eigen::Ref<RowMatrixXd> mat2)
{
	if ( (mat1.rows() != mat2.rows()) || (mat1.cols() != mat2.cols()) )
		throw "incompatible matrix size";
}

/*
 * matrix exponential
*/ 
RowMatrixXd expm(Eigen::Ref<RowMatrixXd> mat) {
	
	RowMatrixXd ret(mat.rows(), mat.cols());
	for (size_t i=0; i<mat.rows(); ++i)
		for (size_t j=0; j<mat.cols(); ++j)
			ret(i, j) = exp(mat(i, j));
	
	return ret;
}

/*
 * matrix log
*/
RowMatrixXd logm(Eigen::Ref<RowMatrixXd> mat){

	RowMatrixXd ret(mat.rows(), mat.cols());
	for (size_t i=0; i<mat.rows(); ++i)
		for (size_t j=0; j<mat.cols(); ++j)
			ret(i, j) = log(mat(i, j));

	return ret;
}


/*
 * sum by row
*/
RowMatrixXd sum_row(Eigen::Ref<RowMatrixXd> mat){

	RowMatrixXd ret(mat.rows(), 1);
	ret.setZero();
	for (size_t i=0; i<mat.rows(); ++i)
	{
		for (size_t j=0; j<mat.cols(); ++j)
		{
			ret(i, 0) += mat(i, j);
		}
	}

	return ret;
}

/*
 * convert matrix to  onehot style
 *  [1,1,3,0,2].T for exaple
 * :
 * [[0,0,0,1,0],
 *  [1,1,0,0,0],
 *  [0,0,0,0,1],
 *  [0,0,1,0,0],]
*/
RowMatrixXd onehot_matrix(Eigen::Ref<RowMatrixXd> mat, int dim){
	
	RowMatrixXd ret(mat.rows(), dim);
	ret.setZero();
	for (size_t i=0; i<mat.rows(); ++i)
	{
		ret(i, int(mat(i,0))) = 1.0;
	}
	return ret;
}


/*
 * matrix softmax
*/ 
RowMatrixXd cal_softmax(Eigen::Ref<RowMatrixXd> mat){
	
	RowMatrixXd ret = RowMatrixXd(mat.rows(), mat.cols());

	RowMatrixXd temp = expm(mat);
	
	RowMatrixXd temp_rowsum = sum_row(temp);

	for (size_t i=0; i<mat.rows(); ++i)
	{
		for (size_t j=0; j<mat.cols(); ++j)
		{
			ret(i, j) = temp(i, j) / temp_rowsum(i, 0);
		}
	}
	return ret;
}

/*
 * base function of loss
*/
class LossFunc {

public:	
	virtual double cal_loss(const Eigen::Ref<RowMatrixXd> y_out, const Eigen::Ref<RowMatrixXd> y_true) = 0;
	virtual RowMatrixXd cal_grad(Eigen::Ref<RowMatrixXd> y_out, Eigen::Ref<RowMatrixXd> y_true) = 0;

};

/*
 * Combine Softmax and CrossEntropy 
 * A sigmoid input in previous layer is necessary? softmax cross entropy with logit
 * The y_out is considered to be logit
*/
class CrossEntropyLoss : public LossFunc{

public:
	double cal_loss(Eigen::Ref<RowMatrixXd> y_out, Eigen::Ref<RowMatrixXd> y_true)
	{

		RowMatrixXd y_target(y_true.rows(), 1);
		for (size_t i=0; i<y_true.rows(); ++i)
		{
			y_target(i,0) = y_out(i, int(y_true(i, 0)));
		}
		
		RowMatrixXd temp = expm(y_out);

		temp = sum_row(temp);

		temp = logm(temp);

		return (temp-y_target).sum();

	}

	RowMatrixXd cal_grad(Eigen::Ref<RowMatrixXd> y_out, Eigen::Ref<RowMatrixXd> y_true)
	{
		// from 0 to max(y_true)
		int dim = (int) (y_true.maxCoeff())+1;
		
		// (n, 1) -> (n, p)
		RowMatrixXd onehot_y_true = onehot_matrix(y_true, dim);

		RowMatrixXd softmax = cal_softmax(y_out);
		
		return (softmax - onehot_y_true) / y_out.rows();
	}
};

/*
 * MSELoss
*/
class MSELoss : public LossFunc{

public:
	double cal_loss(Eigen::Ref<RowMatrixXd> y_out, Eigen::Ref<RowMatrixXd> y_true)
	{
		return 2.0;
	}
	RowMatrixXd cal_grad(Eigen::Ref<RowMatrixXd> y_out, Eigen::Ref<RowMatrixXd> y_true)
	{
		return RowMatrixXd(10,10);
	}

};
