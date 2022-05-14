#pragma once

#include <iostream>
#include <cmath>

#include <Eigen/Dense>

// set Eigen Matrix row major (default column major)
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXd = Eigen::VectorXd;

//namespace py = pybind11;

/*
 * size check function for two Eigne Matrix
*/
void check_same_size(const Eigen::Ref<RowMatrixXd> mat1, const Eigen::Ref<RowMatrixXd> mat2)
{
	if ( (mat1.rows() != mat2.rows()) || (mat1.cols() != mat2.cols()) )
		throw "incompatible matrix size";
}

/*
 * sum by row
*/
VectorXd sum_row(const Eigen::Ref<RowMatrixXd> mat){

	VectorXd ret(mat.rows());
	ret.setZero();
	for (size_t i=0; i<mat.rows(); ++i)
	{
		ret(i) = mat.row(i).sum();
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
RowMatrixXd onehot_matrix(Eigen::Ref<VectorXd> vec, int dim){
	
	RowMatrixXd ret(vec.size(), dim);
	ret.setZero();
	for (size_t i=0; i<vec.size(); ++i)
	{
		ret(i, int(vec(i))) = 1.0;
	}
	return ret;
}


/*
 * matrix softmax
*/ 
RowMatrixXd cal_softmax(Eigen::Ref<RowMatrixXd> mat){
	
	RowMatrixXd ret = RowMatrixXd(mat.rows(), mat.cols());

	RowMatrixXd temp = mat.unaryExpr([](double x){return exp(x);});
	
	VectorXd temp_rowsum = sum_row(temp);

	for (size_t i=0; i<mat.rows(); ++i)
	{
		double rs = temp.row(i).sum();
		ret.row(i) = (temp.row(i) / rs);
	}
	return ret;
}

/*
 * base function of loss
*/
class LossFunc {

public:	
	virtual double cal_loss(const Eigen::Ref<RowMatrixXd> y_out, const Eigen::Ref<VectorXd> y_true) = 0;
	virtual RowMatrixXd cal_grad(Eigen::Ref<RowMatrixXd> y_out, Eigen::Ref<VectorXd> y_true) = 0;

};

/*
 * Combine Softmax and CrossEntropy 
 * A sigmoid input in previous layer is necessary? softmax cross entropy with logit
 * The y_out is considered to be logit
*/
class CrossEntropyLoss : public LossFunc{

public:
	double cal_loss(Eigen::Ref<RowMatrixXd> y_out, Eigen::Ref<VectorXd> y_true)
	{

		VectorXd y_target(y_true.size());
		for (size_t i=0; i<y_true.size(); ++i)
		{
			y_target(i,0) = y_out(i, int(y_true(i)));
		}
		
		RowMatrixXd t0 = exp(y_out.array());

	 	VectorXd temp = sum_row(t0);

		temp = log(temp.array());

		return (temp-y_target).sum();

	}

	RowMatrixXd cal_grad(Eigen::Ref<RowMatrixXd> y_out, Eigen::Ref<VectorXd> y_true)
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
 * in this case y_out is a vector
*/
class MSELoss : public LossFunc{

public:

	double cal_loss(Eigen::Ref<RowMatrixXd> y_out, Eigen::Ref<VectorXd> y_true)
	{
		return vsquare((y_out.col(0) - y_true)).sum() / y_true.size();
	}
	RowMatrixXd cal_grad(Eigen::Ref<RowMatrixXd> y_out, Eigen::Ref<VectorXd> y_true)
	{
		return RowMatrixXd(y_out - y_true) / y_true.size();
	}

private:

	VectorXd vsquare(VectorXd vec)
	{
		VectorXd ret(vec.size());
		for (size_t i; i<vec.size(); ++i)
			ret(i) = vec(i) * vec(i);
		return ret;
	}

};
