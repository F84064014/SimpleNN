#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <ctime>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include "Layer.h"
#include "Loss.h"

RowMatrixXd backward(RowMatrixXd x_input, RowMatrixXd grad_output, Eigen::Ref<RowMatrixXd> m_weight, Eigen::Ref<RowMatrixXd> m_bias, RowMatrixXd m_out, std::string activate)
{
	Activation* m_activate;

	if (activate == "Sigmoid")
		m_activate = new Sigmoid();
	else if (activate == "ReLu")
		m_activate = new ReLu();		

	double m_lr = 0.1;

	RowMatrixXd temp = m_activate->backward(m_out);
	grad_output = bitwise_mul(grad_output, temp);
	RowMatrixXd grad_input = grad_output * m_weight.transpose();
	
	RowMatrixXd grad_weight = (x_input.transpose() * grad_output);
	m_weight = m_weight - m_lr * grad_weight;

    RowMatrixXd grad_bias = sum_by_row(grad_output);// * x_input.rows();
    m_bias = m_bias - m_lr * grad_bias;

    return grad_input;
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

RowMatrixXd forward(RowMatrixXd x, RowMatrixXd m_weight, Eigen::VectorXd m_bias, Eigen::Ref<RowMatrixXd> m_out)
{
	Activation* m_activate = new Sigmoid();

    RowMatrixXd ret(x.rows(), m_weight.cols());
    ret = x * m_weight;
    ret.rowwise() += m_bias.transpose();

    // copy to m_out
    m_out = ret;

    m_activate->forward(ret);
    return ret;
}

RowMatrixXd sigmoid_backward(Eigen::Ref<RowMatrixXd> x)
{
    RowMatrixXd ret(x.rows(), x.cols());
    for (size_t i=0; i<x.rows(); ++i)
        for (size_t j=0; j<x.cols(); ++j)
            ret(i, j) = ( exp(-x(i,j)) / pow((1+exp(-x(i, j))), 2) );
    return ret;
}



/*
 * To test correctness
*/
PYBIND11_MODULE(_testNN, m) {
	m.def("backward", & backward);
	m.def("bitwise_mul", & bitwise_mul);
	m.def("cal_softmax", & cal_softmax);
	m.def("cal_grad", & cal_grad);
	m.def("forward", & forward);
	m.def("sigmoid_backward", & sigmoid_backward);
}
