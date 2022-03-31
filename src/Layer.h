#pragma once

#include <iostream>
#include <string>
#include <Eigen/Dense>

#include <cstdlib>
#include <ctime>
#include <cmath>

#include "Matrix.h"


// set Eigen Matrix row major (default column major)
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace py = pybind11;

class Layer{

public:
    virtual RowMatrixXd forward(RowMatrixXd) = 0;
    virtual void backward() = 0;
	virtual std::string toString() = 0;
};

// fully connected layer
class Dense : public Layer{
	
public:
	RowMatrixXd m_weight;
	Eigen::VectorXd m_bias;
	double out;
	
	const RowMatrixXd &viewMatrix() 
	{
		return m_weight; 
	}

	Dense(size_t input_dim, size_t output_dim, std::string activation)
	{

		// set correct size
		m_weight.resize(input_dim, output_dim);

		// randomly initialize weight to (-1,1)
		srand(time(NULL));
		for (size_t i=0; i<input_dim; ++i)
			for (size_t j=0; j<output_dim; ++j)
				m_weight(i, j) = (double) rand() / (RAND_MAX + 1.0) * 2 - 1;

		m_bias.resize(output_dim);
		for (size_t i=0; i<output_dim; ++i) m_bias(i) = 1.0;
		out = 0;
	}

	RowMatrixXd forward(RowMatrixXd x)
	{
		RowMatrixXd ret(x.rows(), m_weight.cols());
		ret = x * m_weight;
		ret.rowwise() += m_bias.transpose();
		return ret;
	}

	std::string toString()
	{
		return "Linear layer";
	}

	void backward()
	{
		std::cout << "back" << std::endl;
	}
};


// Identical layer (no activation function
class Identical : public Layer {

public:

	Identical(){}

	RowMatrixXd forward(RowMatrixXd x){return x;}

	void backward(){}
	
	std::string toString(){ return "N";}

};

// ReLu layer equal to np.where(x>0, x, 0)
class ReLu : public Layer {

public:

	// nothing
	ReLu(){}
	
	RowMatrixXd forward(RowMatrixXd x)
	{

		for (size_t i=0; i<x.rows(); ++i)
			for (size_t j=0; j<x.cols(); ++j)
				if (x(i, j) < 0)
					x(i, j) = 0;
		return x;
	}

	void backward(){}

	std::string toString()
	{
		return "ReLu";
	}

};

// sigmoid function 1 / (1 + exp(-x))
class Sigmoid : public Layer {

public:

	// nothing
	Sigmoid(){}

	RowMatrixXd forward(RowMatrixXd x)
	{
		for (size_t i=0; i<x.rows(); ++i)
			for (size_t j=0; j<x.cols(); ++j)
				x(i, j) = 1/(1+exp(-x(i, j)));
		return x;
	}

	void backward(){}

	std::string toString()
	{
		return "Sigmoid";
	}
};
