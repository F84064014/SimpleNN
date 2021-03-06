#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>

#include <cstdlib>
#include <ctime>
#include <cmath>

#include <pybind11/pybind11.h>

// set Eigen Matrix row major (default column major)
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXd = Eigen::VectorXd;

namespace py = pybind11;

// sum matrix by row
VectorXd sum_by_row(Eigen::Ref<RowMatrixXd> x)
{
	VectorXd ret(x.cols());
	for (size_t j=0; j<x.cols(); ++j)
	{
		ret(j) = x.col(j).sum();
	}
	return ret;
}

/*
 * bitwise mulitplication
*/
Eigen::Ref<RowMatrixXd> bitwise_mul(const Eigen::Ref<RowMatrixXd> m1, const Eigen::Ref<RowMatrixXd> m2)
{

	if ( (m1.rows() != m2.rows()) || (m1.cols() != m2.cols()) )
		throw "error incorrect dimension";

	RowMatrixXd ret(m1.rows(), m1.cols());
	for (size_t i=0; i<m1.rows(); ++i)
		for (size_t j=0; j<m2.cols(); ++j)
			ret(i, j) = m1(i, j) * m2(i, j);

	return ret;
}

/*
 * activation fucntion (attach to dense layer)
*/
class Activation{

public:
    //virtual RowMatrixXd forward(Eigen::Ref<RowMatrixXd>) = 0;
    virtual void forward(Eigen::Ref<RowMatrixXd>) = 0;
    virtual RowMatrixXd backward(const Eigen::Ref<RowMatrixXd>) = 0;
    virtual std::string toString() = 0;
};


/*
 * layer with trainable parameter
*/
class Layer{

public:
	RowMatrixXd m_weight;
	Eigen::VectorXd m_bias;
	Activation * m_activation;
    
	virtual RowMatrixXd forward(const Eigen::Ref<RowMatrixXd>) = 0;
    virtual RowMatrixXd backward(const Eigen::Ref<RowMatrixXd>, const Eigen::Ref<RowMatrixXd>) = 0;
	virtual std::string toString() = 0;
	virtual const RowMatrixXd & getWeight() = 0;
	virtual const Eigen::VectorXd & getBias() = 0;
	virtual void saveWeight(std::ofstream&) = 0;
	virtual void set_lr(double) = 0;
};


// Identical layer (no activation function
class Identical : public Activation {

public:
	
	Identical(){}

	void forward(Eigen::Ref<RowMatrixXd> x) {}

	RowMatrixXd backward(const Eigen::Ref<RowMatrixXd> x)
	{
		return x;
	}
	
	std::string toString(){ return "N";}

};

/*
 * ReLu layer equal to np.where(x>0, x, 0)
*/
class ReLu : public Activation {

public:

	// nothing
	ReLu(){}
	
	void forward(Eigen::Ref<RowMatrixXd> x)
	{

		for (size_t i=0; i<x.rows(); ++i)
			for (size_t j=0; j<x.cols(); ++j)
				if (x(i, j) < 0)
					x(i, j) = 0;
	}

	RowMatrixXd backward(const Eigen::Ref<RowMatrixXd> x)
	{
		RowMatrixXd ret(x.rows(), x.cols());
		for (size_t i=0; i<x.rows(); ++i)
		{
			for (size_t j=0; j<x.cols(); ++j)
			{
				if(x(i, j) < 0)
					ret(i, j) = 0;
				else
					ret(i, j) = 1;
			}
		}
		return ret;
	}

	std::string toString()
	{
		return "ReLu";
	}

};

/*
 * sigmoid function 1 / (1 + exp(-x))
*/
class Sigmoid : public Activation {

public:

	// nothing
	Sigmoid(){}

	void forward(Eigen::Ref<RowMatrixXd> x)
	{

		//m.unaryExpr([](double x){return x + 1})
		//for (size_t i=0; i<x.rows(); ++i)
		//	for (size_t j=0; j<x.cols(); ++j)
		//		x(i, j) = 1 / (1 + (double)(exp(-x(i, j)) ));
		x = x.unaryExpr([](double x){return 1/(1+exp(-x));});
	}

	RowMatrixXd backward(const Eigen::Ref<RowMatrixXd> x)
	{
		/*
		RowMatrixXd ret(x.rows(), x.cols());
		for (size_t i=0; i<x.rows(); ++i)
			for (size_t j=0; j<x.cols(); ++j)
				ret(i, j) = ( exp(-x(i,j)) / pow((1+exp(-x(i, j))), 2) );
		return ret;*/
		return x.unaryExpr([](double x){return exp(-x) / ((1+exp(-x)) * (1+exp(-x)));});
	}

	std::string toString()
	{
		return "Sigmoid";
	}
};


/*
 * fully connected layer
*/
class Dense : public Layer{
	
public:
	
	Eigen::VectorXd m_bias;
	RowMatrixXd m_weight;
	double m_lr;
	Activation *m_activate;
	
	// record last output (for backward propagation)
	RowMatrixXd m_out;
	
	void set_lr(double lr) {m_lr = lr;}
	const RowMatrixXd & getWeight() {return m_weight;}
	const Eigen::VectorXd & getBias() {return m_bias;}
	

	Dense(size_t input_dim, size_t output_dim, std::string activation, double learning_rate = 0.1, int seed=0)
	{
		// set correct size
		m_weight.resize(input_dim, output_dim);

		// randomly initialize weight to (-1,1)
		srand(seed);
		for (size_t i=0; i<input_dim; ++i)
			for (size_t j=0; j<output_dim; ++j)
				m_weight(i, j) = (double) (rand() / (RAND_MAX + 1.0) * 2 - 1)/(sqrt(input_dim * output_dim));

		m_bias.resize(output_dim);
		for (size_t i=0; i<output_dim; ++i) m_bias(i) = 1.0;
		

		// set activation function
		if (activation == "sigmoid")
		{
			m_activate = new Sigmoid();
		}
		else if (activation == "relu")
		{
			m_activate = new ReLu();
		}
		else if (activation == "identical")
		{
			m_activate = new Identical();
		}

		m_lr = learning_rate;

	}

	RowMatrixXd forward(const Eigen::Ref<RowMatrixXd> x)
	{
		RowMatrixXd ret = x * m_weight;
		ret.rowwise() += m_bias.transpose();

		// copy to m_out
		m_out = ret;

		m_activate->forward(ret);
		return ret;
	}

	std::string toString()
	{
		std::stringstream sstm;
		sstm << "Dense (" << m_weight.rows() << ", " << m_weight.cols() << ") " << m_activate->toString();
		return sstm.str();
	}

	RowMatrixXd backward(const Eigen::Ref<RowMatrixXd> x_input, const Eigen::Ref<RowMatrixXd> grad_output)
	{

		//py::print("m_out", m_out.transpose());//test
		RowMatrixXd temp = m_activate->backward(m_out);
		//py::print("temp", temp.transpose());//test
		temp = bitwise_mul(grad_output, temp);
		RowMatrixXd grad_input = temp * m_weight.transpose();
		//py::print("x_input", x_input.transpose());//test
		//py::print("grad_out", grad_output.transpose());//test

		RowMatrixXd grad_weight = (x_input.transpose() * temp);
		//py::print("grad_w",grad_weight);//test
		//py::print("weight before", m_weight);//test
		m_weight.noalias() -= m_lr * grad_weight;
		//py::print("weight after", m_weight);//test

		VectorXd grad_bias = sum_by_row(temp);// * x_input.rows();
		m_bias.noalias() -= m_lr * grad_bias;

		return grad_input;
	}

	void saveWeight(std::ofstream & f)
	{
		f << "{\"type\": \"Dense\",\"w\": [";
		for (size_t i=0; i<m_weight.rows(); ++i)
		{
			for (size_t j=0; j<m_weight.cols(); ++j)
			{
				f << std::to_string(m_weight(i, j));
				if ( ! ((i==m_weight.rows()-1) && (j==m_weight.cols()-1)) )
				{
					f << ",";
				}
			}
			
		}
		
		f << "],\"b\": [";
		for (size_t i=0; i<m_bias.rows(); ++i)
		{
			f << std::to_string(m_bias(i,0));
			if (i!=m_bias.rows()-1)
				f << ",";
		}
		f << "]}\n";
	}
};
