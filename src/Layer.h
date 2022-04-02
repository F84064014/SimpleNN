#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>

#include <cstdlib>
#include <ctime>
#include <cmath>

// set Eigen Matrix row major (default column major)
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace py = pybind11;


/*
 * activation fucntion (attach to dense layer)
*/
class Activation{

public:
    virtual RowMatrixXd forward(Eigen::Ref<RowMatrixXd>) = 0;
    virtual void backward() = 0;
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
    
	virtual RowMatrixXd forward(RowMatrixXd) = 0;
    virtual void backward() = 0;
	virtual std::string toString() = 0;
	virtual const RowMatrixXd & getWeight() = 0;
	virtual const Eigen::VectorXd & getBias() = 0;
	virtual void saveWeight(std::ofstream&) = 0;
};


// Identical layer (no activation function
class Identical : public Activation {

public:
	
	Identical(){}

	RowMatrixXd forward(Eigen::Ref<RowMatrixXd> x) {return x;}

	void backward(){}
	
	std::string toString(){ return "N";}

};

/*
 * ReLu layer equal to np.where(x>0, x, 0)
*/
class ReLu : public Activation {

public:

	// nothing
	ReLu(){}
	
	RowMatrixXd forward(Eigen::Ref<RowMatrixXd> x)
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

/*
 * sigmoid function 1 / (1 + exp(-x))
*/
class Sigmoid : public Activation {

public:

	// nothing
	Sigmoid(){}

	RowMatrixXd forward(Eigen::Ref<RowMatrixXd> x)
	{
		for (size_t i=0; i<x.rows(); ++i)
			for (size_t j=0; j<x.cols(); ++j)
				x(i, j) = 1 / (1 + (double)(exp(-x(i, j)) ));
		return x;
	}

	void backward(){}

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
	double out;
	Activation *m_activate;
	
	const RowMatrixXd & getWeight() {return m_weight;}
	const Eigen::VectorXd & getBias() {return m_bias;}
	

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

	}

	RowMatrixXd forward(RowMatrixXd x)
	{
		RowMatrixXd ret(x.rows(), m_weight.cols());
		ret = x * m_weight;
		ret.rowwise() += m_bias.transpose();
		m_activate->forward(ret);
		return ret;
	}

	std::string toString()
	{
		std::stringstream sstm;
		sstm << "Dense (" << m_weight.rows() << ", " << m_bias.cols() << ") " << m_activate->toString();
		return sstm.str();
	}

	void backward()
	{
		std::cout << "back" << std::endl;
	}

	void saveWeight(std::ofstream & f)
	{
		f << "{\"type\": \"Dense\",\"w\": [";
		for (size_t i=0; i<m_weight.rows(); ++i)
		{
			for (size_t j=0; j<m_weight.cols(); ++j)
			{
				f << m_weight(i, j);
				if ( ! ((i==m_weight.rows()-1) && (j==m_weight.cols()-1)) )
				{
					f << ",";
				}
			}
			
		}
		
		f << "],\"b\": [";
		for (size_t i=0; i<m_bias.rows(); ++i)
		{
			f << m_bias(i,0);
			if (i!=m_bias.rows()-1)
				f << ",";
		}
		f << "]}\n";
	}
};
