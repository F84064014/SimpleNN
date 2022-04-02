#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include "Layer.h"
#include "Loss.h"

namespace py = pybind11;

// set Eigen Matrix row major (default column major)
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/*
 * som string operation
 * split into list
 * trim white space
*/
const std::vector<std::string> split(const std::string &str, const char &delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string tok;

    while (std::getline(ss, tok, delimiter)) {
        result.push_back(tok);
    }
    return result;
}

const std::string & trim(std::string &s)
{
	if (s.empty())
	{
		return s;
	}

	s.erase(0, s.find_first_not_of(" "));
	s.erase(s.find_last_not_of(" ") + 1);
	return s;
}

/*
 * FNN class
*/

class FNN{

public:

	FNN(std::string);
	RowMatrixXd forward(const Eigen::Ref<RowMatrixXd>);
	py::array_t<double> fit(
		Eigen::Ref<RowMatrixXd> X_train,
		Eigen::Ref<RowMatrixXd> y_train,
		Eigen::Ref<RowMatrixXd> X_valid,
		Eigen::Ref<RowMatrixXd> y_valid,
		size_t epochs, 
		double lr //learning rate
	);
	RowMatrixXd evaluate(const Eigen::Ref<RowMatrixXd> X);
	double getLoss(const Eigen::Ref<RowMatrixXd> y_out, const Eigen::Ref<RowMatrixXd> y_true);
	void summary();
	void save(std::string);
	void load(std::string);
	void backward();

protected:
	bool m_trainFlag;
	std::list<Layer*> m_layers;
	LossFunc* m_lossfunc;
};

/*
 *	set input/output dimension and layer configuration
*/
FNN::FNN(std::string config_str)
{
	std::vector<std::string> config_vec = split(config_str, '\n');	
	for (std::string config_line: config_vec)
	{
		std::vector<std::string> config_unit_vec = split(config_line, ',');
		

		// Dense layer
		if (trim(config_unit_vec.at(0)) == "linear")
		{
			size_t input_dim = std::stoi(config_unit_vec.at(1));
			size_t output_dim = std::stoi(config_unit_vec.at(2));

			// if no specified activation set identical function
			std::string activation;
			if (config_unit_vec.size() == 3)
				activation = "identical";
			else
				activation = trim(config_unit_vec.at(3));
			
			Dense * new_layer_ptr = new Dense(input_dim, output_dim, activation);
			m_layers.push_back(new_layer_ptr);
			
		}
		// CrossEntropy loss
		else if (config_unit_vec.at(0) == "CrossEntropy")
		{
			m_lossfunc = new CrossEntropyLoss();
		}
	}
}

/*
 *	forward data to each layer
*/
RowMatrixXd FNN::forward(const Eigen::Ref<RowMatrixXd> x)
{
	// copy x, dont know if there is better way
	RowMatrixXd _x(2,2);
	_x = x;

	for (Layer * layer: m_layers)
	{
		_x = layer->forward(_x);
	}

	
	return _x;
}

/*
 * train/evaluate input x with model parameter
 * X_train, y_train are not default 
*/
py::array_t<double> FNN::fit(
	Eigen::Ref<RowMatrixXd> X_train,
	Eigen::Ref<RowMatrixXd> y_train,
	Eigen::Ref<RowMatrixXd> X_valid,
	Eigen::Ref<RowMatrixXd> y_valid,
	size_t epochs = 100,
	double lr = 1e-2
)
{

	for (size_t epoch=0; epoch<epochs; ++epoch)
	{
		RowMatrixXd out = forward(X_train);	
	}

}

/*
 * evaluate the input x
*/
RowMatrixXd FNN::evaluate(const Eigen::Ref<RowMatrixXd> x)
{
	return forward(x);
}

/*
 * calculate loss
*/
double FNN::getLoss(const Eigen::Ref<RowMatrixXd> y_out, const Eigen::Ref<RowMatrixXd> y_true)
{

	return m_lossfunc->cal_loss(y_out, y_true);
}

/*
 * backward proporgation
*/
void FNN::backward()
{
	std::cout << m_lossfunc->cal_grad(y_out, y_true);
}

/*
 * show detail architecture
*/
void FNN::summary()
{
	std::cout << "==========================" << std::endl;
	for (Layer * layer: m_layers)
	{
		std::cout << layer->toString() << std::endl;
	}
	std::cout << "==========================" << std::endl;
}

/*
 * save model weight to specific filepath
*/
void FNN::save(std::string fp)
{

	std::ofstream weightfile;
	weightfile.open(fp);

	for (Layer * layer: m_layers)
		layer -> saveWeight(weightfile);

	weightfile.close();
}


PYBIND11_MODULE(_SimpleNN, m) {

    py::class_<FNN>(m, "FNN")	
		.def(py::init<std::string>())
		.def("fit", & FNN::fit,
			py::arg("X_train"),
			py::arg("y_train"),
			py::arg("X_valid") = py::none(),
			py::arg("y_valid") = py::none(),
			py::arg("epochs") = 100,
			py::arg("lr") = 1e-2
		)
		.def("evaluate", & FNN::evaluate, py::return_value_policy::reference_internal)
		.def("getLoss", &FNN::getLoss)
		.def("backward", &FNN::backward)
		.def("summary", & FNN::summary)
		.def("save", & FNN::save);

		/*
		
	m.def("multiply_naive", & multiply_naive);
	m.def("multiply_mkl", & multiply_mkl);
	m.def("multiply_tile", & multiply_tile);
*/
} 

/*
 * To test layer
*/
PYBIND11_MODULE(_Layer, m) {
    py::class_<Dense>(m, "Dense")
        .def(py::init<size_t, size_t, std::string>())
        .def("forward", & Dense::forward, py::return_value_policy::reference_internal)
		.def("getw", &Dense::getWeight)
		.def("getb", &Dense::getBias);

	py::class_<ReLu>(m, "ReLu")
		.def(py::init<>())
		.def("forward", & ReLu::forward, py::return_value_policy::reference_internal);

	py::class_<Sigmoid>(m, "Sigmoid")
		.def(py::init<>())
		.def("forward", & Sigmoid::forward, py::return_value_policy::reference_internal);
}

/*
 * To test Loss
*/
PYBIND11_MODULE(_Loss, m) {
    py::class_<CrossEntropyLoss>(m, "CrossEntropyLoss")
		.def(py::init<>())
		.def("cal_loss", &CrossEntropyLoss::cal_loss)
		.def("cal_grad", &CrossEntropyLoss::cal_grad);

	py::class_<MSELoss>(m, "MSELoss")
		.def(py::init<>())
		.def("cal_loss", &MSELoss::cal_loss)
		.def("cal_grad", &MSELoss::cal_grad);
}
