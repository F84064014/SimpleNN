#include <iostream>
#include <string>
#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include "Matrix.h"
#include "Layer.h"

namespace py = pybind11;

// set Eigen Matrix row major (default column major)
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/*
 * split string into list of string
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

/*
 * FNN class
*/

class FNN{

public:

	FNN(std::string);
	Matrix forward(const Matrix);
	py::array_t<double> fit(
		py::array_t<double> X_train,
		py::array_t<double> X_valid,
		py::array_t<double> y_train,
		py::array_t<double> y_valid,
		size_t epochs, 
		double lr //learning rate
	);
	RowMatrixXd evaluate(Eigen::Ref<RowMatrixXd> X);
	void summary();

protected:
	bool m_trainFlag;
	std::list<Layer*> m_layers;
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
		if (config_unit_vec.at(0) == "linear")
		{
			size_t input_dim = std::stoi(config_unit_vec.at(1));
			size_t output_dim = std::stoi(config_unit_vec.at(2));
			std::string activation = config_unit_vec.at(3);
			Dense * new_layer_ptr = new Dense(input_dim, output_dim, activation);
			m_layers.push_back(new_layer_ptr);
		}
	}
}

/*
 *	forward data to each layer
*/
Matrix FNN::forward(const Matrix)
{
	printf("forward");
}

/*
 * train/evaluate input x with model parameter
 * X_train, y_train are not default 
*/
py::array_t<double> FNN::fit(
	py::array_t<double> X_train,
	py::array_t<double> y_train,
	py::array_t<double> X_valid,
	py::array_t<double> y_valid,
	size_t epochs = 100,
	double lr = 1e-2
)
{
	std::cout << "fit" << std::endl;
}

/*
 * evaluate the input x
*/
RowMatrixXd FNN::evaluate(Eigen::Ref<RowMatrixXd> X)
{

	Eigen::MatrixXd ret(2,2);
	ret(0,0) = 0;
	ret(1,0) = 1;
	ret(0,1) = 2;
	ret(1,1) = 3;
	std::cout << X << std::endl;
	
	return ret;
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
		.def("summary", & FNN::summary);

		/*
		.def_readonly("nrow", & Matrix::m_nrow)
		.def_readonly("ncol", & Matrix::m_ncol)
		.def("__assign__", & Matrix::operator=, py::is_operator())
		.def("__eq__", & operator==, py::is_operator())
		.def("__getitem__", [](Matrix &self, py::tuple tup){
			int row = py::cast<int>(tup[0]);
			int col = py::cast<int>(tup[1]);
			return self.m_buffer[self.index(row, col)];
		})
		.def("__setitem__", [](Matrix &self, py::tuple tup, size_t val){
			int row = py::cast<int>(tup[0]);
			int col = py::cast<int>(tup[1]);
			self.m_buffer[self.index(row, col)] = val;
		});
		
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
		.def("getw", &Dense::viewMatrix);

	py::class_<ReLu>(m, "ReLu")
		.def(py::init<>())
		.def("forward", & ReLu::forward, py::return_value_policy::reference_internal);

	py::class_<Sigmoid>(m, "Sigmoid")
		.def(py::init<>())
		.def("forward", & Sigmoid::forward, py::return_value_policy::reference_internal);
}
        
