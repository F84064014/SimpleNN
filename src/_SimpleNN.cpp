//#define EIGEN_USE_MKL_ALL
//#define EIGEN_VECTORIZE_SSE4_2

#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <ctime>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include "Layer.hpp"
#include "Loss.hpp"
#include "Timer.hpp"

namespace py = pybind11;

// set Eigen Matrix row major (default column major)
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXd = Eigen::VectorXd;

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
 * flatten an n*p probability value to a nd vector
*/
VectorXd flatten_output(const Eigen::Ref<RowMatrixXd> y)
{
	VectorXd ret = VectorXd(y.rows());
	for (size_t i=0; i<y.rows(); ++i)
	{
		size_t max_of_col = 0;
		for (size_t j=1; j<y.cols(); ++j)
		{
			if (y(i, j) > y(i, max_of_col))
				max_of_col = j;
		}
		ret(i) = max_of_col;
	}
	return ret;
}

/*
 * FNN class
*/

class FNN{

public:

	FNN(std::string);
	RowMatrixXd forward(const Eigen::Ref<RowMatrixXd>, bool set_train);
	void fit(
		Eigen::Ref<RowMatrixXd> X_train,
		Eigen::Ref<VectorXd> y_train,
		Eigen::Ref<RowMatrixXd> X_valid,
		Eigen::Ref<VectorXd> y_valid,
		size_t epochs, 
		double lr //learning rate
	);
	void fit(
		Eigen::Ref<RowMatrixXd> X_train,
		Eigen::Ref<VectorXd>,
		size_t epochs, 
		double lr //learning rate
	);
	RowMatrixXd evaluate(const Eigen::Ref<RowMatrixXd> X);
	double getLoss(const Eigen::Ref<RowMatrixXd> y_out, const Eigen::Ref<VectorXd> y_true);
	RowMatrixXd getGrad(const Eigen::Ref<RowMatrixXd> y_out, const Eigen::Ref<VectorXd> y_true);
	double getAccuracy(const Eigen::Ref<VectorXd> y_out, const Eigen::Ref<VectorXd> y_true);
	void summary();
	void save(std::string);
	void load(std::string);
	void backward(const Eigen::Ref<RowMatrixXd>, const Eigen::Ref<VectorXd>);

protected:

	bool m_trainFlag;
	std::vector<Layer*> m_layers; //better or not??
	std::vector<RowMatrixXd> m_layer_inputs; //better or not??
	LossFunc* m_lossfunc;
	std::string m_linesplitter = std::string(70, '_');
};

/*
 *	set input/output dimension and layer configuration
*/
FNN::FNN(std::string config_str)
{

	std::srand(std::time(nullptr));

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
			
			Dense * new_layer_ptr = new Dense(input_dim, output_dim, activation, 0.1, int(rand()*100));
			m_layers.push_back(new_layer_ptr);
			
		}
		// CrossEntropy loss
		else if (config_unit_vec.at(0) == "CrossEntropy")
		{
			m_lossfunc = new CrossEntropyLoss();
		}
		else if (config_unit_vec.at(0) == "MSE")
		{
			m_lossfunc = new MSELoss();
		}
	}
}

/*
 *	forward data to each layer
*/
RowMatrixXd FNN::forward(const Eigen::Ref<RowMatrixXd> x, bool set_train = true)
{
	// copy x, dont know if there is better way
	RowMatrixXd _x(x);
	
	// record layer inputs for updating parameter
	if (set_train == true)
	{
		// push back pass by value (copy)
		m_layer_inputs.push_back(_x);
	}

	for (Layer * layer: m_layers)
	{
		_x = layer->forward(_x);

		if (set_train == true)
			m_layer_inputs.push_back(_x);

	}

	
	return _x;
}


/*
 * train/evaluate input x with model parameter
 * update parameter through X_trian, y_train
 * evaluate model performance with X_valid, y_valid
*/
void FNN::fit(
	Eigen::Ref<RowMatrixXd> X_train,
	Eigen::Ref<VectorXd> y_train,
	Eigen::Ref<RowMatrixXd> X_valid,
	Eigen::Ref<VectorXd> y_valid,
	size_t epochs = 100,
	double lr = 1e-1
)
{
	std::cout << "training set size: " << X_train.rows() << std::endl;
	std::cout << "testing set size: " << X_valid.rows() << std::endl;

	for (Layer* layer: m_layers)
		layer->set_lr(lr);

	Timer timer;
	double epoch_duration;

	py::print(m_linesplitter);
	for (size_t epoch=0; epoch<epochs; ++epoch)
	{

		timer.start();
		m_layer_inputs.clear();

		RowMatrixXd y_out = forward(X_train);
		
		double train_loss = getLoss(y_out, y_train);
		backward(y_out, y_train);

		VectorXd y_out_flat = flatten_output(y_out);
		double train_acc = getAccuracy(y_out_flat, y_train);

		epoch_duration = timer.end();

		py::print("epoch", epoch+1, "/", epochs);
		train_acc = round(train_acc * 100) / 100.0;
		train_loss = round(train_loss * 1000) / 1000.0;
		py::print("train accuracy", train_acc, "| train loss:", train_loss, "| epoch time:", epoch_duration, "ms");

		// validation part
		RowMatrixXd y_out_valid = forward(X_valid, false);
		double valid_loss = getLoss(y_out_valid, y_valid);
		VectorXd y_out_flat_valid = flatten_output(y_out_valid);
		double valid_acc = getAccuracy(y_out_flat_valid, y_valid);

		valid_acc = round(valid_acc * 100) / 100.0;
		valid_loss = round(valid_loss * 1000) / 1000.0;
		py::print("valid accuracy", valid_acc, "| valid loss:", valid_loss);
		py::print(m_linesplitter);

	}

	double average_time = round(timer.get_average_time() * 1000) / 1000.0;
	py::print("average time cost for training", average_time, "ms");

}

/*
 * train/evaluate input x with model parameter
 * update parameter through X_train, y_train
 * no validation set
*/
void FNN::fit(
	Eigen::Ref<RowMatrixXd> X_train,
	Eigen::Ref<VectorXd> y_train,
	size_t epochs = 100,
	double lr = 1e-1
)
{

	py::print("taining set size", X_train.rows());
	py::print("testing = None");

	for (Layer* layer: m_layers)
		layer->set_lr(lr);
	

	//TODO add a dimension check here for X_train.cols() and m_layers[0].rows()

	Timer timer;
	double epoch_duration;

	py::print(m_linesplitter);
	for (size_t epoch=0; epoch<epochs; ++epoch)
	{

		timer.start();
		m_layer_inputs.clear();
		
		RowMatrixXd y_out = forward(X_train);
		
		double train_loss = getLoss(y_out, y_train);
		backward(y_out, y_train);

		VectorXd y_out_flat = flatten_output(y_out);
		double train_acc = getAccuracy(y_out_flat, y_train);

		epoch_duration = timer.end();

		py::print("epoch", epoch+1, "/", epochs);
		train_acc = round(train_acc * 100) / 100.0;
		train_loss = round(train_loss * 1000) / 1000.0;
		py::print("train accuracy", train_acc, "| train loss:", train_loss, "| epoch time:", epoch_duration, "ms");
		py::print(m_linesplitter);
	}

}

/*
 * evaluate the input x
*/
RowMatrixXd FNN::evaluate(const Eigen::Ref<RowMatrixXd> x)
{
	return forward(x, false);
}

/*
 * calculate loss
*/
double FNN::getLoss(const Eigen::Ref<RowMatrixXd> y_out, const Eigen::Ref<VectorXd> y_true)
{

	return m_lossfunc->cal_loss(y_out, y_true);
}

/*
 * calculate gradient from loss
 * y_out is a loss probability matrix with dim = n * num_class
*/
RowMatrixXd FNN::getGrad(const Eigen::Ref<RowMatrixXd> y_out, const Eigen::Ref<VectorXd> y_true)
{
	return m_lossfunc->cal_grad(y_out, y_true);
}

/*
 * calculate accuracy (classification
*/
double FNN::getAccuracy(const Eigen::Ref<VectorXd> y_out, const Eigen::Ref<VectorXd> y_true)
{

	if (y_out.size() != y_true.size())
		throw "incompitable size for output and label";

	unsigned int correct_cnt = 0;
	for (size_t i=0; i<y_out.rows(); ++i)
	{
		if ((int)y_out(i) == (int)y_true(i))
			correct_cnt += 1;
	}

	return (double)correct_cnt / y_out.rows();
}

/*
 * backward proporgation
*/
void FNN::backward(const Eigen::Ref<RowMatrixXd> y_out, const Eigen::Ref<VectorXd> y_true)
{
	RowMatrixXd loss_grad = m_lossfunc->cal_grad(y_out, y_true);

	auto it_layer = m_layers.rbegin();
	auto it_input = m_layer_inputs.rbegin();
	it_input++;
	
	for (size_t i=0; i<m_layers.size(); ++i)
	{
		loss_grad = (*it_layer)->backward((*it_input), loss_grad);
		it_layer++;
		it_input++;
	}
}

/*
 * show detail architecture
*/
void FNN::summary()
{
	py::print("==========================");
	for (Layer * layer: m_layers)
	{
		py::print(layer->toString());
	}
	py::print("==========================");
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
		.def("fit", py::overload_cast<Eigen::Ref<RowMatrixXd>, Eigen::Ref<VectorXd>, Eigen::Ref<RowMatrixXd>, Eigen::Ref<VectorXd>, size_t, double>(& FNN::fit),
			py::arg("X_train"),
			py::arg("y_train"),
			py::arg("X_valid") = py::none(),
			py::arg("y_valid") = py::none(),
			py::arg("epochs") = 100,
			py::arg("lr") = 1e-2
		)
		.def("fit", py::overload_cast<Eigen::Ref<RowMatrixXd>, Eigen::Ref<VectorXd>, size_t, double>(& FNN::fit),
			py::arg("X_train"),
			py::arg("y_train"),
			py::arg("epochs") = 100,
			py::arg("lr") = 1e-2
		)
		.def("evaluate", & FNN::evaluate, py::return_value_policy::reference_internal)
		.def("getLoss", &FNN::getLoss)
		.def("getGrad", &FNN::getGrad)
		.def("backward", &FNN::backward)
		.def("summary", & FNN::summary)
		.def("save", & FNN::save);
}

/*
 * To test layer
*/
PYBIND11_MODULE(_Layer, m) {
    py::class_<Dense>(m, "Dense")
        .def(py::init<size_t, size_t, std::string>())
        .def("forward", & Dense::forward, py::return_value_policy::reference_internal)
		.def("backward", & Dense::backward, py::return_value_policy::reference_internal)
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
