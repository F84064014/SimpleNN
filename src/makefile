CXX = g++

all:
	$(CXX) -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` _SimpleNN.cpp -o _SimpleNN`python3-config --extension-suffix` -I /usr/include/python3.8
	
test:	
	$(CXX) -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` _SimpleNN.cpp -o _SimpleNN`python3-config --extension-suffix` -I /usr/include/python3.8
	python3 -m pytest
	rm -f *.so *.out
	rm -rf __pycache__

Layer:
	$(CXX) -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` _SimpleNN.cpp -o _Layer`python3-config --extension-suffix` -I /usr/include/python3.8
	python3 -m pytest

clean:
	rm -f *.so *.out
	rm -rf __pycache__
