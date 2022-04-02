CXX = g++

test = ./tests

all:
	$(CXX) -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` _SimpleNN.cpp -o _SimpleNN`python3-config --extension-suffix` -I /usr/include/python3.8
	
FNN:	
	$(CXX) -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` _SimpleNN.cpp -o _SimpleNN`python3-config --extension-suffix` -I /usr/include/python3.8
	python3 -m pytest -v ${test}/fnn.py
	rm -f *.so *.out
	rm -rf __pycache__

fnn:
	python3 -m pytest -v ${test}/fnn.py
	rm -f *.so *.out
	rm -rf __pycache__
	

Layer:
	$(CXX) -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` _SimpleNN.cpp -o _Layer`python3-config --extension-suffix` -I /usr/include/python3.8
	python3 -m pytest -v ${test}/layer.py

run:
	cp ${test}/fnn.py fnn.py
	python3 fnn.py
	rm fnn.py

clean:
	rm -f *.so *.out
	rm -rf __pycache__
