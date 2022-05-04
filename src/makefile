CXX = g++

pybindpkg = `python3 -m pybind11 --includes`
PY = /usr/include/python3.8
NAME = _SimpleNN`python3-config --extension-suffix`
testdir = ./tests
sampledir = ../sample

all:
	$(CXX) -shared -std=c++14 -fPIC ${pybindpkg} _SimpleNN.cpp -o ${NAME} -I ${PY}
	
FNN:	
	$(CXX) -shared -std=c++14 -fPIC ${pybindpkg} _SimpleNN.cpp -o _SimpleNN`python3-config --extension-suffix` -I /usr/include/python3.8
	python3 -m pytest -v ${testdir}/fnn.py
	rm -f *.so *.out
	rm -rf __pycache__

fnn:
	python3 -m pytest -v ${testdir}/fnn.py
	rm -f *.so *.out
	rm -rf __pycache__
	
Layer:
	$(CXX) -shared -std=c++14 -fPIC ${pybindpkg} _SimpleNN.cpp -o _Layer`python3-config --extension-suffix` -I /usr/include/python3.8
	python3 -m pytest -v ${testdir}/layer.py

run:
	cp ${testdir}/fnn.py fnn.py
	python3 fnn.py
	rm fnn.py

test:
	$(CXX) -shared -std=c++14 -fPIC ${pybindpkg} _testNN.cpp -o _testNN`python3-config --extension-suffix` -I ${PY}
	python3 -m pytest -v ${testdir}/ftest.py


iris:
	$(CXX) -shared -std=c++14 -fPIC ${pybindpkg} _SimpleNN.cpp -o ${NAME} -I ${PY}
	cp ${sampledir}/test.py ./iris_test.py
	python3 iris_test.py
	rm iris_test.py

clean:
	rm -f *.so *.out *.txt
	rm -rf __pycache__
