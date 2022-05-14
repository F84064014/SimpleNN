CXX = g++

pybindpkg = `python3 -m pybind11 --includes`
PY = /usr/include/python3.8
NAME = _SimpleNN`python3-config --extension-suffix`
TESTDIR = ./tests
SAMPLEDIR = ./sample
MKL = -I /usr/include/mkl -L /usr/lib/x86_64-linux-gnu/mkl -lmkl_rt

all:
	$(CXX) -O3 -march=native -fopenmp -shared -std=c++14 -fPIC ${pybindpkg} _SimpleNN.cpp -o ${NAME} -I${PY} ${MKL}
	
FNN:	
	$(CXX) -shared -std=c++14 -fPIC ${pybindpkg} _SimpleNN.cpp -o _SimpleNN`python3-config --extension-suffix` -I /usr/include/python3.8
	python3 -m pytest -v ${TESTDIR}/fnn.py
	rm -f *.so *.out
	rm -rf __pycache__

fnn:
	python3 -m pytest -v ${TESTDIR}/fnn.py
	rm -f *.so *.out
	rm -rf __pycache__
	
Layer:
	$(CXX) -shared -std=c++14 -fPIC ${pybindpkg} _SimpleNN.cpp -o _Layer`python3-config --extension-suffix` -I /usr/include/python3.8
	python3 -m pytest -v ${TESTDIR}/layer.py

run:
	cp ${testdir}/fnn.py fnn.py
	python3 fnn.py
	rm fnn.py

test:
	$(CXX) -shared -std=c++14 -fPIC ${pybindpkg} _testNN.cpp -o _testNN`python3-config --extension-suffix` -I ${PY}
	python3 -m pytest -v ${TESTDIR}/ftest.py


iris:
	#$(CXX) -shared -std=c++14 -fPIC ${pybindpkg} _SimpleNN.cpp -o ${NAME} -I ${PY}
	cp ${SAMPLEDIR}/iris.py ./iris.py
	python3 iris.py
	rm iris.py

handwrite:
	cp ${SAMPLEDIR}/handwrite.py ./handwrite.py
	python3 handwrite.py
	rm handwrite.py

reg:
	cp ${SAMPLEDIR}/regression.py ./regression.py
	python3 regression.py
	rm regression.py

diabetes:
	cp ${SAMPLEDIR}/diabetes.py ./diabetes.py
	python3 diabetes.py
	rm diabetes.py
	

clean:
	rm -f *.so *.out *.txt
	rm -rf __pycache__
