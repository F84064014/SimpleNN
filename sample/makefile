all:
	python3 fetchMNIST.py
	python3 main.py
	make clean

iris:
	python3 main_iris.py

mod:
	cp ../src/_SimpleNN.cpython-38-x86_64-linux-gnu.so ./_SimpleNN.cpython-38-x86_64-linux-gnu.so 
	python3 test.py

clean:
	rm -rf MNIST __pycache__
