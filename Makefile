## This can be added to a pre-commit hook to ensure that the wrappers build
## successfully before each commit
all:
	swig -v -python -c++ -o ngl_wrap.cpp ngl.i
	mv ngl.py nglpy/ngl.py

clean:
	rm ngl_wrap.cpp nglpy/ngl.py
