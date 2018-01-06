## This can be added to a pre-commit hook to ensure that the wrappers build
## successfully before each commit
all:
	swig -python -c++ -o ngl_wrap.cpp ngl.i
	mv ngl.py pyerg/ngl.py

clean:
	rm ngl_wrap.cpp pyerg/ngl.py