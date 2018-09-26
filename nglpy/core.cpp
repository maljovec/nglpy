#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "ngl.h"
#include <cmath>

static PyObject* nglpy_core_prune(PyObject *self, PyObject *args, PyObject* kwargs) {
    //import_array();

    int N;
    int D;
    int M;
    int K;
    float lp = 2.0;
    float beta = 1.0;
    bool relaxed = false;
    PyArrayObject *X_arr;
    PyArrayObject *edges_arr;

    static char* argnames[] = {"X", "edges", "relaxed", "beta", "lp", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&|pff", argnames,
			                         PyArray_Converter, &X_arr,
				                     PyArray_Converter, &edges_arr,
				                     &relaxed,
				                     &beta,
				                     &lp))
        return NULL;

    npy_intp idx[2];
    idx[0] = idx[1] = 0;
    float *X = (float *)PyArray_GetPtr(X_arr, idx);
    int *edges = (int *)PyArray_GetPtr(edges_arr, idx);

    N = PyArray_DIM(X_arr, 0);
    D = PyArray_DIM(X_arr, 1);
    M = PyArray_DIM(edges_arr, 0);
    K = PyArray_DIM(edges_arr, 1);

    std::vector<int> edgeIndices;
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            if (edges[i*K+j] != -1) {
                edgeIndices.push_back(i);
                edgeIndices.push_back(edges[i*K+j]);
            }
        }
    }

    ngl::Geometry<float>::init(D);
    ngl::NGLPointSet<float> *P = new ngl::prebuiltNGLPointSet<float>(X, N, edgeIndices);
    ngl::NGLParams<float> params;
    params.param1 = beta;
    params.iparam0 = K;
    ngl::IndexType *indices = NULL;
    int edge_count = 0;

    if (relaxed) {
        if (lp == 1) {
            ngl::getRelaxedDiamondGraph<float>(*P, &indices, edge_count, params);
        }
        else {
            ngl::getRelaxedBSkeleton<float>(*P, &indices, edge_count, params);
        }
    }
    else {
        if (lp == 1) {
            ngl::getDiamondGraph<float>(*P, &indices, edge_count, params);
        }
        else {
            ngl::getBSkeleton<float>(*P, &indices, edge_count, params);
        }
    }

    delete P;
    PyObject* edge_list = PyList_New(edge_count);
    for(int i = 0; i < edge_count; i++)
    {
      int i1 = indices[2*i+0];
      int i2 = indices[2*i+1];
      float dist = 0;
      for(int d = 0; d < D; d++)
          dist += ((X[i1*D+d]-X[i2*D+d])*(X[i1*D+d]-X[i2*D+d]));

        PyObject* item = Py_BuildValue("(iif)", i1, i2, sqrt(dist));
        PyList_SetItem(edge_list, i, item);
    }

    Py_DECREF(X_arr);
    Py_DECREF(edges_arr);

    return edge_list;
}

static PyMethodDef nglpy_core_methods[] = {
    {"prune",(PyCFunction)nglpy_core_prune, METH_VARARGS|METH_KEYWORDS, ""},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "nglpy.core",
    "A Python wrapper to a C++-based implementation of the Neighborhood Graph Library (NGL).",
    -1,
    nglpy_core_methods
};

PyMODINIT_FUNC PyInit_core(){
    import_array();
    return PyModule_Create(&module_def);
}
