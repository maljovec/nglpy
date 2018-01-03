%module ngl
%include "std_vector.i"
%include "std_set.i"
%include "stl.i"

%{
#include "GraphStructure.h"
%}
%include "GraphStructure.h"

%template(nglGraph) GraphStructure<double>;

namespace std
{
  %template(vectorDouble) vector<double>;
  %template(vectorInt) vector<int>;
  %template(setInt) set<int>;
}
