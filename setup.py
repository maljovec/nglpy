 ##############################################################################
 # Software License Agreement (BSD License)                                   #
 #                                                                            #
 # Copyright 2018 University of Utah                                          #
 # Scientific Computing and Imaging Institute                                 #
 # 72 S Central Campus Drive, Room 3750                                       #
 # Salt Lake City, UT 84112                                                   #
 #                                                                            #
 # THE BSD LICENSE                                                            #
 #                                                                            #
 # Redistribution and use in source and binary forms, with or without         #
 # modification, are permitted provided that the following conditions         #
 # are met:                                                                   #
 #                                                                            #
 # 1. Redistributions of source code must retain the above copyright          #
 #    notice, this list of conditions and the following disclaimer.           #
 # 2. Redistributions in binary form must reproduce the above copyright       #
 #    notice, this list of conditions and the following disclaimer in the     #
 #    documentation and/or other materials provided with the distribution.    #
 # 3. Neither the name of the copyright holder nor the names of its           #
 #    contributors may be used to endorse or promote products derived         #
 #    from this software without specific prior written permission.           #
 #                                                                            #
 # THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR       #
 # IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES  #
 # OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.    #
 # IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,           #
 # INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT   #
 # NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,  #
 # DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY      #
 # THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT        #
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF   #
 # THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.          #
 ##############################################################################
"""
      Setup script for nglpy, a wrapper library for the C++ implementataion of
      the neighborhood graph library (NGL).
"""

from setuptools import setup, Extension
import re


def get_property(prop, project):
    """
        Helper function for retrieving properties from a project's
        __init__.py file
        @In, prop, string representing the property to be retrieved
        @In, project, string representing the project from which we will
        retrieve the property
        @Out, string, the value of the found property
    """
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project + '/__init__.py').read())
    return result.group(1)

FILES = ['ngl_wrap.cpp', 'GraphStructure.cpp', 'UnionFind.cpp']
VERSION = get_property('__version__', 'nglpy')

def long_description():
    """ Reads the README.md file and extracts the portion tagged between
        specific LONG_DESCRIPTION comment lines.
    """
    description = ''
    recording = False
    with open('README.md') as f:
        for line in f:
            if 'END_LONG_DESCRIPTION' in line:
                return description
            elif 'LONG_DESCRIPTION' in line:
                recording = True
                continue

            if recording:
                description += line
            

## Consult here: https://packaging.python.org/tutorials/distributing-packages/
setup(name='nglpy',
      packages=['nglpy'],
      version=VERSION,
      description='A wrapper library for exposing the C++ neighborhood graph '
                  + 'library (NGL) for computing empty region graphs to python',
      long_description=long_description(),
      author = 'Dan Maljovec',
      author_email = 'maljovec002@gmail.com',
      license = 'BSD',
      test_suite='nglpy.tests',
      url = 'https://github.com/maljovec/nglpy',
      download_url = 'https://github.com/maljovec/nglpy/archive/'+VERSION+'.tar.gz',
      keywords = ['geometry', 'neighborhood', 'empty region graph',
                  'neighborhood graph library', 'beta skeleton',
                  'relative neighbor', 'Gabriel graph'],
      ## Consult here: https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: C++',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Mathematics'
      ],
      install_requires=['scikit-learn'],
      python_requires='>=2.7, <4',
      ext_modules=[Extension('_ngl',
                             FILES,
                             extra_compile_args=['-std=c++11'])])
