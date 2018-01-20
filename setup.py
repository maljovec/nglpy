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
      Setup script for pyerg, a wrapper library for the C++ implementataion of
      the neighborhood graph library (NGL).
"""

from setuptools import setup, Extension

FILES = ['ngl_wrap.cpp', 'GraphStructure.cpp', 'UnionFind.cpp']
VERSION = '0.2.6'

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
setup(name='pyerg',
      packages=['pyerg'],
      version=VERSION,
      description='A wrapper library for exposing the C++ neighborhood graph '
                  + 'library (NGL) for computing empty region graphs to python',
      long_description=long_description(),
      author = 'Dan Maljovec',
      author_email = 'maljovec002@gmail.com',
      license = 'BSD',
      test_suite='pyerg.tests',
      url = 'https://github.com/maljovec/pyerg',
      download_url = 'https://github.com/maljovec/pyerg/archive/'+VERSION+'.tar.gz',
      keywords = ['geometry', 'neighborhood', 'empty region graph'],
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
