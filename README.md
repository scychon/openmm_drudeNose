OpenMM Drude Dual Nose-Hoover Integrator Plugin
=====================

This plugin enables applying duan Nose-Hoover thermostat to a extended Lagrangian-scheme
MD simulate with drude-polarizable force field.

Building The Plugin
===================

This project uses [CMake](http://www.cmake.org) for its build system.  To build it, follow these
steps:

1. Create a directory in which to build the plugin.

2. Set environmental variables such as CXXFLAGS='-std=c++11', OPENMM_CUDA_COMPILER=$(which nvcc)

3. Run the CMake GUI or ccmake, specifying your new directory as the build directory and the top
level directory of this project as the source directory.

4. Press "Configure".

5. Set OPENMM_DIR to point to the directory where OpenMM is installed.  This is needed to locate
the OpenMM header files and libraries.

6. Set CMAKE_INSTALL_PREFIX to the directory where the plugin should be installed.  Usually,
this will be the same as OPENMM_DIR, so the plugin will be added to your OpenMM installation.

7. If you plan to build the CUDA platform, make sure that CUDA_TOOLKIT_ROOT_DIR is set correctly
and that EXAMPLE_BUILD_CUDA_LIB is selected.

8. Only CUDA Platform is supported

9. Press "Configure" again if necessary, then press "Generate".

10. For better performance, modify OpenMM src file $OPENMM_SRC_DIR/plugins/drude/openmmapi/include/openmm/internal/DrudeForceImpl.h in the following line. We suggested Peter to change the plugin code, but this is not corrected yet (for OpenMM 7.2)
-    void updateContextState(ContextImpl& context) {
+    void updateContextState(ContextImpl& context, bool& forcesInvalid) {

11. Use the build system you selected to build and install the plugin.
Performing three sets of commands 'make / make install / make PythonInstall' will install the plugin
as 'drudenoseplugin' package in your python



Test Cases
==========

Due to the issues with dynamic loading of regular Drude Kernel vs. static loading of current kernel, `make test` is not supported currently. One could instead use example scripts and codes under `example` directory.

OpenCL and CUDA Kernels
=======================

The OpenCL and CUDA platforms compile all of their kernels from source at runtime.  This
requires you to store all your kernel source in a way that makes it accessible at runtime.  That
turns out to be harder than you might think: simply storing source files on disk is brittle,
since it requires some way of locating the files, and ordinary library files cannot contain
arbitrary data along with the compiled code.  Another option is to store the kernel source as
strings in the code, but that is very inconvenient to edit and maintain, especially since C++
doesn't have a clean syntax for multi-line strings.

This project (like OpenMM itself) uses a hybrid mechanism that provides the best of both
approaches.  The source code for the OpenCL and CUDA implementations each include a "kernels"
directory.  At build time, a CMake script loads every .cl (for OpenCL) or .cu (for CUDA) file
contained in the directory and generates a class with all the file contents as strings.  For
example, the OpenCL kernels directory contains a single file called exampleForce.cl.  You can
put anything you want into this file, and then C++ code can access the content of that file
as `OpenCLExampleKernelSources::exampleForce`.  If you add more .cl files to this directory,
correspondingly named variables will automatically be added to `OpenCLExampleKernelSources`.


Python API
==========

OpenMM uses [SWIG](http://www.swig.org) to generate its Python API.  SWIG takes an "interface
file", which is essentially a C++ header file with some extra annotations added, as its input.
It then generates a Python extension module exposing the C++ API in Python.

When building OpenMM's Python API, the interface file is generated automatically from the C++
API.  That guarantees the C++ and Python APIs are always synchronized with each other and avoids
the potential bugs that would come from have duplicate definitions.  It takes a lot of complex
processing to do that, though, and for a single plugin it's far simpler to just write the
interface file by hand.  You will find it in the "python" directory.

To build and install the Python API, build the "PythonInstall" target, for example by typing
"make PythonInstall".  (If you are installing into the system Python, you may need to use sudo.)
This runs SWIG to generate the C++ and Python files for the extension module
(ExamplePluginWrapper.cpp and exampleplugin.py), then runs a setup.py script to build and
install the module.  Once you do that, you can use the plugin from your Python scripts:

    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
    from drudenoseplugin import DrudeNoseHooverIntegrator
    integ = DrudeNoseHooverIntegrator(temperature, REALFREQ*picosecond, 1*kelvin, DRUDEFREQ*picosecond, 0.001*picoseconds, 20)

If you wish to use independent temperature group for specific atoms,
you may add the following lines to your python script to add new atomgroup and add atoms to the atomgroup.

    YOUR_INTEGRATOR.addTempGroup()

    for ATOMS in YOUR_SYSTEM:
        YOUR_INTEGRATOR.addParticleTempGroup(GROUPIDX)

Here, you have to set addParticleTempGroup for each atoms in your system.
Note that when you add a tempgroup by addTempGroup(), 
the default tempgroup would have idx 0 and the added tempgroup would have idx 1.
When there's bond constraints, atoms under the constraint should be assigned to the same tempgroup.


License
=======

This is a plugin designed to work with OpenMM molecular simulation toolkit

Portions copyright (c) 2017 the Authors.

Authors: Chang Yun Son

Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.

