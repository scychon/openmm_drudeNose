from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform

openmm_dir = '@OPENMM_DIR@'
drudetgnhplugin_header_dir = '@DRUDETGNHPLUGIN_HEADER_DIR@'
drudetgnhplugin_library_dir = '@DRUDETGNHPLUGIN_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = []
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-std=c++11', '-mmacosx-version-min=10.7']
    extra_link_args += ['-std=c++11', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_drudetgnhplugin',
                      sources=['DrudeTGNHPluginWrapper.cpp'],
                      libraries=['OpenMM', 'OpenMMDrudeTGNH'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), drudetgnhplugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), drudetgnhplugin_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='drudetgnhplugin',
      version='1.0',
      py_modules=['drudetgnhplugin'],
      ext_modules=[extension],
     )
