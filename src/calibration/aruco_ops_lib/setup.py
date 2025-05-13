import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class ConanBuildExt(build_ext):
    def build_extension(self, ext):
        source_dir = os.path.abspath(os.path.dirname(__file__))
        build_dir = os.path.join(source_dir, "build")
        os.makedirs(build_dir, exist_ok=True)
        
        # Run conan install
        subprocess.check_call(
            ["conan", "install", source_dir, 
             "--output-folder", build_dir,
             "--build=missing"],
            cwd=build_dir
        )
        
        # Run CMake
        subprocess.check_call(
            ["cmake", source_dir,
             "-DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake",
             "-DCMAKE_BUILD_TYPE=Release"],
            cwd=build_dir
        )
        
        # Build
        subprocess.check_call(
            ["cmake", "--build", "."],
            cwd=build_dir
        )
        
        # Get the name of the output library
        ext_path = self.get_ext_fullpath(ext.name)
        ext_dir = os.path.dirname(ext_path)
        os.makedirs(ext_dir, exist_ok=True)
        
        # Copy the built library
        if sys.platform == "win32":
            built_lib = os.path.join(build_dir, "aruco_ops.pyd")
        else:
            built_lib = list(Path(build_dir).glob("*.so"))[0]
            
        if os.path.exists(built_lib):
            os.replace(built_lib, ext_path)

setup(
    name="aruco_ops",
    version="0.1",
    ext_modules=[
        Extension(
            "aruco_ops",  # This is the module name that will be used in Python
            []
        )
    ],
    cmdclass={
        "build_ext": ConanBuildExt,
    },
    install_requires=['opencv-python'],
    python_requires='>=3.6',
)