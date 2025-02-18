from conan import ConanFile
from conan.tools.cmake import CMake

class ArucoOpsConan(ConanFile):
    name = "aruco_ops"
    version = "0.1"
    settings = "os", "compiler", "build_type" # , "arch"
    generators = "CMakeToolchain", "CMakeDeps"

    def requirements(self):
        self.requires("opencv/[>=4.5.0]")
        self.requires("pybind11/2.11.1")
        self.requires("eigen/3.4.0")  # Added Eigen requirement

    def build_requirements(self):
        self.tool_requires("cmake/[>=3.22.0]")

    def layout(self):
        self.folders.source = "."
        self.folders.build = "build"