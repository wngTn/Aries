from conan import ConanFile
from conan.tools.cmake import cmake_layout


class ExampleRecipe(ConanFile):
    settings = "os", "compiler", "build_type"  # , "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("opencv/4.10.0")
        self.requires("pybind11/2.11.1")
        self.requires("nlohmann_json/3.11.3")  # Add nlohmann_json requirement

    def layout(self):
        cmake_layout(self)
