#!/usr/bin/python3
import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=["./"])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build"
                + " the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            "-DBB_HARVESTER_ONLY=ON",
            "-DCMAKE_BUILD_TYPE=" + cfg
        ]
        build_args = ["--config", cfg, "--target", "bladebit_harvester"]
        install_args = ["--prefix", str(extdir)]

        if platform.system() == "Windows":
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            build_args += ["--", "-j", "6"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--install", "."] + install_args, cwd=self.build_temp
        )


ext_modules = [
    Extension(
        "bladebit",
        [
            "src/harvesting/GreenReaper.cpp",
            "src/harvesting/GreenReaper.h",
            "src/harvesting/GreenReaperPortable.h",
        ],
        include_dirs=[
            "src",
            ".",
        ],
    ),
]

setup(
    name="bladebit",
    author="Chia Network",
    author_email="hello@chia.net",
    description="A high-performance **k32-only**, Chia (XCH) plotter",
    license="Apache License",
    python_requires=">=3.7",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Chia-Network/bladebit",
    ext_modules=[CMakeExtension("bladebit", ".")],
    cmdclass=dict(build_ext=CMakeBuild),
    package_data={"": ["*.h"]},
    zip_safe=False,
)
