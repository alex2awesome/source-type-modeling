from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


ext_modules=[
	Extension(
		"sampler_cy",
		["sampler_cy.pyx"],
		# libraries=["m"],
		extra_compile_args=[
			# '-march=i686',
			# '-O3',
			# '-ffast-math',
			# '-fopenmp',
			# '/Ox',
			# '/openmp',
			# '/fp:fast'
		],
		# extra_link_args=['-fopenmp']
	)
]

setup(
  name = "sampler_cy",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules,
  include_dirs=[numpy.get_include()]
)



# setup(
#     ext_modules=cythonize([
#     	"sampler.pyx",
#     	"sampler_cy.pyx"
#    	], annotate=True,
# 	# extra_compile_args = ["-ffast-math"]
# 	),
#    	include_dirs=[numpy.get_include()],
# )