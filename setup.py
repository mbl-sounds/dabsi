from setuptools import setup
from Cython.Build import cythonize

# force=True regenerates the C sources on every build so stale .c files are never compiled.
setup(
    ext_modules=cythonize(
        [
            "lib/centralized_cy.pyx",
            "lib/distgrad_cy.pyx",
            "lib/dsaawet_cy.pyx",
            "lib/simulation_utils_cy.pyx",
        ],
        force=True,
        compiler_directives={"language_level": 3},
    )
)
