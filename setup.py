from setuptools import find_packages, setup

with open("requirements.txt") as req:
    requirements = req.readlines()   # might need to port it to list

setup(
    name='FSM',
    packages=find_packages(include=['finite_state_machine_lib']),
    version='0.1.2',
    description='My first Python library',
    author='Me',
    license='ME',
    install_requires=requirements,
    setup_requires=[],  # tror inte den här är nödvändig
    tests_require=[],   # skulle gissa att den här bara är specifikt för tester
    test_suite= '',
)
