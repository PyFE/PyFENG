from setuptools import setup

setup(
    name='pyfeng',
    version='0.1.0',
    description='Python Financial Engineering',
    author='Jaehyuk Choi',
    author_email='pyfe@eml.cc',
    url='https://github.com/PyFE/PyFENG',
    project_urls={
        "Bug Tracker": "https://github.com/PyFE/PyFENG/issues",
    },
    packages=['pyfeng'],
    install_requires=[
        'numpy', 'scipy'
    ]
)
