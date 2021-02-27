from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pyfeng',
    version='0.1.0',
    description='Python Financial Engineering',
    long_description=long_description,
    long_description_content_type="text/markdown",
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
