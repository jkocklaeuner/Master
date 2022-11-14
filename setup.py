import setuptools


setuptools.setup(

    name="nnqs-master", # Replace with your username

    version="1.0.0",

    author="jkocklaeuner",

    description="NNQS for Quantum Chemistry: Master project",

    url="<https://github.com/jkocklaeuner/Master.git>",

    packages=setuptools.find_packages(),
    
    install_requires=[
        "netket==3.4.2",
        "openfermion",
        "matplotlib",
        "jaxlib>=0.1, <0.3",	
         "jax==0.2.28" ]
        
)
