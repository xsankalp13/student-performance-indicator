from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]:
    '''This function will return the list of requirements'''
    reqs = []
    with open(file_path) as file_obj:
        reqs=file_obj.readlines()
        reqs=[req.replace("\n", "") for req in reqs]
        if HYPEN_E_DOT in reqs:
            reqs.remove(HYPEN_E_DOT)
    return reqs

setup(
    name = "Student Performance Indicator",
    version="0.0.1",
    author="Sankalp Salve",
    author_email="sankalpbsalve@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt"),


)