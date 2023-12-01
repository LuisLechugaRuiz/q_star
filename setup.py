from setuptools import setup, find_packages

setup(
    name='q_star',
    version='0.0.1',
    author="Luis Lechuga Ruiz",
    author_email="luislechugaruiz@gmail.com",
    description="Q* Algorithm to plan using LLMs.",
    long_description=open('README.md').read(),
    url="https://github.com/LuisLechugaRuiz/q_star",
    packages=find_packages(),
    license="MIT",
    classifiers=[
        ##
    ],
    install_requires=[
        # list your package dependencies here
        # 'numpy>=1.18.0',
        # 'pandas>=1.0.0',
    ],
    python_requires='>=3.6',
)
