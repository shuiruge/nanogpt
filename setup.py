from setuptools import setup, find_packages


NAME = 'nanogpt'
AUTHOR = 'shuiruge'
AUTHOR_EMAIL = 'shuiruge@whu.edu.cn'
URL = 'https://github.com/shuiruge/nanogpt'
VERSION = '0.0.1'


setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages=find_packages(exclude=[
        'tests.*', 'tests',
        'data.*', 'data']),
    classifiers=[
        'Programming Language :: Python :: 3+',
    ],
    zip_safe=False,
)
