from setuptools import find_packages, setup

setup(
    name="mobilephones",
    packages=find_packages(exclude=["mobilephones_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
