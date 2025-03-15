from setuptools import setup, find_packages

setup(
    name="content-based-recommender",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A content-based recommendation system",
    keywords="recommender, content-based, filtering",
    url="https://github.com/yourusername/content-based-recommender",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
