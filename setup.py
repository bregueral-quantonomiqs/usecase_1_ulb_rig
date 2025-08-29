from setuptools import setup, find_packages

setup(
    name="insurance_analysis",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.12.0",
    install_requires=[
        # Core data science and analytics
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.8.0",
        
        # Visualization
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        
        # Quantum computing
        "qiskit>=2.0.0",
        "qiskit-aer>=0.12.0",
        "pylatexenc>=2.10",
        "imbalanced-learn>=0.11.0",
        "smogn>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.2",
            "jupyter>=1.0.0",
        ],
    },
    author="Borja Regueral",
    author_email="bregueral@quantonomiqs.com",
    description="Insurance use case analysis project",
    keywords="insurance, data analysis, machine learning",
)