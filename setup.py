from setuptools import setup, find_packages

setup(
    name="agente_ia_oyp",
    version="6.0.0",
    packages=find_packages(),
    install_requires=[
        # Dependencias principales
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "sqlalchemy>=1.4.0",
        "httpx>=0.23.0",
        "python-dotenv>=0.19.0",
        "pydantic>=1.8.0",
    ],
    # Incluir archivos de datos no-Python
    include_package_data=True,
    # Metadatos
    author="Equipo de Desarrollo Agente IA OyP",
    author_email="desarrollo@agenteiaoyp.com",
    description="Sistema de Agente Inteligente para Operaciones y Procesos",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tu-usuario/agente-ia-oyp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
