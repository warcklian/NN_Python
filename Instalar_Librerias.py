import subprocess
import sys

# Lista de librerías a instalar
libraries = [
    "MetaTrader5",
    "pandas",
    "numpy",
    "requests",
    "tensorflow",
    "scikit-learn",
    "matplotlib",
    "keras-tuner",
    "seaborn",
    "joblib",
    "tqdm"
]

def install_package(package):
    """Instala un paquete usando pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} instalado correctamente.")
    except subprocess.CalledProcessError:
        print(f"Error al instalar {package}.")

def main():
    print("🔄 Instalando bibliotecas necesarias...")
    for lib in libraries:
        install_package(lib)
    print("Instalación completada.")

if __name__ == "__main__":
    main()
