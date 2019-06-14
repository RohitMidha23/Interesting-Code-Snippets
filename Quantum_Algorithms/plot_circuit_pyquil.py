# This specific file is for quantum circuits implemented with pyQuil
# Their Forest SDK doesn't have a visualisation for circuits as yet, hence the code
from pyquil import Program
from pyquil.gates import H
from pyquil.latex import to_latex
import matplotlib.pyplot as plt
import numpy as np
import shutil
import subprocess
from tempfile import mkdtemp


def plot_circuit(circuit):
    latex_diagram = to_latex(circuit)
    tmp_folder = mkdtemp()
    with open(tmp_folder + '/circuit.tex', 'w') as f:
        f.write(latex_diagram)
    proc = subprocess.Popen(['pdflatex', '-shell-escape', tmp_folder + '/circuit.tex'], cwd=tmp_folder)
    proc.communicate()
    image = plt.imread(tmp_folder + '/circuit.png')
    shutil.rmtree(tmp_folder)
    plt.axis('off')
    return plt.imshow(image)
    
circuit = Program()
circuit += H(0)

plot_circuit(circuit)
