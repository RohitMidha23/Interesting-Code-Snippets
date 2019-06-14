import matplotlib.pyplot as plt
import numpy as np



def plot_complex(a):
    for x in range(len(a)):
        plt.plot([0,a[x].real], [0,a[x].imag], 'r-o')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    limit = np.max(np.ceil(np.absolute(a)))
    plt.xlim((-limit,limit))
    plt.ylim((-limit,limit))
    plt.show()

x = 2.3 - 14j
plot_complex([x])
