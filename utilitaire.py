import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
from Spectre import Spectre

def etalonnage(spec: Spectre, theo, show=False):
    xdata = []
    # Fonction linéaire qu'on chercher à fitter
    f = lambda x, a, b: a*x + b

    # Itérations sur les photopics dont l'énergie théorique est dans theo
    for peak in spec.get_Peaks():
        # Le centroid du photopic
        xdata.append(peak[1][1])
    
    # Numpy array pour faciliter le traitement
    xdata = np.array(xdata)
    ydata = np.array(theo)

    # Curve fit
    popt, pcov = sp.curve_fit(f, xdata, ydata)
    fitData = f(xdata, *popt)

    # Calculer R^2
    residuals = ydata- f(xdata, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    r_squared = 1 - (ss_res / ss_tot)

    if show:
        fig, ax = plt.subplots()
        # Titres et labels d'axes
        ax.set_title("Étalonnage du détecteur NaI(Tl)")
        ax.set_xlabel("Canaux (-)")
        ax.set_ylabel("Énergie (keV)")

        # Ticks
        ax.tick_params(direction='in', length=3, width=1, colors='k',
                        grid_color='k', grid_alpha=0.5)
        
        # Équation et R^2
        ax.text(500, 1000, f"E = {popt[0]:.2f}$\cdot$Ch + {popt[1]:.2f}", size=15)
        ax.text(500, 900, f"$R^2 = ${r_squared:.4f}", size=15)

        # Data et fit
        ax.scatter(xdata, ydata, s=60, c="g", marker="^")
        ax.plot(xdata, fitData, "r--")
        plt.show()
    return popt, pcov, fitData

