from Spectre import Spectre
from utilitaire import *
import numpy as np

# TODO : Ajouter le traitement des incertitudes.

def main():
    spectre = Spectre("spectres\calib_totale.Spe", 12)
    spectre.plot_spectrum(_mk='x', fits=True)
    spectre.plot_peaks(multiple=True, fits=True)
    theoValues = [122, 662, 1173, 1332]
    etalonnage(spectre, theoValues, show=True)

main()