import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
import csv
import pandas as pd


class Spectre():
    """
    Classe qui encapsule un spectre de désintégration gamma.
    """
    def __init__(self, filename, header):
        self.filename = filename
        # self.infos est un dictionnaire de la forme {"rois": rois, "date": date, "longueur": longueur}
        data, self.infos = self.read_file(header)
        # Données en count et en channel du spectre
        self.ydata = np.array(data)
        self.xdata = list(range(0, self.infos["longueur"] + 1))
        self.rois_infos = []

        # Boucle sur les rois afin de calculer les fits et stocker les infos
        for roi in self.infos["rois"]:
            popt, pcov, fit_data = self.compute_fit(self.gauss, roi[0], roi[1], False)
            self.rois_infos.append(((roi[0], roi[1]), popt, pcov, fit_data))
        
        #self.rois_info est une liste de tuple. Chaqur tuple correspond à 1 roi et est de la forme
        # ( (start, stop), popt_du_fit, pcov_du_fit, fit_data)

        self.nb_rois = len(self.rois_infos)

    def read_file(self, header):
        """Fonction qui lit le fichier du spectre et en extrait les informations pertinentes.

        Args:
            header (int): Nombre de lignes en haut du fichier .spe

        Returns:
            tuple : Les données en y ainsi qu'un dictionnaire qui comprends les rois,
                la date et la longueur du fichier.
        """
        p = open(self.filename, "r")
        lines = p.readlines()
        data = []
        rois = []
        
        # Boucle sur les lignes
        for index, line in enumerate(lines):
            # On isole la date de prise du spectre
            if line == "$DATE_MEA:\n":
                date = lines[index + 1]
            # On vient chercher les rois
            if line == "$ROI:\n":
                nb_rois = int(lines[index + 1])
                next_line = lines[index + 2]
                i = 0
                while next_line != "$PRESETS:\n":
                    rois.append( ( int(next_line.split(" ")[0]), int(next_line.split(" ")[1][:-1] ) ) )
                    i += 1
                    next_line = lines[index + 2 + i]
            if line == "$DATA:\n":
                longueur = int(lines[index + 1].split(" ")[1])
        # Les données sont extraites avec numpy
        data = np.genfromtxt(self.filename, skip_header=header, skip_footer =14+nb_rois)
        return data, {"rois": rois, "date": date, "longueur": longueur}
    
    def plot_spectrum(self, _mk="o", _ms = 0.5, _color="r", fits=False):
        """Affiche le spectre complet.

        Args:
            _mk (str, optional): Type de marqueur utilisé. Defaults to "o".
            _ms (float, optional): Taille des marqueurs. Defaults to 0.5.
            _color (str, optional): Couleur des marqueurs. Defaults to "r".
            fits (bool, optional): Option d'afficher ou non les fits gaussien sur le spectre. Defaults to False.
        """
        plt.scatter(self.xdata, self.ydata, s=_ms, c=_color, marker=_mk)
        if fits:
            for roi in self.rois_infos:
                start, stop = roi[0][0], roi[0][1]
                xdata = self.get_xdata(start, stop)
                plt.plot(xdata, roi[3])
        plt.show()
            
    def compute_fit(self, func, roi1, roi2, show=False):
        """Calcule les fits gaussien des différents rois du graphe

        Args:
            func (fonction): fonction à fitter.
            roi1 (int): Début du ROI.
            roi2 (int): Fin du ROI.
            show (bool, optional): Option d'afficher ou non les fits une fois calculés. Defaults to False.

        Returns:
            tuple: Les paramètress optimaux du fit, la covariance de ceux-ci et les données fités.
        """
        popt, pcov = sp.curve_fit(func, self.xdata[roi1:roi2], self.ydata[roi1:roi2], p0=[100, (roi2+roi1)/2, (roi2-roi1)/2, 0])
        fit_data = func(self.xdata[roi1:roi2], *popt)
        if show:
            fig, ax = plt.subplots()
            ax.scatter(self.xdata[roi1: roi2], self.ydata[roi1: roi2], s=0.5, c="r", marker="o")
            ax.plot(self.xdata[roi1: roi2], fit_data, '-')
            fig.legend(["Données expérimentales", "Fit gaussien"])
            plt.show()
        return popt, pcov, fit_data

    def plot_peaks(self, multiple=True, fits=False):
        """Affiches les ROIs.

        Args:
            multiple (bool, optional): Option d'afficher les ROIs sur une même figure. Defaults to True.
            fits (bool, optional): Option d'afficher les fits sur les ROIs. Defaults to False.
        """
        if multiple:
            fig, axs = plt.subplots(ncols=self.nb_rois, nrows=1)
            for roi in range(self.nb_rois):
                start, stop = self.rois_infos[roi][0][0], self.rois_infos[roi][0][1]
                xdata = self.get_xdata(start, stop)
                axs[roi].scatter(xdata, self.ydata[start: stop], s=0.5, marker="o", c="red")
                if fits:
                    ydata = self.rois_infos[roi][3]
                    axs[roi].plot(xdata, ydata, color="blue")
            plt.show()
        else:
            for roi in range(self.nb_rois):
                fig, ax = plt.subplots()
                start, stop = self.rois_infos[roi][0][0], self.rois_infos[roi][0][1]
                xdata = self.get_xdata(start, stop)
                ax.scatter(xdata, self.ydata[start: stop], s=0.5, marker="o", c="red")
                if fits:
                    ydata = self.rois_infos[roi][3]
                    ax.plot(xdata, ydata, color="blue")
                plt.show()
        
    def get_xdata(self, init, stop):
        """Permet d'avoir accès à seulement une partie des données en x du spectre.

        Args:
            init (_type_): Début de la région.
            stop (_type_): Fin de la région.

        Returns:
            np.array(): Les données en x coupés.
        """
        return self.xdata[init:stop]
    
    def get_Peaks(self):
        """Retourne l'information sur les pics

        Returns:
            liste: Informations sur les différents ROIs.
        """
        return self.rois_infos
    
    @staticmethod
    def gauss(x, A, m, s, B):
        return A*np.exp(-(x - m)**2 / (2*s**2) + B)