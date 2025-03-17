import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class RateInstrument(ABC):
    """ Classe abstraite pour les instruments de taux """
    
    @abstractmethod
    def get_rate(self):
        pass
    
    @abstractmethod
    def get_maturity(self):
        pass


class Deposit(RateInstrument):
    """ Modélise un dépôt interbancaire """
    
    def __init__(self, rate, maturity):
        self.rate = rate
        self.maturity = maturity
    
    def get_rate(self):
        return self.rate
    
    def get_maturity(self):
        return self.maturity


class FRA(RateInstrument):
    """ Modélise un Forward Rate Agreement """
    
    def __init__(self, rate, start_period, maturity):
        self.rate = rate
        self.start_period = start_period
        self.maturity = maturity
    
    def get_rate(self):
        return self.rate
    
    def get_maturity(self):
        return self.maturity


class IRS(RateInstrument):
    """ Modélise un Interest Rate Swap """
    
    def __init__(self, fixed_rate, maturity, frequency="Annual"):
        self.fixed_rate = fixed_rate
        self.maturity = maturity
        self.frequency = frequency
    
    def get_rate(self):
        return self.fixed_rate
    
    def get_maturity(self):
        return self.maturity


class YieldCurve:
    """ Classe pour construire la courbe des taux avec interpolation quadratique """
    
    def __init__(self):
        self.instruments = []
        self.coeffs = None  # Stocker les coefficients du polynôme
    
    def add_instrument(self, instrument):
        """ Ajoute un instrument à la courbe """
        self.instruments.append(instrument)
    
    def build_curve(self):
        """ Tri des instruments selon leur maturité et ajustement quadratique """
        self.instruments.sort(key=lambda inst: inst.get_maturity())
        
        maturities = np.array([inst.get_maturity() for inst in self.instruments])
        rates = np.array([inst.get_rate() for inst in self.instruments])
        
        if len(maturities) >= 3:
            # Ajustement d'un polynôme de degré 2
            self.coeffs = np.polyfit(maturities, rates, 2)
        else:
            raise ValueError("Au moins 3 instruments sont nécessaires pour une interpolation quadratique.")
    
    def get_yield(self, maturity):
        """ Calcule le taux interpolé avec le polynôme quadratique """
        if self.coeffs is None:
            raise ValueError("La courbe des taux n'a pas encore été construite.")
        
        # Évaluation du polynôme aux maturities demandées
        return np.polyval(self.coeffs, maturity)


class Plotter:
    """ Classe utilitaire pour tracer la courbe des taux """
    
    @staticmethod
    def plot_yield_curve(yield_curve, maturities):
        rates = [yield_curve.get_yield(m) for m in maturities]
        
        plt.figure(figsize=(8, 5))
        plt.plot(maturities, rates, marker="o", linestyle="-", label="Yield Curve (Quadratic Interpolation)")
        plt.xlabel("Maturité (années)")
        plt.ylabel("Taux (%)")
        plt.title("Courbe des Taux - Interpolation Quadratique")
        plt.legend()
        plt.grid(True)
        plt.show()


# Exemple d'utilisation
curve = YieldCurve()
curve.add_instrument(Deposit(1.5, 1))
curve.add_instrument(FRA(2.0, 1, 2))
curve.add_instrument(IRS(2.5, 5))
curve.add_instrument(IRS(3.0, 10))
curve.build_curve()

# Tracer la courbe avec interpolation quadratique
maturities = np.linspace(1, 10, 50)  # De 1 à 10 ans
Plotter.plot_yield_curve(curve, maturities)
