import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, CubicSpline

# Base class for financial instruments
class Instrument:
    def __init__(self, maturity, rate):
        self.maturity = maturity
        self.rate = rate  # Market interest rate

# Deposit class (Short-term instrument)
class Deposit(Instrument):
    def __init__(self, maturity, rate):
        super().__init__(maturity, rate)

# FRA (Forward Rate Agreement) class
class FRA(Instrument):
    def __init__(self, start, maturity, fixed_rate):
        super().__init__(maturity, fixed_rate)
        self.start = start  # FRA start time

# Interest Rate Swap (IRS) class
class Swap(Instrument):
    def __init__(self, maturity, fixed_rate):
        super().__init__(maturity, fixed_rate)

# Bootstrapping class to construct the zero-coupon yield curve
class Bootstrapping:
    def __init__(self, instruments):
        self.instruments = sorted(instruments, key=lambda x: x.maturity)
        self.zero_rates = {}  # Dictionary to store zero-coupon rates

    def interpolate_yield_inut(self, maturity):
        """ Calcule le taux interpolé avec le polynôme quadratique """        
        # Évaluation du polynôme aux maturities demandées
        maturities = list(self.zero_rates.keys())  # Abscisses
        rates = list(self.zero_rates.values())    # Ordonnées
        return np.interp(maturity, maturities, rates)
    
#########################################################################
    def interpolate_yield(self, maturity):
        """Interpole un taux zéro-coupon pour une maturité donnée"""
        maturities = sorted(self.zero_rates.keys())  # Maturités triées
        rates = [self.zero_rates[m] for m in maturities]  # Taux correspondants

        if len(maturities) < 2:
            raise ValueError("Impossible d'interpoler : il faut au moins 2 taux zéro-coupon.")

        if len(maturities) == 2:
            # Interpolation linéaire
            return np.interp(maturity, maturities, rates)

        else:
            # Interpolation quadratique ou cubique (préférence pour CubicSpline)
            if len(maturities) >= 3:
                spline = interp1d(maturities, rates, kind='quadratic', fill_value="extrapolate")
            else:
                spline = CubicSpline(maturities, rates, bc_type='natural')
                

            return spline(maturity)
    #######################################################################

    def solve_zero_rate(self, instrument):
        """Computes the zero-coupon rate using bootstrapping."""

        T = instrument.maturity

        # Case 1: Deposits (Direct zero rate calculation)
        if isinstance(instrument, Deposit):
            P_T = 1 / (1 + instrument.rate * T)  # Discount factor
            return -np.log(P_T) / T  # Zero rate from discount factor

        # Case 2: FRA (Forward Rate Agreement)
        elif isinstance(instrument, FRA):
            T1 = instrument.start
            T2 = instrument.maturity

            # Get previously bootstrapped rate for T1
            if T1 in self.zero_rates:
                R1 = self.zero_rates[T1]
            else:
                R1=self.interpolate_yield(T1)
                self.zero_rates[T1] = R1

            # FRA fixed forward rate
            Rf = instrument.rate

            # Compute zero rate at T2 using the given formula
            R2 = ((T2 - T1) * Rf + R1 * T1) / T2

            return R2  # Return the computed zero rate at T2

        # Case 3: Swap (IRS)
        elif isinstance(instrument, Swap):
            def equation(r):
                total_value = 0
                for t in np.arange(1, T + 1):
                    if t in self.zero_rates:
                        discount_factor= np.exp(-self.zero_rates[t]*t)
                    elif t==T:
                        discount_factor= np.exp(-r*t)
                    else:
                        ri=self.interpolate_yield(t)
                        discount_factor = np.exp(-ri*t)
                        self.zero_rates[t] = ri
                    total_value += instrument.rate * discount_factor
                total_value += np.exp(-r * T)  # Final cash flow at maturity
                return total_value - 1  # The equation must balance

            r_T = fsolve(equation, 0.02)[0]  # Solve for zero rate at T
            return r_T  # Return the computed zero rate

    def build_curve(self):
        """Constructs the zero-coupon yield curve using bootstrapping."""
        for inst in sorted(self.instruments, key=lambda x: x.maturity):
            rate = self.solve_zero_rate(inst)
            self.zero_rates[inst.maturity] = rate
        return self.zero_rates

    def interpolate(self, method='linear'):
        """Performs interpolation on the yield curve."""
        maturities = np.array(list(self.zero_rates.keys()))
        rates = np.array(list(self.zero_rates.values()))

        if method == 'linear':
            self.interpolation_func = interp1d(maturities, rates, kind='linear', fill_value="extrapolate")
        elif method == 'quadratic':
            self.interpolation_func = interp1d(maturities, rates, kind='quadratic', fill_value="extrapolate")
        elif method == 'cubic':
            self.interpolation_func = CubicSpline(maturities, rates, bc_type='natural')
        else:
            raise ValueError("Unsupported interpolation method.")

    def get_rate(self, maturity):
        """Returns the interpolated zero-coupon rate for a given maturity."""
        if maturity in self.zero_rates:
            return self.zero_rates[maturity]
        elif hasattr(self, 'interpolation_func'):
            return self.interpolation_func(maturity)
        else:
            raise ValueError("No interpolation defined. Run `interpolate()` first.")

# Function to plot the zero-coupon yield curve
def plot_zero_curve(bootstrap):
    maturities = np.array(list(bootstrap.zero_rates.keys()))
    rates = np.array(list(bootstrap.zero_rates.values()))

    plt.figure(figsize=(10, 6))

    # Scatter plot for bootstrapped rates
    plt.scatter(maturities, rates, color='red', label='Bootstrapped Zero Rates', zorder=3)

    # Linear interpolation
    bootstrap.interpolate(method='linear')
    fine_maturities = np.linspace(min(maturities), max(maturities), 100)
    linear_rates = [bootstrap.get_rate(m) for m in fine_maturities]
    plt.plot(fine_maturities, linear_rates, linestyle='--', color='blue', label='Linear Interpolation')

    # Quadratic interpolation
    bootstrap.interpolate(method='quadratic')
    quadratic_rates = [bootstrap.get_rate(m) for m in fine_maturities]
    plt.plot(fine_maturities, quadratic_rates, linestyle='--', color='green', label='Quadratic Interpolation')

    # Cubic spline interpolation
    bootstrap.interpolate(method='cubic')
    cubic_rates = [bootstrap.get_rate(m) for m in fine_maturities]
    plt.plot(fine_maturities, cubic_rates, linestyle='-', color='black', label='Cubic Interpolation')

    # Labels and title
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Zero Coupon Rate")
    plt.title("Zero-Coupon Yield Curve with Interpolations")
    plt.legend()
    plt.grid(True)
    plt.show()

# Sample market data (realistic example)
instruments = [
    Deposit(0.5, 0.03),  # 6-month deposit, 3% interest rate
    Deposit(1.0, 0.035),  # 1-year deposit, 3.5% interest rate
    FRA(1.0, 1.5, 0.04),  # FRA for 1.5 years, 4% fixed rate
    Swap(2.0, 0.045),  # 2-year swap, 4.5% fixed rate
    Swap(3.0, 0.05)  # 3-year swap, 5% fixed rate
]

# Build the zero-coupon curve
bootstrap = Bootstrapping(instruments)
zero_curve = bootstrap.build_curve()

# Print bootstrapped rates
print("Zero-Coupon Yield Curve (Bootstrapping):")
for m, r in zero_curve.items():
    print(f"Maturity {m} years: {r:.4%}")

# Plot the yield curve with different interpolations
plot_zero_curve(bootstrap)
