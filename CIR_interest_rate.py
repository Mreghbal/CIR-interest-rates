######################################################################################################

import pandas as pd
import numpy as np
import math

######################################################################################################

def CIR_interest_interest_rates(number_of_years = 10, number_of_scenarios = 1, a = 0.05, b = 0.03,
                                sigma = 0.05, steps_per_year = 12, initial_interest_rate = None):
    """

    Generate random interest rate evolution over time using the CIR model:

    1- "CIR" stands for Cox Ingersoll Ross.

    2- "a" is the speed of mean reversion, "b" is the long term mean rate and
       "sigma" is the volatility parameter.
    
    3- Use "ATI" function to convert an annual interest rate to an instantaneous interest rate.

    4- Use "ITA" function to convert an instantaneous interest rate to an annual interest rate.

    5- Use "random_prices" to generate the random prices evolution of a Zero-Coupon Bond.

    """
######################################################################################################

    def ATI(interset_rate):
        """

        Convert an annual interest rate to an instantaneous interest rate:

        1- ATI stands for annualize to instantaneous.

        """
        return np.log1p(interset_rate)
    
    def ITA(interset_rate):
        """

        Convert an instantaneous interest rate to an annual interest rate:

        1- ITA stands for instantaneous to annualize.

        """
        return np.expm1(interset_rate)

    def random_prices(time, interest_rate):
        """
        
        Generating the random prices evolution of a Zero-Coupon Bond:

        1- The model can also be used to generate the movement of bond prices for a 
           zero coupon bond that are implied by the generated interest rate.

        """
        h = math.sqrt(a ** 2 + 2 * sigma ** 2)
        first = ((2 * h * math.exp((h + a) * time / 2)) / (2 * h + (h + a) * 
                (math.exp(h * time) - 1))) ** (2 * a * b / sigma ** 2)
        second = (2 * (math.exp(h * time) - 1)) / (2 * h + (h + a) * (math.exp(h * time) - 1))
        third = first * np.exp(-second * interest_rate)
        return third

######################################################################################################
    
    if initial_interest_rate is None:
        initial_interest_rate = b 
    
    initial_interest_rate = ATI(initial_interest_rate)
    dt = 1 / steps_per_year
    number_of_steps = int(number_of_years * steps_per_year) + 1
    shock = np.random.normal(0, scale=np.sqrt(dt), size = (number_of_steps, number_of_scenarios))
    interest_rates = np.empty_like(shock)
    interest_rates[0] = initial_interest_rate
    
    prices = np.empty_like(shock)
    prices[0] = random_prices(number_of_years, initial_interest_rate)
    
    for step in range(1, number_of_steps):
        r_t = interest_rates[step - 1]
        d_r_t = a * (b - r_t) * dt + sigma * np.sqrt(r_t) * shock[step]
        interest_rates[step] = abs(r_t + d_r_t)
        prices[step] = random_prices(number_of_years - step * dt, interest_rates[step])

    interest_rates = pd.DataFrame(data = ITA(interest_rates), index = range(number_of_steps))
    prices = pd.DataFrame(data = prices, index = range(number_of_steps))
  
    return interest_rates, prices

######################################################################################################