import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from matplotlib.pyplot import subplot


def normal(mean, std):
    
    values = []
    
    plt.figure(figsize=(12, 6))
    
    
    for i in range(200):  
        value = norm.rvs(loc=mean, scale=std, size=1, random_state=None)  
        values.append(value[0])  
        
        plt.clf()  
        
        # Plot histogram
        plt.subplot(1, 2, 1)
        plt.hist(values, bins=100, color='black', density=True)
        plt.title('Histogram of Values')
        plt.xlabel('Sample Value')
        plt.ylabel('Density')
        
        # Plot trace plot
        plt.subplot(1, 2, 2)
        plt.plot(values, color='black', alpha=0.7)
        plt.title('Trace Plot')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        
        # Redraw the updated plots
        plt.pause(0.01)
    
    plt.show()

# Call the function
normal(1, 1)


