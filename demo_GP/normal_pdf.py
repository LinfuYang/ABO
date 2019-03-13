from scipy import stats
from scipy.spatial.distance import pdist
import numpy as np

print(pdist(np.vstack([np.array([0.11, 0.11]), np.array([0.12, 0.12])])))
print(np.sqrt(np.sum(np.square(np.array([0.11, 0.11]) - np.array([0.12, 0.12])))))

print(stats.norm.pdf(0.99999, 15, 20))