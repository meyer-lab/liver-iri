import numpy as np
import pandas as pd
from tensorpack import perform_CP

from .dataimport import cytokine_data

OPTIMAL_COMPONENTS = 9

def get_factors(rank=9):
    data = cytokine_data()
    cp = perform_CP(data.values, rank)
    factors = pd.DataFrame(
        cp.factors[0],
        index=data.Patient.values,
        columns=np.arange(1, rank + 1)
    )
    return factors
