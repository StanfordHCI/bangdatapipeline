# bangdatapipeline
a module for the data pipeline for the Scaled Humanity team

optimized for Aug-Sep 2019 single-team experiments with these settings:
```
SETTINGS = {
    "VIABILITY_START": 0, 
    "VIABILITY_END": 13,
    "FRACTURE_INDEX": 14,
    "FRACTURE_WHY": 15,
    "LENGTH": 16
}
```

## usage
Import into your python script by cloning the repo and then using

```
# Install bangdatapipeline & multibatch packages
!pip install ./bangdatapipeline

from bangdatapipeline import BangDataPipeline
from multibatch import Multibatch # imports default Multibatch
from multibatch.parallelworlds import Multibatch # or whatever version you add
```