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
Import into your python script using

```
# base bangdatapipeline 
!curl --remote-name \
     -H 'Accept: application/vnd.github.v3.raw' \
     --location https://raw.githubusercontent.com/StanfordHCI/bangdatapipeline/master/bangdata.py

# custom multibatch engine - base is multibatch.py
!curl --remote-name \
     -H 'Accept: application/vnd.github.v3.raw' \
     --location https://raw.githubusercontent.com/StanfordHCI/bangdatapipeline/master/multibatch/parallelworlds.py
```