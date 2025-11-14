#!/usr/bin/env python3
import pandas as pd
pd.read_csv("radiomics_features.csv").to_excel("radiomics_features.xlsx", index=False)
