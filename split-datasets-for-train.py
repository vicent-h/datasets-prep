import pandas as pd
import pyarrow.parquet as pq

df_analise = pd.read_parquet('out/analise_textos.parquet', columns=['index'], filters=[('eval', '==', True)])

df_analise