import pandas as pd
import pyarrow.parquet as pq

# df_analise = pd.read_parquet(
#     'out/analise_textos.parquet', 
#     columns=['dataset_id', 'index'], 
#     filters=[('eval', '==', True)],
#     engine='pyarrow',
#     dtype_backend='pyarrow' # Faz o Pandas usar strings e ints otimizados do Arrow
# )

# # Ainda é recomendado converter para categoria se houver pouca cardinalidade
# df_analise['dataset_id'] = df_analise['dataset_id'].astype('category')

import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pandas as pd

# Definição das probabilidades
# Treino: 90%, Validação: 1%, Teste: 9%
probabilidades = [0.90, 0.01, 0.09]
rotulos = ['train', 'val', 'test']

print('Iniciando o processamento em lotes...')
parquet_file = pq.ParquetFile('/media/alvarinho/dados/Datasets/refined/traducao/analise_textos.parquet')
writer = None

# Lendo o arquivo em lotes de 1 milhão de linhas (ajuste se precisar de mais ou menos RAM)
for i, batch in enumerate(parquet_file.iter_batches(batch_size=1_000_000)):
    print(f"Processando lote {i+1}...")
    
    # 1. Converte o batch para Pandas
    df_chunk = batch.to_pandas()
    
    # 2. Aplica o filtro 'eval' (caso ele seja uma coluna no arquivo)
    # Ignoramos esse passo se você já filtrou antes, mas incluí por segurança baseado no seu print
    if 'eval' in df_chunk.columns:
        df_chunk = df_chunk[df_chunk['eval'] == True].copy()
        
    if len(df_chunk) == 0:
        continue
        
    # 3. O "Pulo do Gato": Atribuição Aleatória Ponderada
    # Para cada linha do chunk, ele sorteia 'train', 'val' ou 'test' respeitando a probabilidade.
    df_chunk['split'] = np.random.choice(rotulos, size=len(df_chunk), p=probabilidades)
    
    # Converte a coluna para categoria para economizar espaço em disco
    df_chunk['split'] = df_chunk['split'].astype('category')
    
    # 4. Converte de volta para tabela do PyArrow
    table_chunk = pa.Table.from_pandas(df_chunk)
    
    # 5. Inicializa o escritor (writer) apenas na primeira iteração, usando o schema do lote
    if writer is None:
        writer = pq.ParquetWriter('/media/alvarinho/dados/Datasets/refined/traducao/analise_textos_split.parquet', table_chunk.schema)
        
    # 6. Salva o lote no disco
    writer.write_table(table_chunk)

# Fecha o arquivo final
if writer:
    writer.close()

print('Concluído! Arquivo salvo como "/media/alvarinho/dados/Datasets/refined/traducao/analise_textos_split.parquet"')