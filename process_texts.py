from src.clean_character import CleanSentences
from src.deduplicator import ExactDuplicator, MinHashDetector
from src.lang_identify import LangIdentifier
from src.length_clash import LengthClash
from src.numbers_clash import NumbersClash
from src.unidecode_norm import UnidecodeNorm
from src.pipeline import Pipeline
from typing import List
from src.processor import Processor
import pandas as pd
from tqdm import tqdm

DATASETS = [
    {'path': '/media/alvarinho/dados/Datasets/raw/emea en-es.txt/', 'src_file': 'EMEA.en-es.en', 'tgt_file': 'EMEA.en-es.es'},
    {'path': '/media/alvarinho/dados/Datasets/raw/emea en-pt.txt/', 'src_file': 'EMEA.en-pt.en', 'tgt_file': 'EMEA.en-pt.pt'},
 
    {'path': '/media/alvarinho/dados/Datasets/raw/paracrawl en-es.txt/', 'src_file': 'ParaCrawl.en-es.en', 'tgt_file': 'ParaCrawl.en-es.es'},
    {'path': '/media/alvarinho/dados/Datasets/raw/paracrawl en-pt.txt/', 'src_file': 'ParaCrawl.en-pt.en', 'tgt_file': 'ParaCrawl.en-pt.pt'},
    
    {'path': '/media/alvarinho/dados/Datasets/raw/scielo en-pt.txt/', 'src_file': 'SciELO.en-pt.en', 'tgt_file': 'SciELO.en-pt.pt'},

    {'path': '/media/alvarinho/dados/Datasets/raw/wikimatrix en-es.txt/', 'src_file': 'WikiMatrix.en-es.en', 'tgt_file': 'WikiMatrix.en-es.es'},
    {'path': '/media/alvarinho/dados/Datasets/raw/wikimatrix en-pt.txt/', 'src_file': 'WikiMatrix.en-pt.en', 'tgt_file': 'WikiMatrix.en-pt.pt'},
]



pipe = Pipeline()
pipe.add_step(CleanSentences())
pipe.add_step(UnidecodeNorm())
pipe.add_step(LangIdentifier(model_path=None, threshold=0.75, k=1))
pipe.add_step(LengthClash(max_length_diff_ratio=2.0))
# pipe.add_step(ExactDuplicator())
# pipe.add_step(MinHashDetector())
pipe.add_step(NumbersClash(threshold=0.75))

df_analise = pd.DataFrame()

for dataset_atual in DATASETS:
    path = dataset_atual['path']
    src_file = dataset_atual['src_file']
    tgt_file = dataset_atual['tgt_file']

    print('Analisando o path: ', path)

    with open(path + src_file, 'r', encoding='utf-8') as f_src, open(path + tgt_file, 'r', encoding='utf-8') as f_tgt:
        for i, (src_line, tgt_line) in tqdm(enumerate(zip(f_src, f_tgt))):
            kwargs, eval, metrics = pipe.process_all(
                {'text1': src_line.strip(), 
                'text2': tgt_line.strip(), 
                'expected_lang1': 'en', 
                'expected_lang2': 'es'
                })  

            df_analise_atual = pd.DataFrame({
                'path': [path],
                'src_file': [src_file],
                'tgt_file': [tgt_file],
                'index': [i],
                'eval': [eval]})
            df_analise_atual_metrics = pd.DataFrame.from_dict(metrics, orient='index').T
            df_analise_atual = pd.concat([df_analise_atual, df_analise_atual_metrics], axis=1)
            df_analise = pd.concat([df_analise, df_analise_atual], ignore_index=True)

df_analise.to_parquet('out/analise_textos.parquet', index=False)