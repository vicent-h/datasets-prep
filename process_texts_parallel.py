from src.clean_character import CleanSentences
from src.deduplicator import ExactDuplicator, MinHashDetector
from src.lang_identify import LangIdentifier
from src.length_clash import LengthClash
from src.numbers_clash import NumbersClash
from src.unidecode_norm import UnidecodeNorm
from src.pipeline import Pipeline

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from itertools import islice
from tqdm import tqdm

import gc
import os

import time


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DATASETS = [
    {
        'dataset_id': 'EMEA_en_es',

        'path': '/media/alvarinho/dados/Datasets/raw/emea en-es.txt/',
        'src_file': 'EMEA.en-es.en',
        'tgt_file': 'EMEA.en-es.es',

        'src_lang': 'en',
        'tgt_lang': 'es',
    },

    {
        'dataset_id': 'EMEA_en_pt',

        'path': '/media/alvarinho/dados/Datasets/raw/emea en-pt.txt/',
        'src_file': 'EMEA.en-pt.en',
        'tgt_file': 'EMEA.en-pt.pt',

        'src_lang': 'en',
        'tgt_lang': 'pt',
    },

    {
        'dataset_id': 'ParaCrawl_en_es',

        'path': '/media/alvarinho/dados/Datasets/raw/paracrawl en-es.txt/',
        'src_file': 'ParaCrawl.en-es.en',
        'tgt_file': 'ParaCrawl.en-es.es',

        'src_lang': 'en',
        'tgt_lang': 'es',
    },

    {
        'dataset_id': 'ParaCrawl_en_pt',

        'path': '/media/alvarinho/dados/Datasets/raw/paracrawl en-pt.txt/',
        'src_file': 'ParaCrawl.en-pt.en',
        'tgt_file': 'ParaCrawl.en-pt.pt',

        'src_lang': 'en',
        'tgt_lang': 'pt',
    },

    {
        'dataset_id': 'SciELO_en_pt',

        'path': '/media/alvarinho/dados/Datasets/raw/scielo en-pt.txt/',
        'src_file': 'SciELO.en-pt.en',
        'tgt_file': 'SciELO.en-pt.pt',

        'src_lang': 'en',
        'tgt_lang': 'pt',
    },

    {
        'dataset_id': 'WikiMatrix_en_es',

        'path': '/media/alvarinho/dados/Datasets/raw/wikimatrix en-es.txt/',
        'src_file': 'WikiMatrix.en-es.en',
        'tgt_file': 'WikiMatrix.en-es.es',

        'src_lang': 'en',
        'tgt_lang': 'es',
    },

    {
        'dataset_id': 'WikiMatrix_en_pt',

        'path': '/media/alvarinho/dados/Datasets/raw/wikimatrix en-pt.txt/',
        'src_file': 'WikiMatrix.en-pt.en',
        'tgt_file': 'WikiMatrix.en-pt.pt',

        'src_lang': 'en',
        'tgt_lang': 'pt',
    },
]

# ============================================================
# LIMITE DE FRASES
# ============================================================

# Quantidade máxima de pares de frases a serem processados
# POR DATASET.
#
# Exemplo:
# MAX_SENTENCE_PAIRS = 100_000
#
# Use None para processar o dataset inteiro.
MAX_SENTENCE_PAIRS = 25_000_000

EXPORT_STATISTICS = False  # True para exportar estatísticas de cada dataset

# ============================================================
# PROCESSAMENTO
# ============================================================

# Quantidade de linhas processadas por tarefa.
#
# 25.000 é um valor razoável para começar.
# Se a RAM ficar muito alta, reduza para 10.000.
# Se os workers estiverem gastando muito tempo com overhead,
# pode aumentar para 50.000.
CHUNK_SIZE = 100_000


# Número de processos simultâneos.
#
# IMPORTANTE:
# Cada worker possui sua própria instância do Pipeline
# e, consequentemente, sua própria instância do LangIdentifier.
#
# Se cada modelo consumir muita RAM, reduza esse número.
NUM_WORKERS = 16


# Mantemos somente alguns chunks esperando.
#
# Exemplo:
#
# NUM_WORKERS = 16
# MAX_PENDING = 32
#
# No máximo 32 chunks ficam enviados para os workers.
# Isso evita colocar milhares de chunks na RAM.
MAX_PENDING = NUM_WORKERS * 2


# ============================================================
# ARQUIVOS DE SAÍDA
# ============================================================

OUTPUT_PATH = '/media/alvarinho/dados/Datasets/refined/traducao'
OUTPUT_FILE = os.path.join(OUTPUT_PATH, 'analise_textos.parquet')
OUTPUT_TSV_FILE = os.path.join(OUTPUT_PATH, 'analise_textos.tsv')
TSV_SEPARATOR = '<SEP>'
TSV_SEPARATOR_METADATA = '<METADATA>'


# ============================================================
# PIPELINE
# ============================================================

def create_pipeline():

    pipe = Pipeline()

    pipe.add_step(
        CleanSentences()
    )

    pipe.add_step(
        UnidecodeNorm()
    )

    pipe.add_step(
        LangIdentifier(
            model_path=None,
            threshold=0.3,
            k=1
        )
    )

    pipe.add_step(
        LengthClash(
            max_length_diff_ratio=1.7
        )
    )

    # --------------------------------------------------------
    # DESABILITADOS
    #
    # Essas etapas foram deixadas comentadas porque podem
    # possuir estado global de deduplicação.
    # --------------------------------------------------------

    pipe.add_step(
        ExactDuplicator()
    )

    # pipe.add_step(
    #     MinHashDetector()
    # )

    pipe.add_step(
        NumbersClash(
            threshold=0.75
        )
    )

    return pipe


# ============================================================
# PIPELINE DO WORKER
# ============================================================

# Variável global EXISTENTE DENTRO DE CADA PROCESSO.
#
# Cada processo terá sua própria variável PIPE.
PIPE = None


def init_worker():
    """
    Executado uma única vez quando cada worker é criado.

    Assim o Pipeline não é recriado para cada chunk.
    """

    global PIPE

    PIPE = create_pipeline()


# ============================================================
# GERADOR DE CHUNKS
# ============================================================

def create_chunks(
    dataset_id,
    path,
    src_file,
    tgt_file,
    src_lang,
    tgt_lang,
    chunk_size,
    max_sentence_pairs=None
):
    """
    Lê os dois arquivos simultaneamente e produz chunks.

    Apenas max_sentence_pairs serão processados quando
    o limite estiver definido.

    O limite é aplicado por dataset.
    """

    with open(
        path + src_file,
        'r',
        encoding='utf-8'
    ) as f_src, open(
        path + tgt_file,
        'r',
        encoding='utf-8'
    ) as f_tgt:

        start_index = 0

        while True:

            # ----------------------------------------------------
            # Verifica se atingimos o limite
            # ----------------------------------------------------

            if (
                max_sentence_pairs is not None
                and start_index >= max_sentence_pairs
            ):
                break

            # ----------------------------------------------------
            # Define o tamanho deste chunk
            # ----------------------------------------------------

            current_chunk_size = chunk_size

            if max_sentence_pairs is not None:
                remaining = max_sentence_pairs - start_index

                current_chunk_size = min(
                    chunk_size,
                    remaining
                )

            # ----------------------------------------------------
            # Lê o chunk
            # ----------------------------------------------------

            chunk = list(
                islice(
                    zip(f_src, f_tgt),
                    current_chunk_size
                )
            )

            if not chunk:
                break

            yield (
                dataset_id,
                path,
                src_file,
                tgt_file,
                src_lang,
                tgt_lang,
                start_index,
                chunk
            )

            start_index += len(chunk)


# ============================================================
# PROCESSAMENTO DE UM CHUNK
# ============================================================

def process_chunk(args):

    global PIPE

    (
        dataset_id,
        path,
        src_file,
        tgt_file,
        src_lang,
        tgt_lang,
        start_index,
        lines
    ) = args

    results = []

    for offset, (src_line, tgt_line) in enumerate(lines):

        # ----------------------------------------------------
        # INDEX ORIGINAL
        # ----------------------------------------------------
        #
        # Esse índice NÃO é o índice do TSV.
        #
        # Ele corresponde à posição da linha no arquivo bruto
        # daquele dataset.
        #
        index = start_index + offset

        raw_text1 = src_line.strip()
        raw_text2 = tgt_line.strip()

        processed_kwargs, eval_result, metrics = PIPE.process_all(
            {
                'text1': raw_text1,
                'text2': raw_text2,
                'expected_lang1': src_lang,
                'expected_lang2': tgt_lang
            }
        )

        text1 = processed_kwargs['text1']
        text2 = processed_kwargs['text2']

        result = {

            # ------------------------------------------------
            # IDENTIFICAÇÃO DO DATASET
            # ------------------------------------------------
            'dataset_id': dataset_id,

            # ------------------------------------------------
            # INFORMAÇÕES DO ARQUIVO
            # ------------------------------------------------
            'path': path,
            'src_file': src_file,
            'tgt_file': tgt_file,

            # ------------------------------------------------
            # IDENTIFICADOR ORIGINAL
            # ------------------------------------------------
            'index': index,

            # ------------------------------------------------
            # RESULTADO DA AVALIAÇÃO
            # ------------------------------------------------
            'eval': eval_result,

            # ------------------------------------------------
            # MÉTRICAS
            # ------------------------------------------------
            'len_text1': len(text1),
            'len_text2': len(text2),

            # ------------------------------------------------
            # TEXTOS
            # ------------------------------------------------
            'text1': text1,
            'text2': text2
        }

        if EXPORT_STATISTICS:
            result.update(metrics)
        else:
            result['lang_identify_prob'] = metrics.get('lang_identify_prob1') * metrics.get('lang_identify_prob2')


        results.append(result)

    return results


# ============================================================
# SALVAR RESULTADO NO TSV
# ============================================================

def write_results_to_tsv(results, tsv_handle):
    """
    Escreve SOMENTE exemplos válidos no TSV.

    O TSV mantém a identificação original do exemplo.

    Formato:

        dataset_id<SEP>index<SEP>text1<SEP>text2

    Exemplo:

        EMEA_en_es<SEP>15342<SEP>Hello world<SEP>Hola mundo

    IMPORTANTE:
    O index é o índice original no dataset bruto.
    Portanto, remover exemplos inválidos não quebra
    o rastreamento.
    """

    if not results:
        return

    for result in results:

        # ----------------------------------------------------
        # FILTRO
        # ----------------------------------------------------

        if result['eval']:

            dataset_id = str(
                result['dataset_id']
            )

            index = result['index']

            # ------------------------------------------------
            # TEXTOS
            # ------------------------------------------------

            text1 = (
                str(result.get('text1', ''))
                .replace('\n', ' ')
                .replace('\r', ' ')
                .replace('\t', ' ')
            )

            text2 = (
                str(result.get('text2', ''))
                .replace('\n', ' ')
                .replace('\r', ' ')
                .replace('\t', ' ')
            )

            # ------------------------------------------------
            # ESCREVE
            # ------------------------------------------------

            tsv_handle.write(
                f'{dataset_id}{TSV_SEPARATOR}'
                f'{index}{TSV_SEPARATOR_METADATA}'
                f'{text1}{TSV_SEPARATOR}'
                f'{text2}\n'
            )


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Cria diretórios de saída
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(OUTPUT_TSV_FILE),
        exist_ok=True
    )

    parquet_writer = None

    tsv_handle = open(
        OUTPUT_TSV_FILE,
        'w',
        encoding='utf-8'
    )

    try:

        # ====================================================
        # HEADER DO TSV
        # ====================================================
        #
        # O header também utiliza o mesmo separador.
        # ====================================================

        tsv_handle.write(
            f'dataset_id{TSV_SEPARATOR}'
            f'index{TSV_SEPARATOR}'
            f'text1{TSV_SEPARATOR}'
            f'text2\n'
        )

        # ====================================================
        # DATASETS
        # ====================================================

        for dataset_index, dataset in enumerate(DATASETS):

            dataset_id = dataset['dataset_id']

            path = dataset['path']
            src_file = dataset['src_file']
            tgt_file = dataset['tgt_file']

            src_lang = dataset['src_lang']
            tgt_lang = dataset['tgt_lang']

            print()
            print('=' * 80)

            print(
                f'Dataset '
                f'{dataset_index + 1}/{len(DATASETS)}'
            )

            print(
                f'ID: {dataset_id}'
            )

            print(
                f'{src_file} -> {tgt_file}'
            )

            print(
                f'Idiomas: '
                f'{src_lang} -> {tgt_lang}'
            )

            print('=' * 80)

            # ------------------------------------------------
            # CHUNKS
            # ------------------------------------------------

            chunks = create_chunks(
                dataset_id=dataset_id,
                path=path,
                src_file=src_file,
                tgt_file=tgt_file,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                chunk_size=CHUNK_SIZE,
                max_sentence_pairs=MAX_SENTENCE_PAIRS
            )

            # ------------------------------------------------
            # PROCESS POOL
            # ------------------------------------------------

            with ProcessPoolExecutor(
                max_workers=NUM_WORKERS,
                initializer=init_worker
            ) as executor:

                pending = set()

                # =================================================
                # Inicializa os primeiros trabalhos
                # =================================================

                for _ in range(MAX_PENDING):

                    try:

                        chunk = next(chunks)

                    except StopIteration:

                        break

                    future = executor.submit(
                        process_chunk,
                        chunk
                    )

                    pending.add(future)

                # =================================================
                # PROGRESS BAR
                # =================================================

                with tqdm(
                    desc=src_file,
                    unit='linhas'
                ) as progress:

                    while pending:

                        done, pending = wait(
                            pending,
                            return_when=FIRST_COMPLETED
                        )

                        # =================================================
                        # PROCESSA CHUNKS TERMINADOS
                        # =================================================

                        for future in done:

                            results = future.result()

                            # ------------------------------------------------
                            # PROGRESSO
                            # ------------------------------------------------

                            progress.update(
                                len(results)
                            )

                            # =================================================
                            # TSV
                            # =================================================
                            #
                            # SOMENTE eval=True é escrito.
                            #
                            # Porém dataset_id + index são preservados.
                            # =================================================

                            write_results_to_tsv(
                                results,
                                tsv_handle
                            )

                            # =================================================
                            # PARQUET
                            # =================================================
                            #
                            # Diferentemente do TSV, o Parquet mantém
                            # TODOS os exemplos, inclusive eval=False.
                            #
                            # Portanto ele continua sendo o arquivo
                            # completo de análise.
                            # =================================================

                            df = pd.DataFrame(
                                results
                            )

                            df = df[df['eval'] == True]

                            # ------------------------------------------------
                            # Remove os textos do Parquet
                            # ------------------------------------------------
                            #
                            # Os textos válidos estão no TSV.
                            # O Parquet fica contendo metadados/métricas.
                            # ------------------------------------------------

                            if 'text1' in df.columns:

                                df = df.drop(
                                    columns=['text1']
                                )

                            if 'text2' in df.columns:

                                df = df.drop(
                                    columns=['text2']
                                )

                            # ------------------------------------------------
                            # Converte para Arrow
                            # ------------------------------------------------

                            table = pa.Table.from_pandas(
                                df,
                                preserve_index=False
                            )

                            # =================================================
                            # CRIA PARQUET WRITER
                            # =================================================

                            if parquet_writer is None:

                                parquet_writer = pq.ParquetWriter(
                                    OUTPUT_FILE,
                                    table.schema,
                                    compression='zstd'
                                )

                            else:

                                # =================================================
                                # Garante schema consistente
                                # =================================================

                                if (
                                    table.schema
                                    != parquet_writer.schema
                                ):

                                    table = table.cast(
                                        parquet_writer.schema
                                    )

                            # =================================================
                            # ESCREVE PARQUET
                            # =================================================

                            parquet_writer.write_table(
                                table
                            )

                            # =================================================
                            # LIBERA MEMÓRIA
                            # =================================================

                            del table
                            del df
                            del results

                            gc.collect()

                            # =================================================
                            # COLOCA NOVO CHUNK NA FILA
                            # =================================================

                            try:

                                chunk = next(chunks)

                                new_future = executor.submit(
                                    process_chunk,
                                    chunk
                                )

                                pending.add(
                                    new_future
                                )

                            except StopIteration:

                                pass

            # ========================================================
            # PROCESS POOL TERMINOU
            # ========================================================

            del chunks

            gc.collect()

            print()
            print(
                f'Dataset {dataset_index + 1} concluído.'
            )

            print(
                'Workers encerrados e memória liberada.'
            )

    finally:

        # ========================================================
        # FECHA PARQUET
        # ========================================================

        if parquet_writer is not None:

            parquet_writer.close()

            parquet_writer = None

        # ========================================================
        # FECHA TSV
        # ========================================================

        if tsv_handle is not None:

            tsv_handle.close()

            tsv_handle = None

        gc.collect()

    # ============================================================
    # FINAL
    # ============================================================

    print()
    print('=' * 80)
    print('PROCESSAMENTO CONCLUÍDO')
    print('=' * 80)

    print(
        f'Parquet salvo em: {OUTPUT_FILE}'
    )

    print(
        f'TSV salvo em: {OUTPUT_TSV_FILE}'
    )

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time

    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(
        f'Tempo total: '
        f'{int(hours):02d}:'
        f'{int(minutes):02d}:'
        f'{seconds:05.2f}'
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':

    main()

