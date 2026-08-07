
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


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DATASETS = [
    {
        'path': '/media/alvarinho/dados/Datasets/raw/emea en-es.txt/',
        'src_file': 'EMEA.en-es.en',
        'tgt_file': 'EMEA.en-es.es',
        'src_lang': 'en',
        'tgt_lang': 'es',
    },
    {
        'path': '/media/alvarinho/dados/Datasets/raw/emea en-pt.txt/',
        'src_file': 'EMEA.en-pt.en',
        'tgt_file': 'EMEA.en-pt.pt',
        'src_lang': 'en',
        'tgt_lang': 'pt',
    },

    {
        'path': '/media/alvarinho/dados/Datasets/raw/paracrawl en-es.txt/',
        'src_file': 'ParaCrawl.en-es.en',
        'tgt_file': 'ParaCrawl.en-es.es',
        'src_lang': 'en',
        'tgt_lang': 'es',
    },
    {
        'path': '/media/alvarinho/dados/Datasets/raw/paracrawl en-pt.txt/',
        'src_file': 'ParaCrawl.en-pt.en',
        'tgt_file': 'ParaCrawl.en-pt.pt',
        'src_lang': 'en',
        'tgt_lang': 'pt',
    },

    {
        'path': '/media/alvarinho/dados/Datasets/raw/scielo en-pt.txt/',
        'src_file': 'SciELO.en-pt.en',
        'tgt_file': 'SciELO.en-pt.pt',
        'src_lang': 'en',
        'tgt_lang': 'pt',
    },

    {
        'path': '/media/alvarinho/dados/Datasets/raw/wikimatrix en-es.txt/',
        'src_file': 'WikiMatrix.en-es.en',
        'tgt_file': 'WikiMatrix.en-es.es',
        'src_lang': 'en',
        'tgt_lang': 'es',
    },
    {
        'path': '/media/alvarinho/dados/Datasets/raw/wikimatrix en-pt.txt/',
        'src_file': 'WikiMatrix.en-pt.en',
        'tgt_file': 'WikiMatrix.en-pt.pt',
        'src_lang': 'en',
        'tgt_lang': 'pt',
    },
]


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


# Arquivo final
OUTPUT_FILE = 'out/analise_textos.parquet'
OUTPUT_TSV_FILE = '/media/alvarinho/dados/Datasets/refined/analise_textos.tsv'
TSV_SEPARATOR = '<SEP>'


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
            max_length_diff_ratio=2.0
        )
    )

    # --------------------------------------------------------
    # DESABILITADOS
    #
    # Essas etapas foram deixadas comentadas porque podem
    # possuir estado global de deduplicação.
    # --------------------------------------------------------

    # pipe.add_step(
    #     ExactDuplicator()
    # )

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
    path,
    src_file,
    tgt_file,
    src_lang,
    tgt_lang,
    chunk_size
):
    """
    Lê os dois arquivos simultaneamente e produz chunks.

    Importante:
    o arquivo inteiro NÃO é carregado na RAM.

    Apenas CHUNK_SIZE pares de linhas ficam na memória
    por vez no processo principal.
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

            chunk = list(
                islice(
                    zip(f_src, f_tgt),
                    chunk_size
                )
            )

            if not chunk:
                break

            yield (
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
            'path': path,
            'src_file': src_file,
            'tgt_file': tgt_file,
            'index': index,
            'eval': eval_result,
            'len_text1': len(text1),
            'len_text2': len(text2),
            'text1': text1,
            'text2': text2
        }

        result.update(metrics)

        results.append(result)

    return results


# ============================================================
# SALVAR RESULTADO NO PARQUET
# ============================================================

def write_results_to_parquet(
    results,
    parquet_writer
):
    """
    Converte os resultados de um chunk para Arrow e escreve
    diretamente no Parquet.

    O DataFrame existe apenas durante esta função.
    """

    if not results:
        return

    df = pd.DataFrame(results)

    table = pa.Table.from_pandas(
        df,
        preserve_index=False
    )

    parquet_writer.write_table(table)

    # Liberamos imediatamente as referências grandes.
    del table
    del df
    del results


def write_results_to_tsv(results, tsv_handle):
    """
    Escreve os pares de texto recebidos no kwargs em um arquivo TSV.
    Cada linha segue o formato:

        text1<Sep>text2

    """

    if not results:
        return

    for result in results:
        if(result['eval']):
            text1 = str(result.get('text1', '')).replace('\n', ' ').replace('\t', ' ')
            text2 = str(result.get('text2', '')).replace('\n', ' ').replace('\t', ' ')
            tsv_handle.write(f'{text1}{TSV_SEPARATOR}{text2}\n')


# ============================================================
# PROCESSA UM DATASET
# ============================================================

def process_dataset(
    dataset,
    parquet_writer
):

    path = dataset['path']
    src_file = dataset['src_file']
    tgt_file = dataset['tgt_file']

    src_lang = dataset['src_lang']
    tgt_lang = dataset['tgt_lang']

    print()
    print('=' * 80)
    print(f'Analisando: {src_file} -> {tgt_file}')
    print(f'Idiomas: {src_lang} -> {tgt_lang}')
    print('=' * 80)

    chunks = create_chunks(
        path=path,
        src_file=src_file,
        tgt_file=tgt_file,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        chunk_size=CHUNK_SIZE
    )

    # --------------------------------------------------------
    # ProcessPool
    #
    # Cada worker:
    #
    # 1. é criado
    # 2. executa init_worker()
    # 3. cria seu Pipeline
    # 4. processa vários chunks
    # 5. é destruído ao sair do "with"
    # --------------------------------------------------------

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=init_worker
    ) as executor:

        pending = set()

        # ----------------------------------------------------
        # Enviamos inicialmente apenas MAX_PENDING chunks.
        #
        # NÃO enviamos todos os chunks do arquivo de uma vez.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Enquanto existirem tarefas pendentes...
        # ----------------------------------------------------

        with tqdm(
            desc=src_file,
            unit='linhas'
        ) as progress:

            while pending:

                done, pending = wait(
                    pending,
                    return_when=FIRST_COMPLETED
                )

                # ------------------------------------------------
                # Cada future terminado libera seus dados depois
                # que escrevemos no Parquet.
                # ------------------------------------------------

                for future in done:

                    results = future.result()

                    # Quantidade de linhas processadas
                    progress.update(
                        len(results)
                    )

                    # Escreve imediatamente no Parquet
                    write_results_to_parquet(
                        results,
                        parquet_writer
                    )

                    # --------------------------------------------
                    # Importante:
                    #
                    # Não guardamos "results".
                    # Ele é liberado logo após ser escrito.
                    # --------------------------------------------

                    del results

                    # --------------------------------------------
                    # Tenta colocar um novo chunk na fila.
                    # --------------------------------------------

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

                    # Libera objetos temporários do Python.
                    gc.collect()

    # --------------------------------------------------------
    # IMPORTANTE:
    #
    # Ao chegar aqui, o "with ProcessPoolExecutor" terminou.
    #
    # Portanto TODOS os workers deste dataset foram encerrados.
    # --------------------------------------------------------

    print(
        f'Finalizado: {src_file}'
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_TSV_FILE), exist_ok=True)

    parquet_writer = None
    tsv_handle = open(OUTPUT_TSV_FILE, 'w', encoding='utf-8')

    try:

        for dataset_index, dataset in enumerate(DATASETS):

            # ------------------------------------------------
            # Primeiro dataset:
            #
            # criamos o ParquetWriter.
            #
            # Os próximos datasets continuam escrevendo
            # no mesmo arquivo.
            # ------------------------------------------------

            if parquet_writer is None:

                # ------------------------------------------------
                # Precisamos descobrir o schema do primeiro chunk.
                #
                # Para isso, processamos o primeiro dataset
                # normalmente e o writer será criado na primeira
                # escrita.
                # ------------------------------------------------

                pass

            # ------------------------------------------------
            # Como precisamos criar o writer com o schema do
            # primeiro resultado, usamos uma versão especial
            # para o primeiro chunk.
            # ------------------------------------------------

            path = dataset['path']
            src_file = dataset['src_file']
            tgt_file = dataset['tgt_file']

            src_lang = dataset['src_lang']
            tgt_lang = dataset['tgt_lang']

            print()
            print('=' * 80)
            print(
                f'Dataset {dataset_index + 1}/{len(DATASETS)}'
            )
            print(
                f'{src_file} -> {tgt_file}'
            )
            print('=' * 80)

            chunks = create_chunks(
                path=path,
                src_file=src_file,
                tgt_file=tgt_file,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                chunk_size=CHUNK_SIZE
            )

            # ------------------------------------------------
            # Processamos o dataset.
            # ------------------------------------------------

            with ProcessPoolExecutor(
                max_workers=NUM_WORKERS,
                initializer=init_worker
            ) as executor:

                pending = set()

                # ------------------------------------------------
                # Inicializa os primeiros trabalhos.
                # ------------------------------------------------

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

                # ------------------------------------------------
                # Barra de progresso.
                # ------------------------------------------------

                with tqdm(
                    desc=src_file,
                    unit='linhas'
                ) as progress:

                    while pending:

                        done, pending = wait(
                            pending,
                            return_when=FIRST_COMPLETED
                        )

                        for future in done:

                            results = future.result()

                            progress.update(
                                len(results)
                            )

                            write_results_to_tsv(
                                results,
                                tsv_handle
                            )

                            # ------------------------------------
                            # DataFrame temporário
                            # ------------------------------------

                            df = pd.DataFrame(
                                results
                            )

                            if 'text1' in df.columns:
                                df = df.drop(columns=['text1'])

                            if 'text2' in df.columns:
                                df = df.drop(columns=['text2'])

                            table = pa.Table.from_pandas(
                                df,
                                preserve_index=False
                            )

                            # ------------------------------------
                            # Cria o ParquetWriter na primeira
                            # vez que tivermos um schema.
                            # ------------------------------------

                            if parquet_writer is None:

                                parquet_writer = pq.ParquetWriter(
                                    OUTPUT_FILE,
                                    table.schema,
                                    compression='zstd'
                                )

                            else:

                                # --------------------------------
                                # Garante que o schema permaneça
                                # consistente entre os chunks.
                                # --------------------------------

                                if table.schema != parquet_writer.schema:

                                    table = table.cast(
                                        parquet_writer.schema
                                    )

                            # ------------------------------------
                            # Escreve imediatamente.
                            # ------------------------------------

                            parquet_writer.write_table(
                                table
                            )

                            # ------------------------------------
                            # Libera memória.
                            # ------------------------------------

                            del table
                            del df
                            del results

                            gc.collect()

                            # ------------------------------------
                            # Coloca mais um chunk na fila.
                            # ------------------------------------

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

            # ------------------------------------------------
            # Aqui o ProcessPoolExecutor terminou.
            #
            # Todos os workers deste dataset foram encerrados.
            # ------------------------------------------------

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

        # ----------------------------------------------------
        # Fecha o ParquetWriter.
        # ----------------------------------------------------

        if parquet_writer is not None:

            parquet_writer.close()

            parquet_writer = None

        if tsv_handle is not None:

            tsv_handle.close()

            tsv_handle = None

        gc.collect()

    print()
    print('=' * 80)
    print('PROCESSAMENTO CONCLUÍDO')
    print('=' * 80)

    print(
        f'Arquivo salvo em: {OUTPUT_FILE}'
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':

    main()


