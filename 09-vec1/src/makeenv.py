import urllib.request
import bz2
import logging
from logging import getLogger, StreamHandler, Formatter
from pathlib import Path

# logの設定
logger = getLogger('nlp_09')
logger.setLevel(logging.DEBUG)
stream_handler = StreamHandler()
stream_handler.setLevel(logging.DEBUG)
handler_format = Formatter('%(asctime)s - %(name)s - [%(levelname)s] %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

files = {
    'dir': Path.home() / 'tmp' / '09',
    'raw_bz2': Path.home() / 'tmp' / '09' / "enwiki-20150112-400-r10-105752.txt.bz2",
    'raw_txt': Path.home() / 'tmp' / '09' / "enwiki-20150112-400-r10-105752.txt",
}

# 初期設定
def init():
    logger.info("set directory")
    if not files['dir'].is_dir():
        files['dir'].mkdir(parents=True)

    logger.info("Download dataset")
    if not files['raw_txt'].exists():
        logger.info("download bz2 file from Internet")
        if not files['raw_bz2'].exists():
            url = "http://www.cl.ecei.tohoku.ac.jp/nlp100/data/enwiki-20150112-400-r10-105752.txt.bz2"
            urllib.request.urlretrieve(url, files['raw_bz2'])

        logger.info("decompress bz2 -> txt")
        with open(files['raw_bz2'], 'rb') as f, open(files['raw_txt'], 'wb') as txt:
            txt.write(bz2.decompress(f.read()))

        logger.info("remove temporal file")
        files['raw_bz2'].unlink()

    logger.info('now environment is ready!')

# 環境削除
def clean():
    logger.info("delete environment")
    for f in files['dir'].iterdir():
        f.unlink()

    files['dir'].rmdir()
    logger.info('environment removed')

if __name__ == '__main__':
    init()