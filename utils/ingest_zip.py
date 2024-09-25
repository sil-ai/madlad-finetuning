from zipfile import ZipFile
import os
import tempfile
import io
from pathlib import Path
from machine.corpora import extract_scripture_corpus, ParatextTextCorpus



def to_vrefs(tmp_pt):
    corpus = ParatextTextCorpus(tmp_pt.resolve())
    output = list(extract_scripture_corpus(corpus))
    output_file = b'\n'.join(line.encode() for line, _, _ in output)
    output_file += b'\n'
    return output_file


def convert_paratext_to_vref(file):
    with tempfile.TemporaryDirectory() as tmpdirname:
        with ZipFile(io.BytesIO(file), 'r') as zip:
            zip.extractall(tmpdirname)

            dirs = os.listdir(tmpdirname)
            dirs = [dir for dir in dirs if dir != '__MACOSX']

            if len(dirs) == 1:
                tmp_pt = Path(tmpdirname) / dirs[0]
            else:
                tmp_pt = Path(tmpdirname)

            for filename in os.listdir(tmp_pt):
                if filename.endswith('.BAK') or filename.endswith('.bak'):
                    os.remove(os.path.join(tmp_pt, filename))

        output_file = to_vrefs(tmp_pt)
    
    return output_file
