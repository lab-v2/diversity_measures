import pandas

# ======================================
# This function assumes that the files
# extensions are named appropriately and
# the files are formatted correctly.
# ======================================
def read_file(file_path: str):
    split_file_path = file_path.split('.')
    if len(split_file_path) == 0: raise ValueError('file_path does not have an extension')

    file_extension = split_file_path[-1]
    if file_extension == 'jsonl': return pandas.read_json(file_path, lines=True)
    if file_extension == 'json': return pandas.read_json(file_path)
    if file_extension == 'csv': return pandas.read_csv(file_path)