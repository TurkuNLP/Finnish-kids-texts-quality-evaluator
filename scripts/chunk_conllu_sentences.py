from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_chunk import sentence_chunking_main
import json
import os
import re
from tqdm import tqdm
import sys

def extract_sentences_from_file(file_path):
    """
    Extracts sentences from a text file where sentences are in the format "# text = {sentence}".

    Args:
        file_path (str): Path to the text file to be read.

    Returns:
        list: A list of extracted sentences.
    """
    pattern = r'#\s*text\s*=\s*(.*)'  # Regex pattern to match "# text = {sentence}"
    sentences = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                match = re.search(pattern, line.strip())
                if match:
                    sentences.append(match.group(1).strip())
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    return sentences

# Example usage:
# sentences = extract_sentences_from_file("example.txt")
# print(sentences)

def main(cmd_args):

    output_path = cmd_args[0]

    ds_items = []
    TCBC_path = "data/TCBC/"
    with tqdm(range(525), desc="Chunking book sentences...") as pbar:
        for f in os.listdir(TCBC_path):
            book_id = f.replace('.conllu', '')
            sentences = extract_sentences_from_file(TCBC_path+f)
            model = SentenceTransformer('TurkuNLP/sbert-cased-finnish-paraphrase')
        
            chunks, chunk_lens = sentence_chunking_main(512, 0.6, True, sentences, model)

            for i,c in enumerate(chunks):
                ds_items.append({"book_id":book_id, "chunk_id":i, "chunk_len":chunk_lens[i], "text":c})
            pbar.update(1)
            break
    
    with open(output_path, 'w', encoding='utf-8') as writer:
        for d in ds_items:
            writer.write(json.dumps(d)+'\n')


if __name__ == "__main__":
    main(sys.argv[1:])
