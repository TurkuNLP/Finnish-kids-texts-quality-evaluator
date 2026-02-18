
#imports
from vllm import LLM
import json
import sys


def main(cmd_args):
    MODEL_PATH = cmd_args[0]
    llm = LLM(model=MODEL_PATH, max_model_len=32768, gpu_memory_utilization=0.9,)
    source_ds_path = cmd_args[1]
    output_ds_path = cmd_args[2]
    ds_items = []
    with open(source_ds_path, 'r', encoding="UTF-8") as reader:
        for l in reader:
            if len(l) > 1:
                ds_items.append(json.loads(l.strip()))

    base_prompt = "Muokkaa annetusta tekstistä kömpelömpi versio. Pidä huolta siitä että teksti pysyy luettavana. Alla on annettu esimerkki, jossa on ensin annettu teksti ja tämän jälkeen siitä muokattu kömpelömpi versio.\nEsimerkki annetusta tekstistä: Menimme eilen luokan kanssa retkelle. Ensimmäinen kohteemme oli metsä, jossa linnut lauloivat. Opettaja antoi meille pitkän ja kevyen laudan, jota jokainen kantoi vuorollaan. Rakensimme sen avulla pienen sillan puron yli. Jätimme laudan metsään sellaiseen paikkaan, jonka varmasti muistamme seuraavalla retkellä.\nEsimerkki kömpelöstä versiosta: Eilen menimme luokan kanssa retkelle, ja ensimmäinen paikka oli metsä, jossa linnut lauloivat. Opettaja antoi meille pitkän esineen nimeltä lauta, joka oli niin kevyt, että jokainen jaksoi kantaa sitä vuorollaan. Rakensimme laudan avulla pienen sillan puron yli, ja se jäi metsään paikalle, jonka muistamme varmasti seuraavalla retkellä.\nAnnettu teksti:"

    prompts = [base_prompt+x['text'].replace('\n', ' ') for x in ds_items]

    outputs = llm.generate(prompts)
    res_d = []
    for i,o in enumerate(outputs):
        res_d.append({'perturbation_type':'clumsification' ,'model':MODEL_PATH, 'text':o.outputs[0].text, 'og_text':ds_items[i]['text']})

    print("Parsed outputs!")

    with open(output_ds_path, "w") as writer:
        for d in res_d:
            writer.write(json.dumps(d)+'\n')
    
    print("done!")


if __name__ == "__main__":
    main(sys.argv[1:])