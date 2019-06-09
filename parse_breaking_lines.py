import json
import os



parsed_jsonl = []
with open("data/nlidata/snli_1.0/breaking_dataset.jsonl") as f:
    while True:
        line = f.readline()

        if not line:
          break

        jsonl = json.loads(line)
        print(jsonl)
        

        for sent in ['sentence1', 'sentence2']:
            # put sentence in temp file
            with open('temp.txt', 'w') as tmp:
                tmp.write(jsonl[sent])
                tmp.close()

            # run parser
            os.system("./stanford-parser-full-2018-10-17/lexparser.sh temp.txt > output.txt")

            # read output
            txt = ""
            with open('output.txt') as out:
                while True:
                    line = out.readline()
                    if not line:
                        break
                    txt += line.strip() + " "
                out.close()
            print(txt)
            
            # write output to sentence1_parse or sentence2_parse
            tag = sent + '_parse'
            jsonl[tag] = txt

        parsed_jsonl.append(jsonl)
        # print(jsonl)
        # break
    f.close()

with open('data/nlidata/snli_1.0/parsed_breaking_dataset.jsonl', 'w') as f:
    for jsonl in parsed_jsonl:
        json.dump(jsonl, f)
        f.write('\n')
    f.close()
