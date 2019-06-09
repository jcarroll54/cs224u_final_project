from nltk.tree import Tree
import json

def str2tree(s, binarize=False):
    if not s.startswith('('):
        s = "( {} )".format(s)
    if binarize:
        s = s.replace("(", "(X")
    return Tree.fromstring(s)


def check_hypothesis(hypo_tree):
    hypo_pos = hypo_tree.pos()

    for i in range(1, len(hypo_pos) - 1):
        prev_elem = hypo_pos[i-1]
        elem = hypo_pos[i]
        next_elem = hypo_pos[i+1]

        if elem[0] not in ['is', 'are', 'was', 'were', 'are', 'am']:
            continue

        # elem is currently the IS
        if prev_elem[0] in ['there', 'There']:
            return False

        if next_elem[1] == 'IN':
            return False

        return True

if __name__ == '__main__':
    matching_hypotheses = []
    with open('data/nlidata/snli_1.0/parsed_breaking_dataset.jsonl') as f:
        while True:
            line = f.readline()
            if not line:
                break

            json_line = json.loads(line)

            label = json_line['gold_label']

            if label not in ['contradiction', 'neutral', 'entailment']:
                continue # skip over the '-' label

            hypothesis = json_line['sentence2']
            hypothesis_parse = json_line['sentence2_parse']
            hypothesis_tree = str2tree(hypothesis_parse)

            if (check_hypothesis(hypothesis_tree)):
                matching_hypotheses.append(json_line)

        f.close()

    with open('data/nlidata/snli_1.0/breaking_is_a_test.jsonl', 'w') as f:
        for x in matching_hypotheses:
            print(x)
            json.dump(x, f)
            f.write('\n')
        f.close()



