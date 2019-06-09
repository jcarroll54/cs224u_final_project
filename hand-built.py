from nltk.tree import Tree
import json
import utils
from sklearn.metrics import classification_report

# for matching problem
from nltk.corpus import wordnet

def str2tree(s, binarize=False):
    if not s.startswith('('):
        s = "( {} )".format(s)
    if binarize:
        s = s.replace("(", "(X")
    return Tree.fromstring(s)

def find_noun_phrases(tree):
    return [subtree for subtree in tree.subtrees(lambda t: t.label()=='NP')]

def find_head_of_np(np):
    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
    top_level_trees = [np[i] for i in range(len(np)) if type(np[i]) is Tree]
    ## search for a top-level noun
    top_level_nouns = [t for t in top_level_trees if t.label() in noun_tags]
    if len(top_level_nouns) > 0:
        ## if you find some, pick the rightmost one, just 'cause
        return top_level_nouns[-1][0]
    else:
        ## search for a top-level np
        top_level_nps = [t for t in top_level_trees if t.label()=='NP']
        if len(top_level_nps) > 0:
            ## if you find some, pick the head of the rightmost one, just 'cause
            return find_head_of_np(top_level_nps[-1])
        else:
            ## search for any noun
            nouns = [p[0] for p in np.pos() if p[1] in noun_tags]
            if len(nouns) > 0:
                ## if you find some, pick the rightmost one, just 'cause
                return nouns[-1]
            else:
                ## return the rightmost word, just 'cause
                return np.leaves()[-1]

def find_entity_attributes(head, pos, search_distance):
    # iterate through pos to find this head
    idx = 0

    for i in range(0, len(pos)):
        if pos[i][0] == head:
            idx = i
            break

    attribute_tags = ['VBN', 'JJ', 'CD', 'VBG', 'VBZ', 'NNP', 'VBP']

    # search left and right for attribute tags
    attributes = []
    for i in range(1, search_distance + 1):
        left = idx - i
        right = idx + i

        if left > 0 and pos[left][1] in attribute_tags and pos[left][0] not in ['is', 'are', 'was', 'were', 'are', 'am']: 
            attributes.append(pos[left][0])

        if right < len(pos) and pos[right][1] in attribute_tags and pos[right][0] not in ['is', 'are', 'was', 'were', 'are', 'am']:
            attributes.append(pos[right][0])
    return attributes


def find_entities_and_attributes(tree, search_distance):
    entities = {}
    nps = find_noun_phrases(tree)
    for np in nps:
        phrase_head = find_head_of_np(np)
      
        if phrase_head in entities:
            continue

        attributes = find_entity_attributes(phrase_head, tree.pos(), search_distance)
        entities[phrase_head] = attributes

    return entities



def predict(premise_tree, hypothesis_tree, label, search, sim, diff, most_common_label):
    premise_pos = premise_tree.pos()
    hypothesis_pos = hypothesis_tree.pos()

    # 1. match is-a pattern in hypothesis
    hypothesis_entities = find_entities_and_attributes(hypothesis_tree, search)

    # 2. find all the entities (heads) in the premise and their corresponding attributes
    premise_entities = find_entities_and_attributes(premise_tree, search)

    # 3. corefer entities in premise and combine attributes
    

    # 4. Match entities and attributes, make prediction
    # if entities overlap, either contradiction or entailment, else neutral
    # convert entities/attributes to GloVe encodings
    matching_entities = []
    for h_ent in hypothesis_entities:
        h_ent_embedding = wordnet.synsets(h_ent.lower())
        if not h_ent_embedding:
            continue

        for p_ent in premise_entities:
            p_ent_embedding = wordnet.synsets(p_ent.lower())
            if not p_ent_embedding:
                continue

            similarity = h_ent_embedding[0].wup_similarity(p_ent_embedding[0])

            if not similarity:
                continue

            if similarity >= sim:
                matching_entities.append((h_ent, p_ent))

    # TODO: potentially change to contradiction, potentially look for dissimilar entities?
    if len(matching_entities) == 0:
        if most_common_label == 'contradiction':
          return 'neutral'
        else:
          return 'contradiction'

    # then check each attribute of each matching entity
    entailment_evidence = 0
    contradiction_evidence = 0
    for match in matching_entities:
        h_ent, p_ent = match

        for h_attr in hypothesis_entities[h_ent]:
            h_attr_embedding = wordnet.synsets(h_attr.lower())
            
            if not h_attr_embedding:
                continue

            for p_attr in premise_entities[p_ent]:
                p_attr_embedding = wordnet.synsets(p_attr.lower())
            
                if not p_attr_embedding:
                    continue

                similarity = h_attr_embedding[0].wup_similarity(p_attr_embedding[0])

                if not similarity:
                    continue

                # if we happen to find one matching attribute among entities, predict entailment
                if similarity >= sim:
                    entailment_evidence += 1
                elif similarity <= diff:
                    contradiction_evidence += 1

    if entailment_evidence > 0:
        return 'entailment'
    elif contradiction_evidence > 0:
        return 'contradiction'
    else:
        return 'neutral'

def most_common_label(counts):
  most_common_label = 'entailment'
  most_common_count = counts['entailment']

  if (counts['neutral'] > most_common_count):
    most_common_label = 'neutral'
    most_common_count = counts['neutral']

  if (counts['contradiction'] > most_common_count):
    most_common_label = 'contradiction'
    most_common_count = counts['contradiction']

  return most_common_label

def hyper_parameter_search_iteration(search, sim, diff):
    count = 0
    predictions = []
    gold_labels = []

    prediction_counts = {
        "entailment": 0,
        "contradiction": 0,
        "neutral": 0
    }

    with open('data/nlidata/snli_1.0/breaking_is_a_test.jsonl') as f:
        while True:
            line = f.readline()
            if not line:
                break

            json_line = json.loads(line)

            premise = json_line['sentence1']
            hypothesis = json_line['sentence2']
            label = json_line['gold_label']

            premise_tree = str2tree(json_line['sentence1_parse'])
            hypothesis_tree = str2tree(json_line['sentence2_parse'])

            most_common = most_common_label(prediction_counts)
            prediction = predict(premise_tree, hypothesis_tree, label, search, sim, diff, most_common)
            predictions.append(prediction)
            gold_labels.append(label)
            prediction_counts[prediction] += 1

            count += 1

            if (count % 100 == 0):
                print("Processed", count)
        
        f.close()

    report = classification_report(gold_labels, predictions, output_dict=True)
    print(classification_report(gold_labels, predictions))
    return report


if __name__ == '__main__':
    # best params: [5, 0.65, 0.3] 

    # search_range = [2, 3, 4, 5, 6, 7, 8] 
    # sim = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7] 
    # diff = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45] 

    search_range = [5]
    sim = [0.65]
    diff = [0.3]

    best_report = {}
    best_f1_score = 0
    best_params = []

    for x in search_range: 
        for y in sim:
            for z in diff:
                report = hyper_parameter_search_iteration(x, y, z)
                score = report['macro avg']['f1-score']

                if score > best_f1_score:
                    best_f1_score = score
                    best_params = [x, y, z]
                    best_report = report
                    print('best score so far:', str(best_f1_score))

    
    print(best_params, best_report)



