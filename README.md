# cs224u_final_project

The data directory and files are not included in this repo due to size constraints. 
To reproduce the experiments, extract the SNLI dataset into a folder called ```data/nlidata```.

Also, extract the "Breaking Dataset" from here [https://github.com/BIU-NLP/Breaking_NLI] and 
place it within the snli_1.0 folder that is created from the step above. Run ```parse_breaking_lines.py```
to add parsed sentences to this dataset (this takes on the order of 7 hours).

Finally, use ```parse.py``` to generate the is-a subset datasets.
