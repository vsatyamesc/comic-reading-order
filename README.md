# comic-reading-order
A Git Repo to host the code for Synthesizing an Training a Transformer Model for Predicting the Sequence of Reading Comics Bubbles and Panels. 

## :eight_spoked_asterisk: Synthesize Images According to Scripts :eight_spoked_asterisk:
## Why? :confused:
I plan to automate the process of translation of comics because I like them, most of them are not available for free to read or their english translation isn't even available.

What to do then??

I trained Comic Bubble Detector and Comic panel detector, use it with OCR model and Translation Models, to pipeline the process of getting the translated texts. However, there was a new problem, Translation models require sequential context, the Detector models don't give the bubbles in sequence so I had to do something. 

Thus trained this, It is named "Comic Reading Order", it is basically for Panels, but can be used for bubbles too. This gives takes the Panels/Bubbles and gives their sequence. So I can sort them and feed them to translators.

## How did I get the Datasets?? :sweat_smile:

Comic Bubble - Well I did manual job and labeled them myself using LabelImg. I labeled a few of them and trained a YOLO model, then synthesized more dataset based on it.

Panel Detector - Same as above

Unfortunately, I've lost the above Two Dataset files because my Hard drive corrupted.

Reading Sequencer :question: :question: - 

It is a hard task, there weren't any models I could use, I tried the same method as above however they weren't working, I couldn't get proper results out of transformer even after I had manually labeled over 200 actual comic image. I realized that I have to use at least 1000 images for it to work, it is too much of a menial labor for just 1 person.

Dataset - https://huggingface.co/datasets/vsatyamesc/Comic-Panel-Sequence-LtoR 

:heavy_check_mark: Solution: 

Create my own manga panels generator lol, you might ask can't I just code the algorithm to calculate the sequence from the given yolo panel/bubble coordinates?? Well Yes but no, it will be too complex with too many edge cases, best thing is to train an ML model for it. there are too many stuffs to consider in such cases.

Generator Working :recycle:

Generates a Grid, and then randomly produces panels based on probability and randomizes their sizes, this way, I can get the Grid positions as required data to solve it as a graph problem to get the reading sequence as a feature for training a model in it. This way, I'm able to generate A very big amount of Dataset easily, however I'm using it as a Generalization mode, to generalize simple rules, then train on real datasets.

## Training

The training is pretty easy, just give it datasets and it'll train, follow the convention
for each training .txt file, Max 20 panels, store it sequentially
> Weight X_Center Y_Center Width Height

Set the Training Model Name etc. Just Read the Code, in ```def train_model```, The Hyperparameters are set according to the best result produced by the corresponding model.

The Dataset Loader has a built-in Augmenter to augment the dataset to give it variety.

Saves the Training metrics in ```training_metrics_.png```
