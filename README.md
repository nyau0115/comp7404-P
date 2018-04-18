


# Natural Language Processing - Question Answering System

We are Group P. For the presentation we intend to focus on Dynamic Memory Network, which is a machine learning technique useful for natural language processing.  We will illustrate the concept by implementing a question answering system with the dataset from Wikipedia to allow users to input a question and get a relevant answer from the dataset.

### Natural Language Processing
Natural Language Processing (NLP) is a study focusing on enabling computers to understand, process and analyze natural language in a useful way. It allows the computer to relay the human language and find out the pattern, feature and knowledge.

### Question Answering System

Question Answering System (QAS) retrieves information from an input dataset and output a relevant answer. It is a task of NLP and mainly focused on factoid questions which is simple and factual expression.


# Methodology

We propose to implement the QAS using Dynamic Memory Network (DMN). DMN is a neural network that can solve general QA task with the benefit of its memory system.

## Neural Network Architecture

### Dynamic Memory Network (DMN)

The QAS will be implemented with DMN because it is a mature network for common factoid question answering task.

## Utilities

| file | description |
| --- | --- |
| `server.py` |  |
| `dmn.py` | weights of answer module are tied; trains faster |
| `babi_utils.py` | tools for working with bAbI tasks |
| `helper_utils.py` | helper functions on top of Theano and Lasagne |
| `.theanorc` |

## Installation

***Langauges & Version***

    Python 2.7.10

***Modules Required:***

Theano is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently

    pip install theano

Lasagne is a lightweight library to build and train neural networks in Theano

    pip install install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
    pip install https://github.com/Lasagne/Lasagne/archive/master.zip

Keras is a high-level neural networks API, written in Python and capable of running on top of Theano

    pip install keras

Numpy is the fundamental package for scientific computing with Python.

    pip install numpy

Scipy is a Python-based ecosystem of open-source software for mathematics, science, and engineering.

    pip install scipy

Scikit-Learn is a simple and efficient tools for data mining and data analysis.

    pip install scikit-learn

Flask is a micro web framework written in Python and is used to implement the web application in this project.

    pip install flask

NLTK(Natural Language Toolkit) is a leading platform for building Python programs to work with human language data and is used for sentence splitting in this project.

    pip install nltk

 After installing NLTK, download Punkt Tokenizer Models by running python interpreter in your terminal  with the commands below:

   import nltk
    nltk.download('punkt')

`.theanorc` file contains theano configuration that was used; dmn.py might yield NaN if included `.theanorc` configuration is not used.
definitely make sure floatX is set equal to float32 in `.theanorc` file

## Usage

Training network for bAbI task
Use `train.py` to train a network for bAbI task 1:

    python train.py --network dmn --mode train --babi_id 1

The trained network will be stored in `networks` folder.
bAbI pretrained networks include
task: 1, 2, 3, 6, 11, 14, 17, 18

To run your own webapp locally:

    python server.py

then go to http://127.0.0.1:8000/ in browser
Reminder: Loading pre-trained network may take up to few minutes.

## Data
Dataset used in this project is English bAbI task (10k) from Facebook research

According to Facebook:

**Training Set Size:** For each task, there are 1000 questions for training, and 1000 for testing. However, we emphasize that the goal is to use as little data as possible to do well on the tasks (i.e. if you can use less than 1000 that’s even better) — and without resorting to engineering task-specific tricks that will not generalize to other tasks, as they may not be of much use subsequently. Note that the aim during evaluation is to use the _same_ learner across all tasks to evaluate its skills and capabilities.

**Supervision Signal:** Further while the MemNN results in the paper use full supervision (including of the supporting facts) results with weak supervision would also be ultimately preferable as this kind of data is easier to collect. Hence results of that form are very welcome. E.g.  [this paper](http://arxiv.org/abs/1503.08895)  does include weakly supervised results.

| Task id | Description |
| --- | --- |
| 1 | 1 supporting facts |
| 2 | 2 supporting facts |
| 3 | 3 supporting facts |
| 6 | yes/no questions |
| 11 | basic coreference |
| 14 | time reasoning |
| 17 | positional reasoning |
| 18 | size reasoning |


## Result

| Task id | Test error rate (%) |
| --- | --- |
| 1 | 0.2 |
| 2 | 9.5 |
| 3 | 28.9 |
| 6 | 0.3 |
| 11 | 10.8 |
| 14 | 1.1 |
| 17 | 20.2 |
| 18 | 8.6 |

## Deployment
The project is deployed in AWS: http://52.36.74.244:8002/
 You are welcome to report if there is any issue.

## Acknowledgements

- Karol Kurach, Marcin Andrychowicz & Ilya Sutskever  Neural Random-Access Machines, ICLR, 2016

- Emilio Parisotto & Ruslan Salakhutdinov  Neural Map: Structured Memory for Deep Reinforcement Learning, ArXiv, 2017

- Pritzel et. al. Neural Episodic Control, ArXiv, 2017

- Oriol Vinyals,Meire Fortunato, Navdeep Jaitly  Pointer Networks, ArXiv, 2017

- Jack W Rae et al., Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes, ArXiv 2016

- Antoine Bordes, Y-Lan Boureau, Jason Weston, Learning End-to-End Goal-Oriented Dialog, ICLR 2017

- Junhyuk Oh, Valliappa  Chockalingam, Satinder Singh, Honglak Lee, Control of Memory, Active Perception, and Action in Minecraft, ICML 2016

- Wojciech Zaremba, Ilya Sutskever, Reinforcement Learning Neural Turing Machines, ArXiv 2016

- YerevaNN, Dynamic Memory Networks  in Theano, 2017

- Vinh Khuc, End-To-End Memory Networks for bAbI question-answering tasks, 2017

- Facebook, bAbI Research, 2015