# BOhT

***
**A Tag-based Methodology for the Detection of User
Repair Strategies in Task-Oriented Conversational
Agents**

In Italian, "boh" means "I don't know/I don't understand", indeed
this project aims to capture misunderstandings during the conversation with a (chat)bot.

This repository contains three neural network models to identify and classify the type of misunderstandings
according to a particular tag schema we developed.

***

<!--## Table of Contents
1. [General Info](#general-info)
2. [Technologies](#technologies)
4. [Usage](#usage)

***
-->

## General info

Mutual  comprehension  is  a  crucial  component  that  makes  a  conversation succeed between the chatbot and the user.
However, the chatbot could fail at the understanding level, when they are not capable to correctly parse an input in natural language.
Humans themselves often have difficulties in understanding the language produced by other speakers.
The mechanism that makes mutual understanding possible is our capability to signal that error
to the counterpart and to manifest the incomprehension in order to initiate a repair of the conversation.
Our goal is thus to transfer a common and spontaneous phenomenon of human communication to the field of human-computer interaction.
We aim at teaching a chatbot o detect repair strategies put into place by users who interact with it,
and then to make the chatbot react appropriately.
To accomplish this task, we developed a hierarchical tag system, where the top ones are:
* Inherit: the repair strategy is related to the message itself, i.e.self-contained.
* Backward: the repair strategy is related  to  previous  messages in the dialogue.
Each tag has subtags that specify the kind of repair strategy.

We then developed three neural network classifiers to recognize:
* the presence of a repair strategy in the conversation (either Inherit or Backward);
* the kind of Inherit strategy analyzing the current message;
* the kind of Backward strategy analyzing the entire conversation.

Each neural network model uses an Italian pre-trained version of Bert as Word Embedding,
followed by either a Recurrent Neural Network or a Convolutional one to generate the sentence representation.
For the Backward model, another Recurrent Neural Network reads all the sentence representations to identify the repair strategy.

***

## Technologies

We used python as programming language. The neural network models are defined using pytorch; we used example_pb2 from tensorflow
to store and to load the dataset.

### Requirements

* [Python](https://www.python.org/) - Python 3.9.1
* [PyTorch](https://pytorch.org/) - PyTorch 1.7.1
* [TensorFlow](https://www.tensorflow.org/) - TensorFlow 1.5
* [scikit-learn](https://scikit-learn.org/stable/) - Scikit-learn
* [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) - HuggingFace Transformers
* [TensorBoard](https://www.tensorflow.org/tensorboard) - TensorBoard
* [matplotlib](https://matplotlib.org/) - Matplotlib
* [numpy](http://www.numpy.org/) - NumPy


***

## Usage

The software can be execute using the following command:

```
python classifier.py
```

The python file *classifier.py* can be followed by one or more parameters. The best set of parameters for each model, and the best models, are memorized in parameters.json file.

For the usage, the relevant parameters are:

| Parameter                 | Description   |
| :------------------------ | :-------------|
| -m --mode 	       | it specifies the running mode of the classifier, can be: train mode (train), test mode (test) or prototype mode (demo);
| -o --output | the output folder path
| -t --tokenizer | pretrained tokenzier to use

For test and demo mode additional parameters are required:
```
--test_data       dataset used for test
--demo_data       dataset used for demo
--checkpoint      pretrained model to use
```

Meanwhile, for training the required parameters are:
```
----train_data_path     dataset used training
--valid_data_path       dataset used for evalutation
--params                path to the json file containing all the configurable hyper-parameters of the network
--classifier            the name of the desired classifier model (the list of classifiers is available in parameters.json)
--epoch                 maximum number of epoch for the training; if it set to -1, an early stopping criterion is used.
--batch                 batch size
--conv                  maximum conversation length
--msg                   maximum message length
```



## Authors

* **Giovanni Siragusa** - [giovanni.siragusa@unito.it](mailto:giovanni.siragusa@unito.it)
* **Roger Ferrod** - [roger.ferrod@unito.it](mailto:roger.ferrod@unito.it)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details