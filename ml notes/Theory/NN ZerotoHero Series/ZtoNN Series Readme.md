From Autor: 
	- [/nn-zero-to-hero/README](Theory/NN%20ZerotoHero%20Series/nn-zero-to-hero/README.md)
	- [makemore/README](Theory/NN%20ZerotoHero%20Series/nn-zero-to-hero/lectures/makemore/makemore/README.md)
## 1st video: The spelled-out intro to neural networks and backpropagation: building micrograd
### Useful links
- *micrograd* on GitHub: https://github.com/karpathy/micrograd
- jupyter notebooks built in this video: 

> [!NOTE] Notebooks
> 1. [micrograd_lecture_first_half_roughly.ipynb](lectures/micrograd/micrograd_lecture_first_half_roughly.ipynb)
> 2. [micrograd_lecture_second_half_roughly.ipynb](lectures/micrograd/micrograd_lecture_second_half_roughly.ipynb)
> 

- From author
	- its a basic library which have the concept of neural networks.
	- it contain 2 main files: `engine` and `nn`:
	- `engine` file mainly contain all the mathematics to proceed with operations like backpropagation 
	- `nn` just fined the what neurons, layer and mlp (comb. of layers).
	- PyTorch and tensors are just pieces where we define to optimize the network and efficiency of whole modelling put together.

### Exercises: 
you should now be able to complete the following google collab, good luck!: [https://colab.research.google.com/dri...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbXBBSHZ6aE14SXgwQlIxUDh4Um9VV0dLRU10QXxBQ3Jtc0ttVVZCaFlnZXNkaTVRVHp4M05uakhHOFFnWktleHVWSnU2VXQzQzc3aVJsc1NkSmFGNms4U2FfQ0NmcmszUnZSSzZGYVdSWDJ4VkJqZm1Dc2JBekFBRlc2cnJ2TVg1RGhBYlpnRFRWb2M2djNtVlE0RQ&q=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1FPTx1RXtBfc4MaTkf7viZZD4U2F9gtKN%3Fusp%3Dsharing&v=VMj-3S1tku0)


### Sequence:
##1
Chapters: [00:00:00](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=0s)
intro [00:00:25](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=25s) 
micrograd overview [00:08:08](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=488s) 
derivative of a simple function with one input [00:14:12](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=852s) 
derivative of a function with multiple inputs [00:19:09](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=1149s) 
starting the core Value object of micrograd and its visualization [00:32:10](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=1930s) 
manual backpropagation example #1: simple expression [00:51:10](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=3070s) 
preview of a single optimization step [00:52:52](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=3172s) 
manual backpropagation example #2: a neuron [01:09:02](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=4142s)
implementing the backward function for each operation [01:17:32](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=4652s) 
implementing the backward function for a whole expression graph [01:22:28](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=4948s) 
fixing a backprop bug when one node is used multiple times [01:27:05](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=5225s) 

##2
breaking up a tanh, exercising with more operations [01:39:31](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=5971s) 
doing the same thing but in PyTorch: comparison [01:43:55](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=6235s) 
building out a neural net library (multi-layer perceptron) in micrograd [01:51:04](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=6664s) 
creating a tiny dataset, writing the loss function [01:57:56](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=7076s) 
collecting all of the parameters of the neural net [02:01:12](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=7272s) 
doing gradient descent optimization manually, training the network [02:14:03](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=8043s) 
summary of what we learned, how to go towards modern neural nets [02:16:46](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=8206s) 
walkthrough of the full code of micrograd on github [02:21:10](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=8470s) 
real stuff: diving into PyTorch, finding their backward pass for tanh [02:24:39](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=8679s) 
conclusion [02:25:20](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1&t=8720s) 
outtakes :)


## 2nd video:  The spelled-out intro to language modeling: building makemore

- language level language model
- covered :
	-  Bigram (one character predicts the next one with a lookup table of counts)

> [!NOTE] Notebooks
> 1. [makemore_part1_bigrams.ipynb](lectures/makemore/makemore_part1_bigrams.ipynb) 

- From Author:
- We implement a bigram character-level language model, which we will further complexify in follow-up videos into a modern Transformer language model, like GPT. In this video, the focus is on
	- (1) introducing torch.Tensor and its subtleties and use in efficiently evaluating neural networks and 
	- (2) the overall framework of language modeling that includes model training, sampling, and the evaluation of a loss (e.g. the negative log likelihood for classification). Links:

### Useful links for practice:

- Python + Numpy tutorial from CS231n [https://cs231n.github.io/python-numpy...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbWxMRXhzTU5FZ2toR0d3UTBxOHBqb3MtRFI3d3xBQ3Jtc0ttdVFrRVhfLXRNX185VWIxcDVfUEtxb3ZGa1F2MTMta2dNT0NZazJwYVdVbHdPNHQ3Q3lZVndOTVBiOUJPQnJUZzUyZU5QQWxqcGxKZWlwd1d3N3JrSlV0MTUyUGNUeHc2S0puaVJybEtVbW9TMl9lQQ&q=https%3A%2F%2Fcs231n.github.io%2Fpython-numpy-tutorial%2F&v=PaCmpygFfXo) . We use torch.tensor instead of numpy.array in this video. Their design (e.g. broadcasting, data types, etc.) is so similar that practicing one is basically practicing the other, just be careful with some of the APIs - how various functions are named, what arguments they take, etc. - these details can vary.
- PyTorch tutorial on Tensor https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
- Another PyTorch intro to Tensor https://pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html

### Exercises: 
- **E01**: train a trigram language model, i.e. take two characters as an input to predict the 3rd one. Feel free to use either counting or a neural net. Evaluate the loss; Did it improve over a bigram model? 

- **E02**: split up the dataset randomly into 80% train set, 10% dev set, 10% test set. Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. What can you see? 

- **E03**: use the dev set to tune the strength of smoothing (or regularization) for the trigram model - i.e. try many possibilities and see which one works best based on the dev set loss. What patterns can you see in the train and dev set loss as you tune this strength? Take the best setting of the smoothing and evaluate on the test set once and at the end. How good of a loss do you achieve? 

- **E04**: we saw that our 1-hot vectors merely select a row of W, so producing these vectors explicitly feels wasteful. Can you delete our use of F.one_hot in favor of simply indexing into rows of W? 

- **E05**: look up and use F.cross_entropy instead. You should achieve the same result. Can you think of why we'd prefer to use F.cross_entropy instead? 

- **E06**: meta-exercise! Think of a fun/interesting exercise and complete it.

### Sequence 
[00:00:00](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=0s) intro 
[00:03:03](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=183s) reading and exploring the dataset 
[00:06:24](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=384s) exploring the bigrams in the dataset
[00:09:24](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=564s) counting bigrams in a python dictionary 
[00:12:45](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=765s) counting bigrams in a 2D torch tensor ("training the model")
[00:18:19](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=1099s) visualizing the bigram tensor 
[00:20:54](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=1254s) deleting spurious (S) and (E) tokens in favor of a single . token 
[00:24:02](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=1442s) sampling from the model
[00:36:17](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=2177s) efficiency! vectorized normalization of the rows, tensor broadcasting 
[00:50:14](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=3014s) loss function (the negative log likelihood of the data under our model) 
[01:00:50](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=3650s) model smoothing with fake counts 
[01:02:57](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=3777s) PART 2: the neural network approach: intro 
[01:05:26](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=3926s) creating the bigram dataset for the neural net 
[01:10:01](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=4201s) feeding integers into neural nets? one-hot encodings 
[01:13:53](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=4433s) the "neural net": one linear layer of neurons implemented with matrix multiplication 
[01:18:46](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=4726s) transforming neural net outputs into probabilities: the softmax 
[01:26:17](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=5177s) summary, preview to next steps, reference to micrograd 
[01:35:49](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=5749s) vectorized loss 
[01:38:36](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=5916s) backward and update, in PyTorch 
[01:42:55](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=6175s) putting everything together
[01:47:49](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=6469s) note 1: one-hot encoding really just selects a row of the next Linear layer's weight matrix [01:50:18](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=6618s) note 2: model smoothing as regularization loss 
[01:54:31](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=6871s) sampling from the neural net 
[01:56:16](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=2&t=6976s) conclusion


## 3rd video: Building makemore Part 2: MLP



> [!NOTE] Notebooks
> 1. [makemore_part2_mlp.ipynb](lectures/makemore/makemore_part2_mlp.ipynb) 
> 2. [a-neural-probabilistic-lang-model-bengio03a](a-neural-probabilistic-lang-model-bengio03a.pdf)


### Useful links:

- PyTorch internals ref [http://blog.ezyang.com/2019/05/pytorc...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbU5MbW5nQ053SnFkeGNfT091c3R0YXBCNnlOUXxBQ3Jtc0tsNFZyNVhYa3pzaUFnb3pVS0UwV2NSTlpyWUlGOEZrYWZfN2JtNDVBamdIQUVjLWFIYndmZ1o0ZzRyU04yS3R4ejc2V2RJdlJKN2hwbTRhQTZrM2gyTjNPMjVLbXJPcWQyMnY4NVZyaDVjNGFMaS1MZw&q=http%3A%2F%2Fblog.ezyang.com%2F2019%2F05%2Fpytorch-internals%2F&v=TCH_1BHY58I)

### Exercises:

- **E01**: Tune the hyperparameters of the training to beat my best validation loss of 2.2
- **E02**: I was not careful with the initialization of the network in this video. 
	- (1) What is the loss you'd get if the predicted probabilities at initialization were perfectly uniform? What loss do we achieve? 
	- (2) Can you tune the initialization to get a starting loss that is much more similar to (1)?
- **E03**: Read the Bengio et al 2003 paper (link above), implement and try any idea from the paper. Did it work?

### Sequence:
[00:00:00](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=0s) intro
[00:01:48](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=108s) Bengio et al. 2003 (MLP language model) paper walkthrough
[00:09:03](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=543s) (re-)building our training dataset 
[00:12:19](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=739s) implementing the embedding lookup table 
[00:18:35](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=1115s) implementing the hidden layer + internals of torch.Tensor: storage, views 
[00:29:15](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=1755s) implementing the output layer 
[00:29:53](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=1793s) implementing the negative log likelihood loss 
[00:32:17](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=1937s) summary of the full network 
[00:32:49](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=1969s) introducing F.cross_entropy and why 
[00:37:56](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=2276s) implementing the training loop, overfitting one batch 
[00:41:25](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=2485s) training on the full dataset, minibatches 
[00:45:40](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=2740s) finding a good initial learning rate 
[00:53:20](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=3200s) splitting up the dataset into train/val/test splits and why 
[01:00:49](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=3649s) experiment: larger hidden layer
[01:05:27](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=3927s) visualizing the character embeddings 
[01:07:16](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=4036s) experiment: larger embedding size 
[01:11:46](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=4306s) summary of our final code, conclusion
[01:13:24](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=4404s) sampling from the model 
[01:14:55](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&t=4495s) google collab (new!!) notebook advertisement


## 4th Video: Building makemore Part 3: Activations & Gradients, BatchNorm



> [!NOTE] Notebooks
> 1. [makemore_part3_bn.ipynb](lectures/makemore/makemore_part3_bn.ipynb) 
> 2. [a-neural-probabilistic-lang-model-bengio03a](a-neural-probabilistic-lang-model-bengio03a.pdf)

- ![](Pasted%20image%2020241123223506.png)



## 5th Video: Building makemore Part 4: Becoming a Backprop Ninja



## 6th video: Building makemore Part 5: Building WaveNet

[wavenet-a-generative-model-for-raw-audio](wavenet-a-generative-model-for-raw-audio.pdf)


## 12th CNN



### Useful links:
- CNN for visulal recognition CV [CNN Readme](CNN%20Readme.md)
- Theory/ML with PyTorch and Sklearn/ch12 : [[Theory/ML with PyTorch and Sklearn/ch12/README|README]]
- Build VIT from scratch [VIT image is worth 16x16 words transformer for images recognition at scale](VIT%20image%20is%20worth%2016x16%20words%20transformer%20for%20images%20recognition%20at%20scale.pdf)  / [Papers Readme](Papers%20Readme.md)
- [Convolutional Neural Network from Scratch](Convolutional%20Neural%20Network%20from%20Scratch.md)
- 