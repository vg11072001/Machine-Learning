https://cs231n.github.io/
Lecture Notes: [Github](https://github.com/cs231n/cs231n.github.io/tree/master?tab=readme-ov-file)
Assignments: [GitHub](https://github.com/Jaskanwal/stanford-CS231N-2023?tab=readme-ov-file)


### Spring 2024 Assignments

Assignment #1: Image Classification, kNN, SVM, Softmax, Fully Connected Neural Network [Link](https://cs231n.github.io/assignments2024/assignment1/)

Assignment #2: Fully Connected and Convolutional Nets, Batch Normalization, Dropout, Pytorch & Network Visualization [Link](https://cs231n.github.io/assignments2024/assignment2/)

Assignment #3: Network Visualization, Image Captioning with RNNs and Transformers, Generative Adversarial Networks, Self-Supervised Contrastive Learning [Link](https://cs231n.github.io/assignments2024/assignment3/)

### Module 1: Neural Networks

1. [Image Classification: Data-driven Approach, k-Nearest Neighbor, train/val/test splits](https://cs231n.github.io/classification/) 
	- [[1.cs231n-github-io-Image-classification.pdf]]
	- L1/L2 distances, 
	- hyperparameter search, 
	- cross-validation

2. [Linear classification: Support Vector Machine, Softmax](https://cs231n.github.io/linear-classify/)
	- [[1.cs231n-github-io-Linear-classification.pdf]]
	- parametric approach, 
	- bias trick, 
	- hinge loss, 
	- cross-entropy loss, 
	- L2 regularization, 
	- web demo

3. [Optimization: Stochastic Gradient Descent](https://cs231n.github.io/optimization-1/)
	- [[1.cs231n-github-io-optimizations-sgd.pdf]]
	-  optimization landscapes, 
	- local search, 
	- learning rate, 
	- analytic/numerical gradient

4. [Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/)
	- [[1.cs231n-github-io-optiomization-backprop.pdf]]
	- chain rule interpretation, 
	- real-valued circuits, 
	- patterns in gradient flow

5. [Neural Networks Part 1: Setting up the Architecture](https://cs231n.github.io/neural-networks-1/)
	- [[1.cs231n-github-io-nn1-architecture.pdf]]
	- model of a biological neuron, activation functions, neural net architecture, representational power

6. [Neural Networks Part 2: Setting up the Data and the Loss](https://cs231n.github.io/neural-networks-2/)
	- [[1.cs231n-github-io-nn1-data-and-loss.pdf]]
	- preprocessing, weight initialization, batch normalization, regularization (L2/dropout), loss functions

7. [Neural Networks Part 3: Learning and Evaluation](https://cs231n.github.io/neural-networks-3/)
	- [[1.cs231n-github-io-nn1-learning-evaluation.pdf]]
	- gradient checks, sanity checks, babysitting the learning process, momentum (+nesterov), second-order methods, Adagrad/RMSprop, hyperparameter optimization, model ensembles

8. [Putting it together: Minimal Neural Network Case Study](https://cs231n.github.io/neural-networks-case-study/)
	- [[1.cs231n-github-io-nn-case-study.pdf]]
	- minimal 2D toy data example

### Module 2: Convolutional Neural Networks

####  [Convolutional Neural Networks: Architectures, Convolution / Pooling Layers](https://cs231n.github.io/convolutional-networks/)
- [[2.cs231n-github-io-cnn-architecture.pdf]], 
- Topics 
	- layers, 
	- spatial arrangement, 
	- layer patterns, 
	- layer sizing patterns, 
	- AlexNet/ZFNet/VGGNet case studies, 
	- computational considerations

##### CNN Architecture by 2019 lecture
- [8.498_FA2019_lecture08-cnn-architectiure](8.498_FA2019_lecture08-cnn-architectiure.pdf)
##### Help: Convolution Layer process by  [YouTube](https://www.youtube.com/watch?v=KuXjwB4LzSA) 3Blue1Brown:
- But what is a convolution? 
- ![[Pasted image 20241118184914.png]]
- Performed the convolution for Probability distribution
- Image processing
- Getting the moving average
- equation multiplication

- For **image processing** which we one extra matrix which we call kernel where we apply convolve with the color matrix of that part of the images.
- ![[Pasted image 20241118185626.png]] It will blur the image resulting
- the matrix sum is zero if we added negative and positive distribution number, and the sum of that matrix is 0.
- we can add effect Edge detection on vertical and horizontal effect![[Pasted image 20241118185828.png]]
- ![[Pasted image 20241118185940.png]]
- Different kernel give different effects on images- edge detection, blurring, sharpening, etc.

- Extra using fast Fourier transformation we get the O(NlogN) computation![[Pasted image 20241118190236.png]]
- ![[Pasted image 20241118190429.png]]
##### Architecture - FC
- ![[Pasted image 20241119072943.png]]
- ![[Pasted image 20241119073211.png]]
- ![[Pasted image 20241119073453.png]]
- ![[Pasted image 20241119073656.png]]
- ![[Pasted image 20241119073722.png]]
- ![[Pasted image 20241119074059.png]]
- ![[Pasted image 20241119074330.png]]
- 
##### Case Study of AlexNet/VGGNet
- Check out pdfs [8.498_FA2019_lecture08-cnn-architectiure](8.498_FA2019_lecture08-cnn-architectiure.pdf)

- [Understanding and Visualizing Convolutional Neural Networks](https://cs231n.github.io/understanding-cnn/)
	- [[2.cs231n-github-io-undersandingandvisual.pdf]]
	- tSNE embeddings, deconvnets, data gradients, fooling ConvNets, human comparisons

- [Transfer Learning and Fine-tuning Convolutional Neural Networks](https://cs231n.github.io/transfer-learning/)
	- [[2.cs231n-github-io-transferlearning-fine-tune.pdf]]


Student-Contributed Posts

- [Taking a Course Project to Publication](https://cs231n.github.io/choose-project/)
- [Recurrent Neural Networks](https://cs231n.github.io/rnn/)