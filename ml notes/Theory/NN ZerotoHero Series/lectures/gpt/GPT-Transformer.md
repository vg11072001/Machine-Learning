## Roots:

- Attention Is All You Need : [PDF](nn-zero-to-hero/lectures/11.Attention-All-You-Need.pdf)
- [github.com/karpathy/LLM101n](https://github.com/karpathy/LLM101n)

---
## 1. 3Blue1Brown 

### Attention in transformers, visually explained | DL6 [Link](https://www.youtube.com/watch?v=eMlx5fFNoYc&t=269s)
============================================================================

Here are a few other relevant resources

- What does it mean for computers to un...  @vcubingx 
- [ChatGPT: 30 Year History (How AI Lear...  ](https://www.youtube.com/watch?v=OFS90-FX6pg) @ ArtOfTheProblem
- **how directions in embedding spaces have meaning** : Efficient Estimation of Word Representations in Vector Space : https://arxiv.org/pdf/1301.3781.pdf

If you're interested in the herculean task of interpreting what these large networks might actually be doing, the Transformer Circuits posts by Anthropic are great. In particular, it was only after reading one of these that I started thinking of the combination of the value and output matrices as being a combined low-rank map from the embedding space to itself, which, at least in my mind, made things much clearer than other sources.
- A Mathematical Framework for Transformer Circuits : https://transformer-circuits.pub/2021/framework/index.html
- https://transformer-circuits.pub/

#### Links to explore:
- A Mathematical Framework for Transformer Circuits : https://transformer-circuits.pub/2021/framework/index.html


### How might LLMs store facts | DL7 [Link](https://www.youtube.com/watch?v=9-Jl0dxWQs8&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=8)
============================================================================

Unpacking the multilayer perceptron's in a transformer, and how they may store facts

Sections:
* 0:00 - Where facts in LLMs live
* 2:15 - Quick refresher on transformers
* 4:39 - Assumptions for our toy example
* 6:07 - Inside a multilayer perceptron
* 15:38 - Counting parameters
* 17:04 - Superposition
* 21:37 - Up next


Resources: 
- AI Alignment forum post from the Deepmind researchers referenced at the video's start:
	- Fact Finding: Attempting to Reverse-Engineer Factual Recall on the Neuron Level [(Post 1)](https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall)

- Anthropic posts about superposition referenced near the end:
	- https://transformer-circuits.pub/2022/toy_model/index.html 
	- https://transformer-circuits.pub/2023/monosemantic-features


Some added resources for those interested in learning more about mechanistic interpretability, offered by Neel Nanda

- Mechanistic interpretability paper reading list
    - An Extremely Opinionated Annotated List of My Favorite Mechanistic Interpretability [Papers v2](https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite)

- Getting started in mechanistic interpretability
	- https://www.neelnanda.io/mechanistic-interpretability/getting-started 

- An interactive demo of sparse autoencoders (made by Neuronpedia)
	- https://www.neuronpedia.org/gemma-scope#main

- Coding tutorials for mechanistic interpretability (made by ARENA)
	- https://arena3-chapter1-transformer-interp.streamlit.app/

---
## 2. Jay Alamar Language AI & NLP Playlist [Link](https://www.youtube.com/playlist?list=PLTx9yCaDlo1UlgZiSgEjq86Zvbo2yC87d)

- https://jalammar.github.io/
- https://newsletter.languagemodels.co/ 

### 2nd Transformer Video

- Blog Post https://jalammar.github.io/illustrated-transformer/
  
- ![alt text](image-1.png)
- ![alt text](image-2.png)
#### Go Forth And Transform
I hope you’ve found this a useful place to start to break the ice with the major concepts of the Transformer. If you want to go deeper, I’d suggest these next steps:
- Read the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper, the Transformer blog post ([Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)), and the [Tensor2Tensor announcement](https://ai.googleblog.com/2017/06/accelerating-deep-learning-research.html).
- Watch [Łukasz Kaiser’s talk](https://www.youtube.com/watch?v=rBCqOTEfxvg) walking through the model and its details
- Play with the [Jupyter Notebook provided as part of the Tensor2Tensor repo](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)
- Explore the [Tensor2Tensor repo](https://github.com/tensorflow/tensor2tensor).

#### Follow-up works:
- [Depthwise Separable Convolutions for Neural Machine Translation](https://arxiv.org/abs/1706.03059)
- [One Model To Learn Them All](https://arxiv.org/abs/1706.05137)
- [Discrete Autoencoders for Sequence Models](https://arxiv.org/abs/1801.09797)
- [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198)
- [Image Transformer](https://arxiv.org/abs/1802.05751)
- [Training Tips for the Transformer Model](https://arxiv.org/abs/1804.00247)
- [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
- [Fast Decoding in Sequence Models using Discrete Latent Variables](https://arxiv.org/abs/1803.03382)
- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)

### Others: Explainable AI Cheat Sheet - Five Key Categories [link](https://www.youtube.com/watch?v=Yg3q5x7yDeM&list=PLTx9yCaDlo1UN8ACYqzjq2tj2m2sVr3gB&index=1)

![](Pasted%20image%2020241206183838.png)

![](Pasted%20image%2020241206183900.png)

Resources : 
- The Explainable AI Cheat Sheet: [https://ex.pegg.io](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqazNxcmVsTWtWY2FiaURDeGgtQzBlbXBKa1JSd3xBQ3Jtc0tuaWdiZ1IycGVBMG8tempGYzVEdHAxOFRYY0M5aGZYZ2ozT2c0T1pTUldKX0NGaTc2UE9GOGJtbUhNLXVIRXZZNzV4Y2F1Qkt5RGhKcXdrWFl5ZU94d0NkMmN2WXJKMEVUQmNCVjFhTm9VTmZOM0hWdw&q=https%3A%2F%2Fex.pegg.io%2F&v=Yg3q5x7yDeM) 
- Mirror: [https://github.com/arpeggiohq/explain...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbWtROGFmUDZ3NlZzbTJOTk1WZVBncEpESG56QXxBQ3Jtc0ttcHItM3BEUThibUZiTjVjdDk2dUZobGo1b1VFenlWZllUbGV0TlU1bzRaUVV0NkNpQTFRZDV2bXpkX2VHODA1ajNteXZNUERwRkZuSnBlYWJ5b2xON182SktaSHVlLVJhQnN2TmdlMDRlajNteTlQYw&q=https%3A%2F%2Fgithub.com%2Farpeggiohq%2Fexplainabile-ai&v=Yg3q5x7yDeM) 
- Interpretable Machine Learning Book (Christoph Molnar): [https://christophm.github.io/interpre...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbWg3ZXoxNU14eWFCX0VFY2JQSVJpNktMQUpIQXxBQ3Jtc0ttV1RuczZXQVRIUEwtY3ZQTTZGQ1NCNmxRaUxQVTFWZDhzVEZWTUllUWFuaXRGTzJIYTB2LWlNdjlnZGlWSHo2YnNEWGtTclB2Vm02WFVBcTQ5Tkh2dDRwdFItQzQ5eTRYaTZoZk9IcUNkV3BPQjFxYw&q=https%3A%2F%2Fchristophm.github.io%2Finterpretable-ml-book%2F&v=Yg3q5x7yDeM) 
- Explainability for NLP (Isabelle Augenstein):    [Practical Talk 1: Explainability for ...](https://www.youtube.com/watch?v=XiaTNOHPpuA&t=0s)   
- NLP Highlights: Interpreting NLP Model Predictions (Sameer Singh):   [117-interpreting-nlp-model-predictions-wit...](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbE9hdWJMcmk2VEpIdnZYelNqdnNkRlRBWDAtUXxBQ3Jtc0tuVnFVTTFIdnJha2dhUkh3VG9sdGkwV092d0J0eWpSQmVoUURBMlFEOXp4Q1hUT0xsTGRobk85cjk0Nmd0cV9VazBWdTlqemtDMWVwcXhTdmxoWWtENVpESkNsb1RzMFJscXNZRUdhQVdGaDF2SElCYw&q=https%3A%2F%2Fsoundcloud.com%2Fnlp-highlights%2F117-interpreting-nlp-model-predictions-with-sameer-singh&v=Yg3q5x7yDeM)   Please Stop Doing "Explainable" ML (Cynthia Rudin):    [Please Stop Doing "Explainable" ML - ...](https://www.youtube.com/watch?v=I0yrJz8uc5Q&t=0s)



## 3. Courses 

### Generative AI with LLM [coursera](https://www.coursera.org/learn/generative-ai-with-llms)

### Large Language Models: Apply and Implement Strategies for Large Language Models [Book]()

- GitHub: https://github.com/peremartra/Large-Language-Model-Notebooks-Course/tree/main

### Finetuning Large Language Models [coursera]

### Stanford CS25 - Transformers United [Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM)
- https://web.stanford.edu/class/cs25/
- Aligning Open Language Models : ![](Pasted%20image%2020241206184535.png)

### Stanford CS224N: Natural Language
## 4. Medium/ other blogs: 

### a. Amazing Things Happen When Attention Heads Are Supercharged Using Mixture-Of-Experts [Link](https://levelup.gitconnected.com/amazing-things-happen-when-attention-heads-are-supercharged-using-mixture-of-experts-b55a6b9a0ac8)

#### Further Reading
- [Research paper titled ‘MoH: Multi-Head Attention as a Mixture of Head Attention’ published in ArXiv](https://arxiv.org/abs/2410.11842v1)
- [GitHub repository containing the code for MoH Attention](https://github.com/SkyworkAI/MoH)
- [Research paper titled ‘Mixture of Attention Heads: Selecting Attention Heads Per Token’ published in ArXiv](https://arxiv.org/pdf/2210.05144)

### b. RAG From Scratch [Link](https://pub.towardsai.net/rag-from-scratch-66c5eff02482)
- Good explanations


### c. LLAMA 2 from gorund up [Link](https://cameronrwolfe.substack.com/p/llama-2-from-the-ground-up)


### d. best models on chatbot arena when LLM act as a judge : **`google/gemma-2-9b-it`**, **`google/gemma-2-27b-it`**, and **`RLHFlow/ArmoRM-Llama3-8B-v0.1`**