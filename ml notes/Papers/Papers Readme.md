 [These AI/ML papers give you an unfair advantage](https://www.youtube.com/@gptLearningHub)

- All papers in time sequence and field wise - https://github.com/dair-ai/ML-Papers-Explained

https://chuanenlin.medium.com/2022-top-ai-papers-a-year-of-generative-models-a7dcd9109e39 


Verygood Paper Implementations https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master#paper-implementations 
 \

https://github.com/hurshd0/must-read-papers-for-ml
https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap

- [https://mandyliu.substack.com/](https://mandyliu.substack.com/)

[https://github.com/hurshd0/must-read-papers-for-ml](https://github.com/hurshd0/must-read-papers-for-ml)

 [https://medium.com/dataseries/24-really-f-ing-interesting-deep-learning-papers-fa26afe3a070](https://medium.com/dataseries/24-really-f-ing-interesting-deep-learning-papers-fa26afe3a070)

- [ ]  [https://towardsdatascience.com/7-best-research-papers-to-read-to-get-started-with-deep-learning-projects-59e11f7b9c32](https://towardsdatascience.com/7-best-research-papers-to-read-to-get-started-with-deep-learning-projects-59e11f7b9c32)

 [https://analyticsindiamag.com/5-important-deep-learning-research-papers-you-must-read/](https://analyticsindiamag.com/5-important-deep-learning-research-papers-you-must-read/)

 - [ ] [https://medium.com/@the.datascientist/10-must-read-research-papers-for-deep-learning-developers-237a57874ac5](https://medium.com/@the.datascientist/10-must-read-research-papers-for-deep-learning-developers-237a57874ac5)

 [https://github.com/terryum/awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers)

[https://colah.github.io/](https://colah.github.io/)

 [https://towardsdatascience.com/10-papers-you-should-read-to-understand-image-classification-in-the-deep-learning-era-4b9d792f45a7](https://towardsdatascience.com/10-papers-you-should-read-to-understand-image-classification-in-the-deep-learning-era-4b9d792f45a7)

 [https://mindfulmodeler.substack.com/p/my-perfectly-imperfect-note-taking](https://mindfulmodeler.substack.com/p/my-perfectly-imperfect-note-taking)

[https://github.com/yassouali/ML-paper-notes?tab=readme-ov-file](https://github.com/yassouali/ML-paper-notes?tab=readme-ov-file)


## Started with :

### Resnet: Deep Residual Learning for Image Recognition


### AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE 

https://arxiv.org/pdf/2010.11929
https://github.com/google-research/vision_transformer

> The dominant approach is to pre-train on a large text corpus and then fine-tune on a smaller task-specific datase

[[VIT image is worth 16x16 words transformer for images recognition at scale.pdf#page=1&selection=82,58,83,70|VIT image is worth 16x16 words transformer for images recognition at scale, page 1]]

> Inspired by NLP successes, multiple works try combining CNN-like architectures with self-attention (Wang et al., 2018; Carion et al., 2020), some replacing the convolutions entirely (Ramachandran et al., 2019; Wang et al., 2020a)

[[VIT image is worth 16x16 words transformer for images recognition at scale.pdf#page=1&selection=88,43,90,73|VIT image is worth 16x16 words transformer for images recognition at scale, page 1]]

> Therefore, in large-scale image recognition, classic ResNet- like architectures are still state of the art (Mahajan et al., 2018; Xie et al., 2020; Kolesnikov et al., 2020)

[[VIT image is worth 16x16 words transformer for images recognition at scale.pdf#page=1&selection=92,43,94,5|VIT image is worth 16x16 words transformer for images recognition at scale, page 1]]

> To do so, we split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Trans- former. 

[[VIT image is worth 16x16 words transformer for images recognition at scale.pdf#page=1&selection=96,71,98,8|VIT image is worth 16x16 words transformer for images recognition at scale, page 1]]

![[Pasted image 20241117044533.png]]

VIT
> An overview of the model is depicted in Figure 1. The standard Transformer receives as input a 1D sequence of token embeddings. 
> - To handle 2D images, we reshape the image x ∈ RH×W ×C into a sequence of flattened 2D patches xp ∈ RN ×(P 2·C), where (H, W ) is the resolution of the original image, C is the number of channels, (P, P ) is the resolution of each image patch, and N = HW/P 2 is the resulting number of patches, which also serves as the effective input sequence length for the Transformer. 
> - The Transformer uses constant latent vector size D through all of its layers, so we flatten the patches and map to D dimensions with a trainable linear projection (Eq. 1). We refer to the output of this projection as the patch embeddings.
> - The Transformer encoder (Vaswani et al., 2017) consists of 
> 	- alternating layers of multiheaded self- attention (MSA, see Appendix A) and 
> 	- MLP blocks (Eq. 2, 3). 
> 	- Layernorm (LN) is applied before every block, and 
> 	- residual connections after every block (Wang et al., 2019; Baevski & Auli, 2019).
> 	- The MLP contains two layers with a GELU non-linearity
> -   
[[VIT image is worth 16x16 words transformer for images recognition at scale.pdf#page=3&selection=100,0,164,13|VIT image is worth 16x16 words transformer for images recognition at scale, page 3]]

Implementation
- 1st GitHub https://github.com/tintn/vision-transformer-from-scratch/tree/main 
	- Local: [[Papers/vision-transformer-from-scratch/readme|readme]]
	- https://tintn.github.io/Implementing-Vision-Transformer-from-Scratch/
	
- 2nd GitHub https://github.com/0xD4rky/Vision-Transformers?tab=readme-ov-file
- Medium blog PyTorch VIT from scratch [Link](https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)
- 

- research.google.com/github/google-research/vision_transformer/blob/main/vit_jax.ipynb#scrollTo=64hwCdaehs42
- [Colab github/google-research/vision_transformerit_jax_augreg.ipynb](https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax_augreg.ipynb)

Links:
[[Extra]]