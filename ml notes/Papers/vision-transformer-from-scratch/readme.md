# Vision Transformer from Scratch [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10AYlqsACfMiuMiMSVQjcW8NnkGiJrHLh?usp=sharing)


- ![[Pasted image 20241117044533.png]]


``` python 
"""
- Summary for configuration parameters  for training the model
- patch_size: The size of the patches extracted from the input image
- hidden_size: The size of the hidden layers
- num_hidden_layers: The number of hidden layers
- num_attention_heads: The number of attention heads
- interediate_size:
- hidden_dropout_prob: The dropout probability for the hidden layers
- attention_probs_dropout_prob: The dropout probability for the attention probabilities
- initializer_range: The range for the random initialization of the weights
- image_size: The size of the input image
- num_classes: The number of classes in the dataset
- num_channels: The number of channels in the input image
- qkv_bias: Whether to include bias in the query, key, and value projections
- use_faster_attention: Whether to use the faster implementation of the attention mechanism
- The configuration parameters are used to create the model and are saved in the experiment directory along with the model checkpoints and training metrics.
"""
config = {
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10, # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}

argument= {
	"batch_size": 256,
	"epochs":100,
	"lr":1e-2,
	"weight_decay":1e-2
	"devive":gpu,
	"save_model_every_freq": 3, # to save the model checkpoints
	"exp_name": vit_testing # name of the result directory 
}

# step 1 Design
trainloader, testloader = prepare_data(batch_size):

# step 2 Design
model = VitforClassifier(config)

# step 3 Design
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
loss_func = nn.CrossEntropyLoss()

# step 4 Design
trainer = Trainer(model, optimizer, loss_func, exp_name, device)

# step 5 Engine
trainer.train(trainloader, testloader, 
			  epochs, 
			  save_model_every_freq = save_model_every_second)



```


### Dimensions: 

| Layer                        | Dimension          | Comments                                |
| ---------------------------- | ------------------ | --------------------------------------- |
| input of images data set     | (50000, 3, 32, 32) |                                         |
| batch divison                | (256, 3, 32, 32)   |                                         |
| x is passed to model         |                    |                                         |
| **model to embedding layer** |                    |                                         |
| patch embedding : conv2d     | (256, 48, 8, 8)    | kernel = 4, strid = 4, hidden_size = 48 |
| flatten and transporse 1,2   | (246, `8*8`, 48)   |                                         |
| clf token of size            | (256, 1, 48)       |                                         |
| cat(x and clf) dim=1         | (256, `8*8`+1, 48) |                                         |
| position_embeddings          | (256, `8*8`+1, 48) | x + position_embeddings happens         |
|                              |                    |                                         |
| **Encoder layer**            |                    |                                         |
|                              |                    |                                         |










Inspired from Tin Nguyen [github](https://github.com/tintn/vision-transformer-from-scratch/tree/main) 
This is a simplified PyTorch implementation of the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). The goal of this project is to provide a simple and easy-to-understand implementation. The code is not optimized for speed and is not intended to be used for production.

Check out [this post](https://medium.com/towards-data-science/implementing-vision-transformer-vit-from-scratch-3e192c6155f0) for step-by-step guide on implementing ViT in detail.

## Usage

Dependencies:
- PyTorch 1.13.1 ([install instructions](https://pytorch.org/get-started/locally/))
- torchvision 0.14.1 ([install instructions](https://pytorch.org/get-started/locally/))
- matplotlib 3.7.1 to generate plots for model inspection

Run the below script to install the dependencies
```bash
pip install -r requirements.txt
```

You can find the implementation in the `vit.py` file. The main class is `ViTForImageClassification`, which contains the embedding layer, the transformer encoder, and the classification head. All of the modules are heavily commented to make it easier to understand.

The model config is defined as a python dictionary in `train.py`, you can experiment with different hyperparameters there. Training parameters can be passed using the command line. For example, to train the model for 10 epochs with a batch size of 32, you can run:

```bash
python train.py --exp-name vit-with-10-epochs --epochs 10 --batch-size 32
```

Please have a look at the `train.py` file for more details.

## Results

The model was trained on the CIFAR-10 dataset for 100 epochs with a batch size of 256. The learning rate was set to 0.01 and no learning rate schedule was used. The model config was used to train the model:

```python
config = {
    "patch_size": 4,
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10,
    "num_channels": 3,
    "qkv_bias": True,
}
```

The model is much smaller than the original ViT models from the paper (which has at least 12 layers and hidden size of 768) as I just want to illustrate how the model works rather than achieving state-of-the-art performance.

These are some results of the model:

![](/assets/metrics.png)
*Train loss, test loss and accuracy of the model during training.*

The model was able to achieve 75.5% accuracy on the test set after 100 epochs of training.

![](/assets/attention.png)
*Attention maps of the model for different test images*

You can see that the model's attentions are able to capture the objects from different classes pretty well. It learned to focus on the objects and ignore the background.

These visualizations are generated using the notebook `inspect.ipynb`.

