# SteganoGAN

This is a reimplementation of the following paper: https://arxiv.org/pdf/1901.03892, for use in our CS 4782 Deep Learning final project. We will attempt to recreate the results found the in the above paper and describe our methodology and work process in doing so.

# Notes

- Tanh in code for basic encoder but not paper
- Also a basicdecoder provided (probably ignore)
- They do LeakyReLu before BatchNorm2d, is a bit weird (could change)
- There is a quantize option, that maybe we should take a look at.
- Multiply encoder_mse by 100 (code) or not (paper)
- Use AdamW over Adam (paper actually was released at the same as AdamW)

Overall Description:
- Basically copying architecture from their GitHub repository, except using some more modern choices like AdamW (we can also switch some of the possible issues we may find)
- Using Div2k X4 over COCO for training time purposes. Also they said Div2k did worse because of some reasons, but it may also just be because of its high quality images, maybe we can get better results with lower quality ones.
- We are going to test all 18 different model combinations that they have, and try it with the same classical analysis tool and Reed-Solomon encryption.
- Going to expand by testing more data depths, maybe graph it in a cool way. Maybe increase the epochs.

TODO:
- Train SteganoGAN on various combinations (we will decide them now)
- Implement Reed-Solomon encryption algorithm (so we can actually test out on text inputs, for some cool examples).
- Obtain the classical steganoanalysis tools, and be ready to test them out on the trained models.

- On RTX 5070 Ti, takes about 15 seconds per epoch. On a M1 Mac takes about 20 mins per epoch.
