# Music's Cover generator[WIP]

## discription

this is a fuse model to make midi+lyric to a fuse embedding that can generate this music's cover
at this project, I use ...

## how to use
first use `source ./modelEnv/env388/bin/activate` into virtual envirment

use `python3 generate_image.py`

and input artist and song name

it will auto download song's lyrics and music

and generate its cover image

## performance
the similarity of fuse embedding and original cover
![fuse model similarity](generated_images/20250606_015613/mean_similarity_per_epoch.png)
the loss of decoder
![Decoder Training Loss](generated_images/20250606_015613/decoder_training_loss.png)

## reference
- [dataset](https://www.kaggle.com/datasets/deepshah16/song-lyrics-dataset)
- [bert](https://arxiv.org/abs/1810.04805)
- [diffusion model](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- [microsoft musicBert](https://microsoft.github.io/muzic/musicbert/)
- [YourMT3](https://github.com/mimbres/YourMT3) this is a model for transform wav to midi
- [LP-MusicCaps](https://github.com/seungheondoh/lp-music-caps)
- [Qiu_Image_Generation_Associated_CVPR_2018_paper](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w49/Qiu_Image_Generation_Associated_CVPR_2018_paper.pdf)
- [Performance-MIDI to Score](https://arxiv.org/abs/2410.00210v1)
- [DALI](https://github.com/gabolsgabs/DALI/tree/master)

## issue

- image output size is only 64x64
- decoder overfitting

## Thanks

- 葉梅珍 教授
- 王鈞右 教授
- 王兆偉
- 鍾詠傑
- 吳文元
