<h1 align="center">SkyReels-A2: Compose Anything in Video Diffusion Transformers</h1> 

<div align='center'>
    <a href='https://scholar.google.com/citations?user=_43YnBcAAAAJ&hl=zh-CN' target='_blank'>Zhengcong Fei<sup> * </sup></a>&emsp;
    <a href='https://debangli.github.io/' target='_blank'>Debang Li<sup> * &dagger;</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=6D_nzucAAAAJ&hl=en' target='_blank'>Di Qiu<sup> * &dagger;</sup></a>&emsp;
    <a href='' target='_blank'>Jiahua Wang</a>&emsp;
    <a href='' target='_blank'>Yikun Dou</a>&emsp;
    <a href='' target='_blank'>Rui Wang</a>&emsp;
</div>

<div align='center'>
    <a href='https://scholar.google.com/citations?user=bL_o7I8AAAAJ&hl=en' target='_blank'>Jingtao Xu</a>&emsp;
    <a href='' target='_blank'>Mingyuan Fan</a>&emsp;
    <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=HukWSw4AAAAJ' target='_blank'>Guibin Chen</a>&emsp;
    <a href='' target='_blank'>Yang Li</a>&emsp;
    <a href='' target='_blank'>Yahui Zhou</a>&emsp;
</div>
<div align='center'>
    <br><sup> * </sup>Equal contribution,<sup> &dagger; </sup>Project lead
    <br><strong>Skywork AI, Kunlun Inc.</strong>
</div>


<div align="center">
  <!-- <a href='LICENSE'><img src='https://img.shields.io/badge/license-MIT-yellow'></a> -->
  <a href='https://arxiv.org'><img src='https://img.shields.io/badge/arXiv-SkyReels A2-red'></a>
  <a href='https://skyworkai.github.io/skyreels-a2.github.io/'><img src='https://img.shields.io/badge/Project-SkyReels A2-green'></a>
  <a href='https://huggingface.co/Skywork/SkyReels-A2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
  <a href='https://huggingface.co/datasets/Skywork/A2-Bench'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-A2Bench-blue'></a>
  <a href='https://www.skyreels.ai'><img src='https://img.shields.io/badge/Demo-Spaces-yellow'></a>
  <br>
</div>
<br>

<p align="center">
  <img src="./assets/demo.gif" alt="showcase" width="800">
  <br>
  🔥 For more results, visit our <a href="https://skyworkai.github.io/skyreels-a2.github.io/"><strong>homepage</strong></a> 🔥
</p>

<p align="center">
    👋 Join our <a href="https://discord.gg/PwM6NYtccQ" target="_blank"><strong>Discord</strong></a> 
</p>

This repo, named **SkyReels-A2**, contains the official PyTorch implementation of our paper [SkyReels-A2: Compose Anything in Video Diffusion Transformers](https://arxiv.org).

## 🎉 News

- [x] Apr 3, 2025: 🔥 Release pre-view version of checkpoints, code of model inference, preprocessing, and gradio demos.
- [ ] Release all model sequence, including infinity-long version.
- [ ] Support A2-Bench evaluation and leaderboard.

## 🪄 Models
| Models                   | Download Link                                                                                                                                                                           | Video Size         |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| A2-Wan2.1-14B-Preview | [Huggingface](https://huggingface.co/Skywork/SkyReels-A2) 🤗                                                                                                                                                              | ~ 81 x 480 x 832    | 
| A2-Wan2.1-14B         | [To be released](https://github.com/SkyworkAI/SkyReels-A2)  | ~ 81 x 480 x 832    | 
| A2-Wan2.1-14B-Infinity         | [To be released](https://github.com/SkyworkAI/SkyReels-A2)  | ~ Inf x 720 x 1080   | 




## 1. Getting Started 🏁 

### 1.1 Clone the code and prepare the environment 🛠️

First git clone the repository with code: 
```bash
git clone https://github.com/SkyworkAI/SkyReels-A2.git
cd SkyReels-A2

# create env using conda
conda create -n skyreels-a2 python=3.10
conda activate skyreels-a2
```
Then, install the remaining dependencies:
```bash
pip install -r requirements.txt
```

### 1.2 Download pretrained weights 📥

You can download the pretrained weights from HuggingFace as:
```bash
# !pip install -U "huggingface_hub[cli]"
huggingface-cli download Skywork/SkyReels-A2 --local-dir local_path --exclude "*.git*" "README.md" "docs"
```
or download from webpage mannually. 


### 1.3 Inference 🚀

You can first set the model path and reference images path and then simply run the inference scripts as: 
```bash
python infer.py
```

If the script runs successfully, you will get an output mp4 file. This file includes the following results: driving video, input image or video, and generated result.


We also support multi-GPU inference scripts for faster inference, as:
```bash
python infer_MGPU.py
```


#### Gradio Interface 🤗

We also provide a [Gradio](https://huggingface.co/docs/hub/spaces-sdks-gradio) interface for a better user experience, just run by:

```bash
python app.py
```

The graphical interactive interface is shown as below.  



## 2. A2-Bench Evaluation 👓

Coming soon.



## Acknowledgements 💐

We would like to thank the contributors of [Wan](https://github.com/Wan-Video/Wan2.1) and [finetrainers](https://github.com/a-r-r-o-w/finetrainers) repositories, for their open research and contributions. 

## Citation 💖
If you find SkyReels-A2 useful for your research, welcome to 🌟 this repo and cite our work using the following BibTeX:
```bibtex
@article{fei2025skyreelsa2,
  title={SkyReels-A2: Compose Anything in Video Diffusion Transformers},
  author={skyreels team},
  journal={arXiv},
  year={2025}
}
```



