# COMP3340 Group 10 - Attention Mechanism and Additional Data

  

## Contact

- This repository contains code for Attention Mechanism (Including ViT, TinyVit, SEResnet) and Additional Data Experiment which utilize Oxford102 to learn more general flower feature extraction and fintune and test on Oxford17.
- Noted that: For some important output pth and log, we will uploaded one zip seperately and the data will also be a seperate zip. 
- All these are done in mmpretrain( not mmclassification)
- Someting to clarify:
-- 0. I have already refine the pretrained ViT, and now it perform a lot better than what is shown in presentaion slides and it grow steadily( I warm up and then let lr go down steadly and finally linear slight adjust, it is shown in the schedule)
-- 1.We find out a lot of problem and did many experiments to solve them, such as data leakage when additional data in Oxford102 has overlap with Oxford17---We will just give the final non-data leakage data and code but not the one with this issue. 
-- 2. In the freezing pretrain-ViT part, this version make it partially frozon because it got best performance--- in the code you can change the freezing parameter to see other performance outcome(I thought not that necessary to upload 3 seperate config)
-- 3. Considering the pth file of some model will be very huge, When saving the pth for testing you could self-defined a directory. ( you could just created a output-directory next to configs and give all model a sub-directory. if do not do these at first and adjust the command use to test and train, !!!a output may be created automatically if you just run the train command shown)


- For any question and enquiry, please feel free to reach out to Ruilin Gao (u3577254@connect.hku.hk) 
- I have tried to only keep the most important to make the repo looks clear.

- Thanks for your effort and enjoy!!!

  

## Overview

****Prerequisite for Reproduction****

1. [Set up conda environment](#env_setup)

2. [Download data and pretrain/additional_data PTH file folder and put them under the correct folder](#downloads)

3. [Run the commands to reproduce important results](#cmd_repro)

  

****Software, Hardware & System Requirements****

- Software
Set up environment as [following](#env_setup)

- Hardware
Experiments are conducted on one NVIDIA GeForce RTX 2080 Ti

- System
Linux

  

****Note****
The ViT models may take 20 minutes for training and others takes 5-30 minutes.


## Environment setup <a id="env_setup"/>

  

### MMPretrain-latest-version environment setup (May Also required by some other Group 10 repos)

  

****Step 1. Create virtual environment using anaconda****

```

conda create -n mmpretrain python=3.8 -y
conda activate mmpretrain

```

  

*_Please make sure that you are create a virtual env with python version 3.8_*

  

****Step 2 Install Pytorch from wheel****

  

```

wget https://download.pytorch.org/whl/cu110/torch-1.7.1%2Bcu110-cp38-cp38-linux_x86_64.whl#sha256=709cec07bb34735bcf49ad1d631e4d90d29fa56fe23ac9768089c854367a1ac9

pip install torch-1.7.1+cu110-cp38-cp38-linux_x86_64.whl

```

  

*_Please double check that you install the correct version of pytorch

  


  

****Step 3 Install cudatoolkit via conda-forge channel****

  

*_You must be on the GPU compute node to install cudatoolkit and mmcv since GCC compiler and CUDA drivers only available on GPU computing nodes_*

  

```

gpu-interactive

conda activate mmpretrain

conda install cudatoolkit=11.1 -c pytorch -c conda-forge -y

```

  

****Step 4 Install torchvision, openmim package using pip & use openmim to install mmpretrain from source****

  

*_Make sure you are on GPU compute node!!_*

  

- `gpu-interactive`

  

*_Make sure you did not previously installed any relevant package_(These are some packages my groupmates and other classmates may use but I am on mmpretrain)*

*_PS. THIS IS just A CHECKING: Following pip show command show output a message like "no such package found"_*

  

```

pip show torchvision

pip show mmcv

pip show mmcv-full

pip show mmcls

```

  

*_remove pip cache_*

  

```

pip cache remove torchvision

pip cache remove mmcv

pip cache remove mmcv-full

pip cache remove mmcls

```

  

*HERE IS THE INSTALL:_install packages_ !!!*

  

```
conda activate mmpretrain  
pip install torchvision==0.11.2  
cd ~/mmpretrain/  
pip install -U openmim && mim install -e .
```

  

## Download data & checkpoints<a id="downloads"/>

Noted: Please 
  
For data and pretrain_pth_file:
[[Attention Mechanism and Additional Data](https://connecthkuhk.sharepoint.com/:f:/s/COMP3340Group11Project2023-24Sem2/Ek3JjIDiz1hJht60M_uRFjMBCiypABm86Ln59WF2rFR4dQ?e=wIHA99)](https://hku.hk)
For log and pth output for major experiment (it is big becasue of vit pth is 500MB each )for a reference:( the vit pth is the old version and tinyvitnoleakage pth is what used to test）they are just proof that I train and adjust models. Actually I have deleted some so a chaos is made(I train some model tens or handreds of time leads to this)
[[output_Attention_Mechanism_and_Additional_Data](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/zhongzy_connect_hku_hk/Ejn1EBU6wEBBms9aTV3eISoBKuU6n65irZnalxaUoCD6rA?e=PzcpLd)](https://hku.hk)
！！！put the pretrain_pth_file and data and output(optional) directly to the folder of code( which is the original mmpretrain folder).
For pretrain model and additional data prelearning on Oxford then train on Oxford17, you need to modify config for the pth path by copy the exact path of where you put these pth, as above, recommended to directly put in the code folder (parallel to configs, data...).

  

## Commands to reproduce results<a id="cmd_repro"/>

  if any command do not work in case. please use the "absolute path" of model configs or pretrain pth.
 !!! In ViT, Additional data and TinyVit, We need to change the path of pth file to the absolute path.!!! then do the training
### Train model command (and testing)
Place: be in the code folder given and activate the mmpretrain env already
Data and pth: drag data folder and pth folder to the coder folder submitted.
Reproduce Section ViT

#### Attention Mechanism: ViT TinyVit SEResnet 

ViT: 

(starting from ViT without pretrain adding warmup and all kinds of augmentation still not perform good, Even worse when I add more mechnism block, so we focus on pretrain-ViT)

For the ViT part, we explore diffierent forzon stage for pretrain ViT, and found that patially forzon is the best. Explantion: it keep knowledge for pretraining and at the same time keep some of the model for learning new tasks. Fully forzon limit the finetune study and zero freezing lead to overfit possibility and need more data to train.

Here is how we can train the partial freezing model( could adjust the freezing stage to get others which I tought not to be that important)
 need to change pth in configs to actually pth absolute path.
Noted: I adjust it furthermore and now can reach 98.5-99.5% partial frozon acc.

```
python tools/train.py configs/vision_transformer/vit_flowers_pretrain.py --work-dir output/vit
```
you can check log file for the training graph provided in presentation.
However, I have refine it by: longer warm-up, smaller lr init and other ways so it can achieve a 97.5% acc. PS. now it can achieve 99% training acc.

  Now we can test by: ( if you use above code to store the pth file) 
```
python tools/test.py configs/vision_transformer/vit_flowers_pretrain.py output/vit/epoch_100.pth
```

TintVit: Then we see a more lighted-weight model: TinyVit, it is really fast, solve overfit of vit, good in this flower dataset and perform a lot better than resnet
 Run this to train(finetune) the pretrained model:
 need to change pth in configs to actually pth absolute path.
 It can get to 99% very quickly during training and even reach 100%
```
python tools/train.py configs/tinyvit/tinyvit_flowers_pretrain.py --work-dir output/tinyvit
```
and test by:
```
python tools/test.py configs/tinyvit/tinyvit_flowers_pretrain.py output/tinyvit/epoch_100.pth
```
It indeed perform very good and in just serval epochs it can achieve a 99% or more accuracy and even got 100% after epochs validation in training process. We can get a around 98-99% acc by testing final trained model. ( graph need to be made so I train 100 epochs but indeed maybe we do not need that much lol)

Additional: If you need to also run the not pretrained tiny ViT we mention in Midterm Report, please train by:

```
python tools/train.py configs/tinyvit/tinyvit_flowers.py --work-dir output/tinyvit_nopretrain
```

and test by:
```
python tools/test.py configs/tinyvit/tinyvit_flowers.py output/tinyvit_nopretrain/epoch_100.pth
```
It is better than renet and ViT.( but sometimes, very very small possibility becasue I did not add warmup to it, it will crash, really very some possibilty but need to mention)

SEResnet50: Then lets come to Squeeze and Excitation, SEResnet50, ( For Resnet50 baseline I did not uploaded if needed in case just contact me)
It does not give so much increase. Possible reaseon is SE's global pooling is not concreate but flowers visually some look similar we need focus on "details", there is possibility that global pooling and given channels less weight will not focus on these details. Also may becasue resnet50 is already a big model for this small dataset which can learn all channel importance in its parameter... 

So this is how we train this SEResnet50:
(because maybe se is not that suitable for our flower dataset as explained in presentation it can just achieve 82% acc not high)
```
python tools/train.py configs/seresnet/seresnet50_flowers.py --work-dir output/seresnet
```
test by:(I got 85.8824% last time testing before uploading)
```
python tools/test.py configs/seresnet/seresnet50_flowers.py output/seresnet/epoch_200.pth
```


#### Additional  Data

Background:
We first learn on Oxford102 for 100 epochs and adjust it on Oxford17 dataset.
First we meet a problem that 102 learning split may have overlap with our final testing Oxford17 dataset so it is not a fair game for "DATA LEAKAGE".

So we use "photo hash" technique presented in our final presentation to solve this. After getting a no leakage version of Oxford102 by deduction of dupicates in test Oxford17 we redo the experiments and find there is still great progress in accuracy.

Check the data and there is a processed no_leakage Oxford102.

About the experiments and how to test it:
*The code and data are the "NO LEAKAGE" version, I did not upload the original
Oxford102 resouces and the leakage pth.

The "oxford102_split_noleak" under data is the refined 102 data split. The 
The "102_epoch_100.pth" is from pretraining 100 epochs on Oxford102 No leakage( prelearn if can not called pretrain)

We can directly loaded this  "102_epoch_100.pth"  under pretrain_pth_file to train Oxford17 Classification Resnet18(or re-train the resnet18 on Oxford102_noleak dataset mentioned above and then use the new generated pth to load as start of traing.)

It has a very fast converage speed and keep at 90-91% acc. I have adjust serval learning rate but find that this learning rate the same as the baseline works best.
( can let it stop at 100 epochs)
Train the resnet18 Oxford18 which has pre-learned knowledge from Oxford102:
It can get 91-92% training acc.
```
python tools/train.py configs/resnet/resnet18_flowers_pretrain102.py --work-dir output/additionaldata
```
And test by:( can also load by absolute path)
It can get more than 90% acc.( my average is 91.5%)
```
python tools/test.py configs/resnet/resnet18_flowers_pretrain102.py output/additionaldata/epoch_100.pth
```

  Optional：
  Train on Oxford102 for 100 epochs and get the pth file:
```
python tools/train.py configs/resnet/resnet18_flowers102.py --work-dir output/resnet102noleak
```
And load pth in "output/resnet102noleak" to "resnet18_flowers_pretrain102.py" in configs by changing the load path to the absolute path of newly generated pth(100 epochs) to reprduce the whole process's train on 102 part.

For the rest adjust on Oxford17 and testing:
redo training and testing above.

### Conlusion

we know that seperate data, pretrain pth load for big model like vit and a new mmpretrain environment did bring you a lot of trouble. Thank you very much for your effort!
if anything is not clear, do contact me through u3577254@connect.hku




  


  

