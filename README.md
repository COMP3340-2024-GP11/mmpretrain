# COMP3340 Group 10 - Attention Mechanism and Additional Data

  

## Contact

- This repository contains code for Attention Mechanism (Including ViT, TinyVit, SEResnet) and Additional Data Experiment which utilize Oxford102 to learn more general flower feature extraction and fintune and test on Oxford17.
- Noted that: For some important output pth and log, we will uploaded one zip seperately and the data will also be a seperate zip. 
- All these are done in mmpretrain( not mmclassification)
- Someting to clarify:
-- 1.We find out a lot of problem and did many experiments to solve them, such as data leakage when additional data in Oxford102 has overlap with Oxford17---We will just give the final non-data leakage data and code but not the one with this issue. 
-- 2. In the freezing pretrain-ViT part, this version make it partially frozon because it got best performance--- in the code you can change the freezing parameter to see other performance outcome(I thought not that necessary to upload 3 seperate config)
-- 3. Considering the pth file of some model will be very huge, When saving the pth for testing you could self-defined a directory. ( My reference is just created a output directory under mmpretrain and give all model a sub-directory.


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
The ViT models may take 1 hour for training and others takes 10-30 minutes.

  

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

*_PS. THIS IS A CHECKING: Following pip show command show output a message like "no such package found"_*

  

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
For log and pth output for major experiment (it is big becasue of vit pth is 500MB each )for a reference:
[[output_Attention_Mechanism_and_Additional_Data](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/zhongzy_connect_hku_hk/Ejn1EBU6wEBBms9aTV3eISoBKuU6n65irZnalxaUoCD6rA?e=PzcpLd)](https://hku.hk)
！！！put the pretrain_pth_file and data and output(optional) directly to the folder of code( which is the original mmpretrain folder).
For pretrain model and additional data prelearning on Oxford then train on Oxford17, you need to modify config for the "pth path" by copy the exact path of where you put these pth, as above, recommended to directly put in the code folder (parallel to configs, data...).

  

## Commands to reproduce results<a id="cmd_repro"/>

  

### Train model command (and testing)
Place: be in the code folder given 
Data and pth: drag data folder and pth folder to the coder folder submitted.
Reproduce Section ViT

#### Attention Mechanism: ViT TinyVit SEResnet 

ViT: 

(starting from ViT without pretrain adding warmup and all kinds of augmentation, it still not perform well, Even worse when I add more mechanism block, so we focused on pretrain-ViT)

For the ViT part, we explored different forzon stages for pretrain ViT and found that patially forzon is the best. Explanation: it keeps knowledge for pretraining and at the same time keep some of the model for learning new tasks. Fully forzon limit the finetune study and zero freezing leads to overfit possibility and needs more data to train.

Here is how we can train the partial freezing model( adjust the freezing stage to see others, which I thought not to be that important)

```

python ./tools/classification/train.py \

./configs/classification/maml/flower/maml_conv4_1xb105_flower_5way-1shot.py \

--work-dir ./output/maml_conv4_1xb105_flower_5way-1shot_meta-test

```
you can check log file for the training graph provided in presentation.
However, I have refined it by: a longer warm-up, smaller lr init, and other ways, so now it can achieve a 98.5% acc(some time need early stop because the last epoch may not get this high)

  Now we can test by: ( if you use the above code to store the pth file) 

```

python ./tools/classification/train.py \

./configs/classification/maml/flower/maml_conv4_1xb105_flower_5way-1shot.py \

--work-dir ./output/maml_conv4_1xb105_flower_5way-1shot_meta-test

```

TintVit: Then we see a more lighted-weight model: TinyVit, it is really fast, solve overfit of vit, good in this flower dataset and perform a lot better than resnet
 Run this to train(finetune) the pretrained model:
```

python ./tools/classification/train.py \

./configs/classification/maml/flower/maml_conv4_1xb105_flower_5way-1shot.py \

--work-dir ./output/maml_conv4_1xb105_flower_5way-1shot_meta-test
```
It indeed perform very good and in just serval epochs it can achieve a 99% or more accuracy and even got 100% after epochs validation in training process. We can get a around 98-99% acc by testing final trained model. ( graph need to be made so I train 100 epochs but indeed maybe we do not need that much lol)

If you need to also run the not pretrained tiny ViT we mention in Midterm Report, please train by:

```
python ./tools/classification/train.py \

./configs/classification/maml/flower/maml_conv4_1xb105_flower_5way-1shot.py \

--work-dir ./output/maml_conv4_1xb105_flower_5way-1shot_meta-test
```


and test by:
```
python ./tools/classification/train.py \

./configs/classification/maml/flower/maml_conv4_1xb105_flower_5way-1shot.py \

--work-dir ./output/maml_conv4_1xb105_flower_5way-1shot_meta-test
```
It is better than renet and ViT.( but sometimes, very very small possibility becasue I did not add warmup to it, it will crash, really very some possibilty but need to mention)

SEResnet50: Then lets come to Squeeze and Excitation, SEResnet50, ( For Resnet50 baseline I did not uploaded if needed in case just contact me)
It does not give so much increase. Possible reaseon is SE's global pooling is not concreate but flowers visually some look similar we need focus on "details", there is possibility that global pooling and given channels less weight will not focus on these details. Also may becasue resnet50 is already a big model for this small dataset which can learn all channel importance in its parameter... 

So this is how we train this SEResnet50:
```
python ./tools/classification/train.py \

./configs/classification/maml/flower/maml_conv4_1xb105_flower_5way-1shot.py \

--work-dir ./output/maml_conv4_1xb105_flower_5way-1shot_meta-test
```
test by:
```
python ./tools/classification/train.py \

./configs/classification/maml/flower/maml_conv4_1xb105_flower_5way-1shot.py \

--work-dir ./output/maml_conv4_1xb105_flower_5way-1shot_meta-test
```


#### Additional  Data

Background:
We first learn on Oxford102 for 100 epochs and adjust it on the Oxford17 dataset.
First, we encounter a problem: the 102 learning split may overlap with our final testing of the Oxford17 dataset, so it is not a fair game for "DATA LEAKAGE."

So, we use the "photo hash" technique presented in our final presentation to solve this. After getting a no-leakage version of Oxford102 by deduction of duplicates in test Oxford17, we redo the experiments and found there is still great progress in accuracy.

About the experiments and how to test it:
*The code and data are the "NO LEAKAGE" version. I did not upload the original
Oxford102 resources and the leakage 100.pth.

The "oxford102_split_noleak" under data is the refined 102 data split. The 
The "102_epoch_100.pth" is from pretraining 100 epochs on Oxford102 No leakage( prelearn if can not called pretrain)

We can directly load this "102_epoch_100.pth" under pretrain_pth_file to train Oxford17 Classification Resnet18(or re-train the resnet18 on Oxford102_noleak dataset mentioned above and then use the newly generated pth to load as the start of training.)

It has a very fast coverage speed and keeps at 90-91% acc. I have adjusted serval learning rates but find that this learning rate is the same as the baseline that works best.

Train the resnet18 Oxford18 which has pre-learned knowledge from Oxford102:

```
python ./tools/classification/train.py \

./configs/classification/maml/flower/maml_conv4_1xb105_flower_5way-1shot.py \

--work-dir ./output/maml_conv4_1xb105_flower_5way-1shot_meta-test
```
And test by:
```
python ./tools/classification/train.py \

./configs/classification/maml/flower/maml_conv4_1xb105_flower_5way-1shot.py \

--work-dir ./output/maml_conv4_1xb105_flower_5way-1shot_meta-test
```

  Optional：
  Train on Oxford102 for 100 epochs and get the pth file:
```
python ./tools/classification/train.py \

./configs/classification/maml/flower/maml_conv4_1xb105_flower_5way-1shot.py \

--work-dir ./output/maml_conv4_1xb105_flower_5way-1shot_meta-test
```
And load it to our Resnet18 for the "oxford17 pretrain on Oxford102" model.

Then, redo the training and testing above.

### Conclusion

we know that separate data, pretrain pth load for big models like vit, and a new mmpretrain environment did bring you a lot of trouble. Thank you very much for your effort!
if anything is not clear, do contact me at u3577254@connect.hku

  



  


  

