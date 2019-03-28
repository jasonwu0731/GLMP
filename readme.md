# GLMP
**Global-to-local Memory Pointer Networks for Task-Oriented Dialogue** (ACL 2018). Chien-Sheng Wu, Richard Socher, Caiming Xiong. Submitted to ***ICLR 2019***. [[PDF]](...). 

This code has been written using PyTorch 0.4.

If you use any source codes or datasets included in this toolkit in your work, please cite the following paper. The bibtex are listed below:
<pre>
</pre>


<p align="center">
<img src="img/new_block.png" width="100%" />
</p>

<p align="center">
<img src="img/new_enc.png" width="50%" />
<img src="img/new_dec.png" width="50%" />
</p>

## Import data
Under the utils folder, we have the script to import and batch the data for each dataset. 

## Train a model for task-oriented dialog datasets
We created `myTrain.py` to train models. You can run:
GLMP bAbI t1-5:
```console
❱❱❱ python3 myTrain.py -lr=0.001 -l=1 -hdd=128 -dr=0.2 -dec=GLMP -bsz=8 -ds=babi -t=1 
```
or GLMP SMD 
```console
❱❱❱ python3 myTrain.py -lr=0.001 -l=1 -hdd=128 -dr=0.2 -dec=GLMP -bsz=8 -ds=kvr -t=
```

While training, the model with the best validation is saved. If you want to reuse a model add `-path=path_name_model` to the function call. The model is evaluated by using per responce accuracy, WER, F1 and BLEU.

## Test a model for task-oriented dialog datasets
We created  `myTest.py` to train models. You can run:
GLMP bAbI t1-5:
```console
❱❱❱ python myTest.py -ds=babi -path=<path_to_saved_model> 
```
or GLMP SMD 
```console
❱❱❱ python myTest.py -ds=kvr -path=<path_to_saved_model> -rec=1
```

## Visualization Memory Access
<p align="center">
<img src="img/VIZ.pdf" width="75%" />
</p>
