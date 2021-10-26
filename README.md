# Fairness via Representation Neutralization

PyTorch code for the Neurips 2021 paper: Fairness via Representation Neutralization. In this code, we use [MEPS dataset](https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb) as example. 


## Usage Instructions:
To run RNF (with proxy attribute annotations):

```
python train_rnf.py 
```


The hyperparameter alpha in the train_rnf.py file is used to control the fairness accuracy trade-off. For MEPS dataset, a reasonable range for the alpha value is between [0, 0.035].

## System requirement:
torch==0.4.1.post2, torchtext==0.2.3

## Reference:
```
@inproceedings{du2021fairness,
  title={Fairness via Representation Neutralization},
  author={Du, Mengnan and Mukherjee, Subhabrata and Wang, Guanchu and Tang, Ruixiang and Awadallah, Ahmed Hassan and Hu, Xia},
  booktitle={Neurips},
  year={2021}
}
```