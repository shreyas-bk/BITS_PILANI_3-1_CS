# NNFL_Term_Paper - CYCLEGAN

Implementation Details:
 - Model trained on subset of Van Gogh dataset (400 images of both categories)
 - Model trained for 200 epochs
 - Architecture similar to diagram given below, but with 9 residual blocks since input image size is 256x256

Generator Architecture:

![](https://github.com/shreyas-bk/NNFL_Term_Paper/blob/main/CycleGAN%20Generator.png)

Saved weights and visual results can be found at : [drive](https://drive.google.com/drive/folders/1h9ZGkrj5Hko4MyniTpyXe2nR6i04oQ4l?usp=sharing)

The contents of the assignment are as follows:
```
13
|   NNFL_Paper13.ppt               - contain the powerpoint presentation for this term paper  
|   NNFL_Assignment_CycleGAN.ipynb - contains the code that is used to create, train and test or implementation
|				     of CycleGAN. All details about the explaination of functions and files, 
|				     and instructions to execute the code is present within the tex cells of the
|				     notebook.
|___Base Code                      - contains original base code for the implementation of CycleGAN (contents of 
|         			     each file and function are given at the start of the notebook)
|___Visual Results
|   |___Cezanne2Vangogh
|   |___Monet2Vangogh
|   |___Real2Vangogh
|   |___Vangogh_examples_for_cycle_consistency
|   train_plot.png        	   - contains the plot of the traning losses when we trained CycleGAN on VanGogh 
|			             dataset for 200 epochs
|   loss_log.txt                   - contaians the numeric values of all the different losses throughout the 
|				     training process of 200 epochs 
|   train_opt.txt                  - contains the details of the configurations we used to train the model
```
Note: The notebook itself is sufficient to train the model

Visual Results:

Real

![](https://github.com/shreyas-bk/NNFL_Term_Paper/blob/main/Visual%20Results/Cezanne2Vangogh/130_R.png)

Transformed

![](https://github.com/shreyas-bk/NNFL_Term_Paper/blob/main/Visual%20Results/Cezanne2Vangogh/130_F.png)
