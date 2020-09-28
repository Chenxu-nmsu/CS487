#########################################
# see readme.ipynb
# link: https://colab.research.google.com/drive/1MAbmRp6IS3oA1PMgPS79dk2Mq61DKfd_?usp=sharing
# a detailed procedure regarding how to run the following codes is in readme.ipynb.
step 1: unzip the hw.zip, then put all files inside your google drive ‘Colab Notebooks’ folder
step 2: Mount google files
step 3: Make sure gpu is active (Change runtime type)
step 4: Access to the right path to execute files (in accordance with the path of ‘Colab Notebooks’ folder). You might have to change the path.
step 5: Run following codes


########### Sample codes ################
### ---> Case 1 (Q1)
python main.py --filters1 4 --filters2 2 --kernel_size1 3 3 --kernel_size2 3 3 --poolingfun1 MaxPool --poolingfun2 MaxPool


### ---> Case 2
python main.py --filters1 16 --filters2 32 --kernel_size1 3 3 --kernel_size2 3 3 --poolingfun1 MaxPool --poolingfun2 MaxPool


### ---> Case 3
python main.py --filters1 4 --filters2 2 --kernel_size1 3 3 --kernel_size2 3 3 --poolingfun1 AveragePool --poolingfun2 AveragePool


### ---> Case 4
python main.py --filters1 4 --filters2 2 --kernel_size1 3 3 --kernel_size2 3 3 --poolingfun1 MaxPool --poolingfun2 MaxPool


### ---> Case 5
python main.py --filters1 4 --filters2 2 --kernel_size1 5 5 --kernel_size2 3 3 --poolingfun1 MaxPool --poolingfun2 MaxPool