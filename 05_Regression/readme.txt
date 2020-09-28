#########################################
########### Sample codes ################
----------------------------------------
########## housing.txt dataset ##########
### -> lr regressor
python main.py lr housing.txt

### -> ransac regressor
python main.py ransac housing.txt

### -> ridge regressor
python main.py ridge housing.txt --alpha1 1 --solver auto
python main.py ridge housing.txt --alpha1 0.5 --solver auto

python main.py ridge housing.txt --alpha1 1 --solver auto
python main.py ridge housing.txt --alpha1 1 --solver svd

### -> lasso regressor
python main.py lasso housing.txt --alpha2 1
python main.py lasso housing.txt --alpha2 0.5

### -> rf regressor
python main.py rf housing.txt --n_estimators 1000 --criterion mse --n_jobs 10
python main.py rf housing.txt --n_estimators 100 --criterion mse --n_jobs 10

python main.py rf housing.txt --n_estimators 100 --criterion mse --n_jobs 10
python main.py rf housing.txt --n_estimators 100 --criterion mae --n_jobs 10

python main.py rf housing.txt --n_estimators 100 --criterion mse --n_jobs 10
python main.py rf housing.txt --n_estimators 100 --criterion mse --n_jobs 100

### -> normal regressor
python main.py normal housing.txt

### -> logistic regressor
python main.py logistic housing.txt

-------------------------------------
########## CRP.csv dataset ##########
### -> lr regressor
python main.py lr CRP_new.csv

### -> ransac regressor
python main.py ransac CRP_new.csv

### -> ridge regressor
python main.py ridge CRP_new.csv --alpha1 1 --solver auto
python main.py ridge CRP_new.csv --alpha1 0.5 --solver auto

python main.py ridge CRP_new.csv --alpha1 1 --solver auto
python main.py ridge CRP_new.csv --alpha1 1 --solver svd

### -> lasso regressor
python main.py lasso CRP_new.csv --alpha2 1
python main.py lasso CRP_new.csv --alpha2 0.5

### -> rf regressor
python main.py rf CRP_new.csv --n_estimators 1000 --criterion mse --n_jobs 10
python main.py rf CRP_new.csv --n_estimators 100 --criterion mse --n_jobs 10

python main.py rf CRP_new.csv --n_estimators 100 --criterion mse --n_jobs 10
python main.py rf CRP_new.csv --n_estimators 100 --criterion mae --n_jobs 10

python main.py rf CRP_new.csv --n_estimators 100 --criterion mse --n_jobs 10
python main.py rf CRP_new.csv --n_estimators 100 --criterion mse --n_jobs 100
