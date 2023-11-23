# Retiring-Adult

How to run the code:

- First of all install poetry
- Then run the following command:
```
poetry install
```

- To download the datasets, run the following command:
```
python download_data.py --dataset_name income
```

You can also specify other parameters:
- --group to change the sensitive value that you want to use
- --target to change the target value that you want to use
- --year to change the year of the dataset that you want to use 
- --state to specify the states that you want to use. 

For example to download the dataset for the state of California and Minnesota for the years 2014 and 2015, you can run the following command:
```
python download_data.py --dataset_name income --state CA MN --year 2014 2015
```

- To preprocess the data, run the following command:
```
python preprocess_data.py --dataset_name income
```
As with the previous command, you can specify the year and the state that you want to use.
For instance, if you want to preprocess the data for the state of California and Minnesota for the years 2014 and 2015, you can run the following command:
```
python preprocess_data.py --dataset_name income --state CA MN --year 2014 2015
```
The preprocess script will store the data in the data folder. For each state it will create a folder with the name of the state (a number) and inside it will store the data in the following files:
- dataframe.npy: the dataset in a numpy array
- groups.npy: the sensitive values in a numpy array
- targets.npy: the target values in a numpy array

If you specify multiple years, the data of the different years will be concatenated. So in the data folder you will have a folder for each state and inside it you will have the data for all the years that you specified.


At the moment you can download all the datasets (income, travel_time, mobility, public_coverage, and unemployment) but you can only 
preprocess the income dataset.