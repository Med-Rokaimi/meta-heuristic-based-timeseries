# Brent-oil-price-forecasting-base-deep-learning-models
implementing various deep learning networks with Pytorch and Keras using multivariate timeseries data.
Five included models and over 100 metaheuristic algorithm available:
## available models:
1. LSTM / Bi-LSTM
2. GRU/Bi-GRU
3. CNN-LSTM
4. CNN-LSTM-Attention
5. Encoder-Decoder-LSTM

##Meta-heuristic 
This code is built upon mealpy library which includes over 100 metaheuristic algorithm
```
@article{van2023mealpy,
   title={MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
   author={Van Thieu, Nguyen and Mirjalili, Seyedali},
   journal={Journal of Systems Architecture},
   year={2023},
   publisher={Elsevier},
   doi={10.1016/j.sysarc.2023.102871}
}
```
## How to use it:
1. clone the repos.
2. Create your own env and install required packages
```
pip install -r requirements.txt
```
3- run main.py 
```
python main.py [model name]
```
You can select [model name] from the available model names shown on the main.py (for instance: Bi-LSTM, CNN-LSTM-att, encoder-decoder-LSTM)
4. you can setup selected features (USD, sentiment score, Brent price, you can add as many as columns you want) from arg.py in the config folder. From argg.py you can also tunning models hyperparameters.
5. GWO is a default metaheuristic optimiser, please see the [mealpy library documentation](https://mealpy.readthedocs.io/en/latest/index.html) for more information about the available algorithms and how to use them. 
