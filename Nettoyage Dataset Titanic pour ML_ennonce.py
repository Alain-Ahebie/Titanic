{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Nettoyage pour Machine Learning : Titanic"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "#je fais mes import\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.pyplot as pltF\n",
    "import seaborn as sns\n",
    "from math import sqrt\n",
    "from numpy import concatenate\n",
    "from matplotlib import pyplot\n",
    "from pandas import read_csv\n",
    "from pandas import DataFrame\n",
    "from pandas import concat\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.metrics import mean_squared_error"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Chargement des jeux de données et préparation des train/test"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Dans cet exercice, deux jeux de données doivent être nettoyés. Un echantillon train celui qui entrainera le modele, ce dataset est donc labellisé (avec résultat). Un echantillon test (sans resultat/label) dont l objectif est de prédire les labels de chacunes de ses lignes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "train = read_csv('train.csv', header=0)\n",
    "test = read_csv('test.csv', header=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>759</th>\n",
       "      <td>760</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Rothes, the Countess. of (Lucy Noel Martha Dye...</td>\n",
       "      <td>female</td>\n",
       "      <td>33.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>110152</td>\n",
       "      <td>86.50</td>\n",
       "      <td>B77</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>214</th>\n",
       "      <td>215</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Kiernan, Mr. Philip</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>367229</td>\n",
       "      <td>7.75</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>585</th>\n",
       "      <td>586</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Taussig, Miss. Ruth</td>\n",
       "      <td>female</td>\n",
       "      <td>18.0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>110413</td>\n",
       "      <td>79.65</td>\n",
       "      <td>E68</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>137</th>\n",
       "      <td>138</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mr. Jacques Heath</td>\n",
       "      <td>male</td>\n",
       "      <td>37.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.10</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>520</th>\n",
       "      <td>521</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Perreault, Miss. Anne</td>\n",
       "      <td>female</td>\n",
       "      <td>30.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>12749</td>\n",
       "      <td>93.50</td>\n",
       "      <td>B73</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass  \\\n",
       "759          760         1       1   \n",
       "214          215         0       3   \n",
       "585          586         1       1   \n",
       "137          138         0       1   \n",
       "520          521         1       1   \n",
       "\n",
       "                                                  Name     Sex   Age  SibSp  \\\n",
       "759  Rothes, the Countess. of (Lucy Noel Martha Dye...  female  33.0      0   \n",
       "214                                Kiernan, Mr. Philip    male   NaN      1   \n",
       "585                                Taussig, Miss. Ruth  female  18.0      0   \n",
       "137                        Futrelle, Mr. Jacques Heath    male  37.0      1   \n",
       "520                              Perreault, Miss. Anne  female  30.0      0   \n",
       "\n",
       "     Parch  Ticket   Fare Cabin Embarked  \n",
       "759      0  110152  86.50   B77        S  \n",
       "214      0  367229   7.75   NaN        Q  \n",
       "585      2  110413  79.65   E68        S  \n",
       "137      0  113803  53.10  C123        S  \n",
       "520      0   12749  93.50   B73        S  "
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#regarder le train\n",
    "train.sample(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>251</th>\n",
       "      <td>1143</td>\n",
       "      <td>3</td>\n",
       "      <td>Abrahamsson, Mr. Abraham August Johannes</td>\n",
       "      <td>male</td>\n",
       "      <td>20.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>SOTON/O2 3101284</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>132</th>\n",
       "      <td>1024</td>\n",
       "      <td>3</td>\n",
       "      <td>Lefebre, Mrs. Frank (Frances)</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>4133</td>\n",
       "      <td>25.4667</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>362</th>\n",
       "      <td>1254</td>\n",
       "      <td>2</td>\n",
       "      <td>Ware, Mrs. John James (Florence Louise Long)</td>\n",
       "      <td>female</td>\n",
       "      <td>31.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>CA 31352</td>\n",
       "      <td>21.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>417</th>\n",
       "      <td>1309</td>\n",
       "      <td>3</td>\n",
       "      <td>Peter, Master. Michael J</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2668</td>\n",
       "      <td>22.3583</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>921</td>\n",
       "      <td>3</td>\n",
       "      <td>Samaan, Mr. Elias</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2662</td>\n",
       "      <td>21.6792</td>\n",
       "      <td>NaN</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Pclass                                          Name  \\\n",
       "251         1143       3      Abrahamsson, Mr. Abraham August Johannes   \n",
       "132         1024       3                 Lefebre, Mrs. Frank (Frances)   \n",
       "362         1254       2  Ware, Mrs. John James (Florence Louise Long)   \n",
       "417         1309       3                      Peter, Master. Michael J   \n",
       "29           921       3                             Samaan, Mr. Elias   \n",
       "\n",
       "        Sex   Age  SibSp  Parch            Ticket     Fare Cabin Embarked  \n",
       "251    male  20.0      0      0  SOTON/O2 3101284   7.9250   NaN        S  \n",
       "132  female   NaN      0      4              4133  25.4667   NaN        S  \n",
       "362  female  31.0      0      0          CA 31352  21.0000   NaN        S  \n",
       "417    male   NaN      1      1              2668  22.3583   NaN        C  \n",
       "29     male   NaN      2      0              2662  21.6792   NaN        C  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#regarder le test\n",
    "test.sample(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Survived est la colonne à prédire de notre jeu d'entrainement. Chaque ligne correspond à un passager, il s'agit d'une observation. La colonne Survived nous indique si ce passager a survécu ou non au nauffrage du Titanic."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Analyse du dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Analyser les datasets avec par ex .describe(include=\"all\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891</td>\n",
       "      <td>891</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>204</td>\n",
       "      <td>889</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unique</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>891</td>\n",
       "      <td>2</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>681</td>\n",
       "      <td>NaN</td>\n",
       "      <td>147</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>top</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Turja, Miss. Anna Sofia</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>CA. 2343</td>\n",
       "      <td>NaN</td>\n",
       "      <td>B96 B98</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>freq</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>577</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7</td>\n",
       "      <td>NaN</td>\n",
       "      <td>4</td>\n",
       "      <td>644</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>NaN</td>\n",
       "      <td>32.204208</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>257.353842</td>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>NaN</td>\n",
       "      <td>49.693429</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>223.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.910400</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>14.454200</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>668.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>31.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>512.329200</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        PassengerId    Survived      Pclass                     Name   Sex  \\\n",
       "count    891.000000  891.000000  891.000000                      891   891   \n",
       "unique          NaN         NaN         NaN                      891     2   \n",
       "top             NaN         NaN         NaN  Turja, Miss. Anna Sofia  male   \n",
       "freq            NaN         NaN         NaN                        1   577   \n",
       "mean     446.000000    0.383838    2.308642                      NaN   NaN   \n",
       "std      257.353842    0.486592    0.836071                      NaN   NaN   \n",
       "min        1.000000    0.000000    1.000000                      NaN   NaN   \n",
       "25%      223.500000    0.000000    2.000000                      NaN   NaN   \n",
       "50%      446.000000    0.000000    3.000000                      NaN   NaN   \n",
       "75%      668.500000    1.000000    3.000000                      NaN   NaN   \n",
       "max      891.000000    1.000000    3.000000                      NaN   NaN   \n",
       "\n",
       "               Age       SibSp       Parch    Ticket        Fare    Cabin  \\\n",
       "count   714.000000  891.000000  891.000000       891  891.000000      204   \n",
       "unique         NaN         NaN         NaN       681         NaN      147   \n",
       "top            NaN         NaN         NaN  CA. 2343         NaN  B96 B98   \n",
       "freq           NaN         NaN         NaN         7         NaN        4   \n",
       "mean     29.699118    0.523008    0.381594       NaN   32.204208      NaN   \n",
       "std      14.526497    1.102743    0.806057       NaN   49.693429      NaN   \n",
       "min       0.420000    0.000000    0.000000       NaN    0.000000      NaN   \n",
       "25%      20.125000    0.000000    0.000000       NaN    7.910400      NaN   \n",
       "50%      28.000000    0.000000    0.000000       NaN   14.454200      NaN   \n",
       "75%      38.000000    1.000000    0.000000       NaN   31.000000      NaN   \n",
       "max      80.000000    8.000000    6.000000       NaN  512.329200      NaN   \n",
       "\n",
       "       Embarked  \n",
       "count       889  \n",
       "unique        3  \n",
       "top           S  \n",
       "freq        644  \n",
       "mean        NaN  \n",
       "std         NaN  \n",
       "min         NaN  \n",
       "25%         NaN  \n",
       "50%         NaN  \n",
       "75%         NaN  \n",
       "max         NaN  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.describe(include=\"all\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Premieres remarques sur note jeu de données:\n",
    "\n",
    "- 891 passagers\n",
    "- 20% des informations sur leurs ages est manquantes \n",
    "- 75% des information cabines sont manquantes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Quel impact du sex sur la chance de survie ?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Percentage de femme ayant survécu: 74.20382165605095\n",
      "Percentage d' homme ayant survécu: 18.890814558058924\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYUAAAEKCAYAAAD9xUlFAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAFANJREFUeJzt3X+wXGd93/H3xzKOB2NIQbc1ox9IBQFRwOD6Wi5NSkwxRE47UhogkexO8NRFwxTZnRLjmkJVKkJpRScUEpGipG4oExDGtKnIqFUSMAwxP6rrYGxko+RWBnQlVK4xP0ySWlz72z92dbJer+6uLB2tfPV+zdzRPmefPfu90tH93POcPc+TqkKSJIBzxl2AJOnMYShIkhqGgiSpYShIkhqGgiSpYShIkhqGgiSpYShIkhqGgiSpce64CzhRixcvrhUrVoy7DEl6UrnzzjsfqKqJYf2edKGwYsUKpqamxl2GJD2pJPnGKP0cPpIkNQwFSVLDUJAkNVoNhSRrk+xPMp3k5gHPL09ye5IvJ7k7yc+1WY8kaX6thUKSRcB24CpgNbAxyeq+bm8Hbq2qS4ANwAfaqkeSNFybZwprgOmqOlBVR4GdwPq+PgU8vfv4GcDhFuuRJA3R5kdSlwAHe9ozwOV9fd4B/EGS64ELgCtbrEeSNESbZwoZsK1/7c+NwO9U1VLg54APJ3lcTUk2JZlKMjU7O9tCqZIkaPdMYQZY1tNeyuOHh64D1gJU1ReSnA8sBr7d26mqdgA7ACYnJ11UWlrgbrrpJo4cOcJFF13Etm3bxl3OWaXNM4W9wKokK5OcR+dC8q6+Pt8EXgmQ5CeA8wFPBaSz3JEjRzh06BBHjhwZdylnndZCoarmgM3AHuA+Op8y2pdka5J13W6/ArwhyVeAjwLXVpVnApI0Jq3OfVRVu4Hdfdu29Dy+F/ipNmuQJI3OO5olSQ1DQZLUMBQkSQ1DQZLUMBQkSQ1DQZLUMBQkSQ1DQZLUMBQkSQ1DQZLUaHWaC0kn5ptbXzzuEs4Icw8+EziXuQe/4d8JsHzLPaftvTxTkCQ1DAVJUsNQkCQ1DAVJUsNQkCQ1DAVJUqPVUEiyNsn+JNNJbh7w/HuT3NX9+tMk32uzHknS/Fq7TyHJImA78CpgBtibZFd3CU4Aquqf9/S/HrikrXokScO1eaawBpiuqgNVdRTYCayfp/9G4KMt1iNJGqLNUFgCHOxpz3S3PU6S5wArgU+3WI8kaYg2QyEDttVx+m4AbquqRwbuKNmUZCrJ1Ozs7CkrUJL0WG2GwgywrKe9FDh8nL4bmGfoqKp2VNVkVU1OTEycwhIlSb3anBBvL7AqyUrgEJ0f/Ff3d0ryAuCvAV9osRZJTyKLz38UmOv+qdOptVCoqrkkm4E9wCLglqral2QrMFVVu7pdNwI7q+p4Q0uSzjI3Xuyn08el1amzq2o3sLtv25a+9jvarEGSNDrvaJYkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNVoNhSRrk+xPMp3k5uP0+cUk9ybZl+QjbdYjSZpfa8txJlkEbAdeBcwAe5Psqqp7e/qsAt4K/FRVfTfJX2+rHknScG2eKawBpqvqQFUdBXYC6/v6vAHYXlXfBaiqb7dYjyRpiDZDYQlwsKc9093W6/nA85PckeSLSda2WI8kaYjWho+ADNhWA95/FXAFsBT4XJIXVdX3HrOjZBOwCWD58uWnvlJJEtDumcIMsKynvRQ4PKDP/6iqH1XV/cB+OiHxGFW1o6omq2pyYmKitYIl6WzXZijsBVYlWZnkPGADsKuvz+8BrwBIspjOcNKBFmuSJM2jtVCoqjlgM7AHuA+4tar2JdmaZF232x7gO0nuBW4H3lJV32mrJknS/Nq8pkBV7QZ2923b0vO4gDd3vyRJY+YdzZKkhqEgSWoYCpKkhqEgSWoYCpKkhqEgSWoYCpKkhqEgSWoYCpKkhqEgSWoYCpKkhqEgSWoYCpKkhqEgSWoYCpKkhqEgSWoYCpKkhqEgSWq0GgpJ1ibZn2Q6yc0Dnr82yWySu7pf/6TNeiRJ82ttjeYki4DtwKuAGWBvkl1VdW9f149V1ea26pAkja7NM4U1wHRVHaiqo8BOYH2L7ydJOklthsIS4GBPe6a7rd9rktyd5LYkywbtKMmmJFNJpmZnZ9uoVZJEu6GQAduqr/1JYEVVXQz8EfChQTuqqh1VNVlVkxMTE6e4TEnSMW2GwgzQ+5v/UuBwb4eq+k5VPdxt/hZwaYv1SJKGmPdCc5KHePxv942qevo8L98LrEqyEjgEbACu7tv/s6vqW93mOuC+UYqWJLVj3lCoqgsBkmwFjgAfpjMsdA1w4ZDXziXZDOwBFgG3VNW+7r6mqmoXcEOSdcAc8CBw7cl9O5KkkzHqR1J/tqou72n/ZpIvAdvme1FV7QZ2923b0vP4rcBbR6xBktSyUa8pPJLkmiSLkpyT5BrgkTYLkySdfqOGwtXALwL/t/v1OvquD0iSnvxGGj6qqq/jjWeStOCNdKaQ5PlJPpXkq932xUne3m5pkqTTbdTho9+ic0H4RwBVdTedj5hKkhaQUUPhqVX1v/u2zZ3qYiRJ4zVqKDyQ5Ll0b2RL8lrgW/O/RJL0ZDPqfQpvAnYAL0xyCLifzg1skqQFZNRQ+EZVXZnkAuCcqnqozaIkSeMx6vDR/Ul2AH8b+GGL9UiSxmjUUHgBnamt30QnIH4jyU+3V5YkaRxGCoWq+suqurWqfgG4BHg68NlWK5MknXYjr6eQ5GeSfAD4E+B8OtNeSJIWkJEuNCe5H7gLuBV4S1X9eatVSZLGYtRPH72kqn7QaiWSpLEbtvLaTVW1DXhXksetwFZVN7RWmSTptBt2pnBsecyptguRJI3fsOU4P9l9eHdVfflEd55kLfA+Ostx/nZV/bvj9Hst8HHgsqoygCRpTEb99NGvJflakncm+clRXpBkEbAduApYDWxMsnpAvwuBG4AvjViLJKklo96n8ArgCmAW2JHknhHWU1gDTFfVgao6Cuxk8EI976Sz1vP/G7lqSVIrRr5PoaqOVNX7gTfS+XjqliEvWQIc7GnPdLc1klwCLKuq359vR0k2JZlKMjU7OztqyZKkEzTqyms/keQd3ZXXfgP4PLB02MsGbGs+wZTkHOC9wK8Me/+q2lFVk1U1OTExMUrJkqQnYNT7FP4L8FHg1VV1eMTXzADLetpLgd7XXgi8CPhMEoCLgF1J1nmxWZLGY2godC8Y/5+qet8J7nsvsCrJSuAQneU7rz72ZFV9H1jc8z6fAW40ECRpfIYOH1XVI8Czkpx3IjuuqjlgM7CHzv0Ot1bVviRbk6x7QtVKklo18iI7wB1JdgHNvEdV9WvzvaiqdgO7+7YNvEBdVVeMWIskqSWjhsLh7tc5dK4FSJIWoJFCoar+TduFSJLGb9Sps2+n5+Okx1TV3zvlFUmSxmbU4aMbex6fD7wGmDv15UiSxmnU4aM7+zbdkcTlOCVpgRl1+OiZPc1zgEk6N5tJkhaQUYeP7uSvrinMAV8HrmujIEnS+Axbee0y4GBVrey2X0/nesLXgXtbr06SdFoNu6P5g8BRgCQvB94NfAj4PrCj3dIkSafbsOGjRVX1YPfxLwE7quoTwCeS3NVuaZKk023YmcKiJMeC45XAp3ueG/V6hCTpSWLYD/aPAp9N8gDwl8DnAJI8j84QkiRpAZk3FKrqXUk+BTwb+IOqOvYJpHOA69suTpJ0eg0dAqqqLw7Y9qftlCNJGqeR12iWJC18hoIkqWEoSJIarYZCkrVJ9ieZTnLzgOffmOSeJHcl+eMkq9usR5I0v9ZCIckiYDtwFbAa2Djgh/5HqurFVfVSYBsw7/KekqR2tXmmsAaYrqoDVXUU2Ams7+1QVT/oaV7AgIV8JEmnT5t3JS8BDva0Z4DL+zsleRPwZuA8YOBKbkk2AZsAli9ffsoLlSR1tHmmkAHbBi3pub2qngv8C+Dtg3ZUVTuqarKqJicmJk5xmZKkY9oMhRlgWU97KXB4nv47gZ9vsR5J0hBthsJeYFWSlUnOAzYAu3o7JFnV0/z7wJ+1WI8kaYjWrilU1VySzcAeYBFwS1XtS7IVmKqqXcDmJFcCPwK+C7y+rXokScO1Ov11Ve0Gdvdt29Lz+J+1+f6SpBPjHc2SpIahIElqGAqSpIahIElqGAqSpIahIElqGAqSpIahIElqGAqSpIahIElqGAqSpIahIElqGAqSpIahIElqtDp1ts5sN910E0eOHOGiiy5i27Zt4y5H0hnAUDiLHTlyhEOHDo27DElnEIePJEmNVkMhydok+5NMJ7l5wPNvTnJvkruTfCrJc9qsR5I0v9ZCIckiYDtwFbAa2JhkdV+3LwOTVXUxcBvgwLYkjVGbZwprgOmqOlBVR4GdwPreDlV1e1X9Rbf5RWBpi/VIkoZoMxSWAAd72jPdbcdzHfA/W6xHkjREm58+yoBtNbBj8o+ASeBnjvP8JmATwPLly09VfZKkPm2eKcwAy3raS4HD/Z2SXAm8DVhXVQ8P2lFV7aiqyaqanJiYaKVYSVK7Zwp7gVVJVgKHgA3A1b0dklwCfBBYW1XfbrGWx7j0Lf/1dL3VGe3CBx5iEfDNBx7y7wS48z2/PO4SpLFr7UyhquaAzcAe4D7g1qral2RrknXdbu8BngZ8PMldSXa1VY8kabhW72iuqt3A7r5tW3oeX9nm+0uSTox3NEuSGoaCJKlhKEiSGoaCJKlhKEiSGoaCJKlhKEiSGq68dhZ79LwLHvOnJBkKZ7E/X/XqcZcg6Qzj8JEkqWEoSJIahoIkqWEoSJIahoIkqWEoSJIahoIkqWEoSJIarYZCkrVJ9ieZTnLzgOdfnuRPkswleW2btUiShmstFJIsArYDVwGrgY1JVvd1+yZwLfCRtuqQJI2uzWku1gDTVXUAIMlOYD1w77EOVfX17nOPtliHJGlEbQ4fLQEO9rRnutskSWeoNkMhA7bVE9pRsinJVJKp2dnZkyxLknQ8bYbCDLCsp70UOPxEdlRVO6pqsqomJyYmTklxkqTHazMU9gKrkqxMch6wAdjV4vtJkk5Sa6FQVXPAZmAPcB9wa1XtS7I1yTqAJJclmQFeB3wwyb626pEkDdfqIjtVtRvY3bdtS8/jvXSGlSRJZwDvaJYkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNVoNhSRrk+xPMp3k5gHP/1iSj3Wf/1KSFW3WI0maX2uhkGQRsB24ClgNbEyyuq/bdcB3q+p5wHuBf99WPZKk4do8U1gDTFfVgao6CuwE1vf1WQ98qPv4NuCVSdJiTZKkebQZCkuAgz3tme62gX2qag74PvCsFmuSJM3j3Bb3Peg3/noCfUiyCdjUbf4wyf6TrE1/ZTHwwLiLOBPkP7x+3CXosTw2j/nXp2QA5TmjdGozFGaAZT3tpcDh4/SZSXIu8Azgwf4dVdUOYEdLdZ7VkkxV1eS465D6eWyOR5vDR3uBVUlWJjkP2ADs6uuzCzj269lrgU9X1ePOFCRJp0drZwpVNZdkM7AHWATcUlX7kmwFpqpqF/CfgQ8nmaZzhrChrXokScPFX8zPbkk2dYfnpDOKx+Z4GAqSpIbTXEiSGoaCGkmuSPL7465DC0OSG5Lcl+R3W9r/O5Lc2Ma+z2ZtfiRV0tntnwJXVdX94y5Eo/NMYYFJsiLJ15L8dpKvJvndJFcmuSPJnyVZ0/36fJIvd/98wYD9XJDkliR7u/36pyiRjivJfwL+JrArydsGHUtJrk3ye0k+meT+JJuTvLnb54tJntnt94bua7+S5BNJnjrg/Z6b5H8luTPJ55K88PR+xwuHobAwPQ94H3Ax8ELgauCngRuBfwl8DXh5VV0CbAH+7YB9vI3OfSOXAa8A3pPkgtNQuxaAqnojnZtVXwFcwPGPpRfROT7XAO8C/qJ7XH4B+OVun/9WVZdV1UuA++hMpNlvB3B9VV1K5zj/QDvf2cLn8NHCdH9V3QOQZB/wqaqqJPcAK+jcOf6hJKvoTCvylAH7eDWwrmfM9nxgOZ3/lNKJON6xBHB7VT0EPJTk+8Anu9vvofNLDcCLkvwq8OPA0+jc+9RI8jTg7wAf75lP88fa+EbOBobCwvRwz+NHe9qP0vk3fyed/4z/sLuGxWcG7CPAa6rKeaZ0sgYeS0kuZ/ixCvA7wM9X1VeSXAtc0bf/c4DvVdVLT23ZZyeHj85OzwAOdR9fe5w+e4Drj01lnuSS01CXFqaTPZYuBL6V5CnANf1PVtUPgPuTvK67/yR5yUnWfNYyFM5O24B3J7mDzhQkg7yTzrDS3Um+2m1LT8TJHkv/CvgS8Id0rocNcg1wXZKvAPt4/NotGpF3NEuSGp4pSJIahoIkqWEoSJIahoIkqWEoSJIahoJ0Arrz+OxLcneSu7o3YEkLhnc0SyNK8jLgHwB/q6oeTrIYOG/MZUmnlGcK0uieDTxQVQ8DVNUDVXU4yaVJPtudoXNPkmcnObc7s+cVAEneneRd4yxeGoU3r0kj6k689sfAU4E/Aj4GfB74LLC+qmaT/BLws1X1j5P8JHAbcAOdu8gvr6qj46leGo3DR9KIquqHSS4F/i6dKaA/Bvwqnemf/7A7tc8i4Fvd/vuSfJjOzJ8vMxD0ZGAoSCegqh6hM6vsZ7pTkb8J2FdVLzvOS14MfA/4G6enQunkeE1BGlGSF3TXoDjmpXTWl5joXoQmyVO6w0Yk+QXgWcDLgfcn+fHTXbN0orymII2oO3T063QWe5kDpoFNwFLg/XSmJD8X+I/Af6dzveGVVXUwyQ3ApVX1+nHULo3KUJAkNRw+kiQ1DAVJUsNQkCQ1DAVJUsNQkCQ1DAVJUsNQkCQ1DAVJUuP/A9ldS2UaP3OcAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "#Dessin d'un bar plot des survivants selon leurs sexe\n",
    "sns.barplot(x=\"Sex\", y=\"Survived\", data=train)\n",
    "\n",
    "print(\"Percentage de femme ayant survécu:\", train[\"Survived\"][train[\"Sex\"] == 'female'].value_counts(normalize = True)[1]*100)\n",
    "\n",
    "print(\"Percentage d' homme ayant survécu:\", train[\"Survived\"][train[\"Sex\"] == 'male'].value_counts(normalize = True)[1]*100)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Quel impact de la classe sur la chance de survie ?\n",
    "\n",
    "Tracer une petite dataViz avec seaborn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7f84dc3fe6a0>"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(x=\"Pclass\", y=\"Survived\", data=train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7f84dc380390>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYUAAAEKCAYAAAD9xUlFAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAFANJREFUeJzt3X+wXGd93/H3xzKOB2NIQbc1ox9IBQFRwOD6Wi5NSkwxRE47UhogkexO8NRFwxTZnRLjmkJVKkJpRScUEpGipG4oExDGtKnIqFUSMAwxP6rrYGxko+RWBnQlVK4xP0ySWlz72z92dbJer+6uLB2tfPV+zdzRPmefPfu90tH93POcPc+TqkKSJIBzxl2AJOnMYShIkhqGgiSpYShIkhqGgiSpYShIkhqGgiSpYShIkhqGgiSpce64CzhRixcvrhUrVoy7DEl6UrnzzjsfqKqJYf2edKGwYsUKpqamxl2GJD2pJPnGKP0cPpIkNQwFSVLDUJAkNVoNhSRrk+xPMp3k5gHPL09ye5IvJ7k7yc+1WY8kaX6thUKSRcB24CpgNbAxyeq+bm8Hbq2qS4ANwAfaqkeSNFybZwprgOmqOlBVR4GdwPq+PgU8vfv4GcDhFuuRJA3R5kdSlwAHe9ozwOV9fd4B/EGS64ELgCtbrEeSNESbZwoZsK1/7c+NwO9U1VLg54APJ3lcTUk2JZlKMjU7O9tCqZIkaPdMYQZY1tNeyuOHh64D1gJU1ReSnA8sBr7d26mqdgA7ACYnJ11UWlrgbrrpJo4cOcJFF13Etm3bxl3OWaXNM4W9wKokK5OcR+dC8q6+Pt8EXgmQ5CeA8wFPBaSz3JEjRzh06BBHjhwZdylnndZCoarmgM3AHuA+Op8y2pdka5J13W6/ArwhyVeAjwLXVpVnApI0Jq3OfVRVu4Hdfdu29Dy+F/ipNmuQJI3OO5olSQ1DQZLUMBQkSQ1DQZLUMBQkSQ1DQZLUMBQkSQ1DQZLUMBQkSQ1DQZLUaHWaC0kn5ptbXzzuEs4Icw8+EziXuQe/4d8JsHzLPaftvTxTkCQ1DAVJUsNQkCQ1DAVJUsNQkCQ1DAVJUqPVUEiyNsn+JNNJbh7w/HuT3NX9+tMk32uzHknS/Fq7TyHJImA78CpgBtibZFd3CU4Aquqf9/S/HrikrXokScO1eaawBpiuqgNVdRTYCayfp/9G4KMt1iNJGqLNUFgCHOxpz3S3PU6S5wArgU+3WI8kaYg2QyEDttVx+m4AbquqRwbuKNmUZCrJ1Ozs7CkrUJL0WG2GwgywrKe9FDh8nL4bmGfoqKp2VNVkVU1OTEycwhIlSb3anBBvL7AqyUrgEJ0f/Ff3d0ryAuCvAV9osRZJTyKLz38UmOv+qdOptVCoqrkkm4E9wCLglqral2QrMFVVu7pdNwI7q+p4Q0uSzjI3Xuyn08el1amzq2o3sLtv25a+9jvarEGSNDrvaJYkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNVoNhSRrk+xPMp3k5uP0+cUk9ybZl+QjbdYjSZpfa8txJlkEbAdeBcwAe5Psqqp7e/qsAt4K/FRVfTfJX2+rHknScG2eKawBpqvqQFUdBXYC6/v6vAHYXlXfBaiqb7dYjyRpiDZDYQlwsKc9093W6/nA85PckeSLSda2WI8kaYjWho+ADNhWA95/FXAFsBT4XJIXVdX3HrOjZBOwCWD58uWnvlJJEtDumcIMsKynvRQ4PKDP/6iqH1XV/cB+OiHxGFW1o6omq2pyYmKitYIl6WzXZijsBVYlWZnkPGADsKuvz+8BrwBIspjOcNKBFmuSJM2jtVCoqjlgM7AHuA+4tar2JdmaZF232x7gO0nuBW4H3lJV32mrJknS/Nq8pkBV7QZ2923b0vO4gDd3vyRJY+YdzZKkhqEgSWoYCpKkhqEgSWoYCpKkhqEgSWoYCpKkhqEgSWoYCpKkhqEgSWoYCpKkhqEgSWoYCpKkhqEgSWoYCpKkhqEgSWoYCpKkhqEgSWq0GgpJ1ibZn2Q6yc0Dnr82yWySu7pf/6TNeiRJ82ttjeYki4DtwKuAGWBvkl1VdW9f149V1ea26pAkja7NM4U1wHRVHaiqo8BOYH2L7ydJOklthsIS4GBPe6a7rd9rktyd5LYkywbtKMmmJFNJpmZnZ9uoVZJEu6GQAduqr/1JYEVVXQz8EfChQTuqqh1VNVlVkxMTE6e4TEnSMW2GwgzQ+5v/UuBwb4eq+k5VPdxt/hZwaYv1SJKGmPdCc5KHePxv942qevo8L98LrEqyEjgEbACu7tv/s6vqW93mOuC+UYqWJLVj3lCoqgsBkmwFjgAfpjMsdA1w4ZDXziXZDOwBFgG3VNW+7r6mqmoXcEOSdcAc8CBw7cl9O5KkkzHqR1J/tqou72n/ZpIvAdvme1FV7QZ2923b0vP4rcBbR6xBktSyUa8pPJLkmiSLkpyT5BrgkTYLkySdfqOGwtXALwL/t/v1OvquD0iSnvxGGj6qqq/jjWeStOCNdKaQ5PlJPpXkq932xUne3m5pkqTTbdTho9+ic0H4RwBVdTedj5hKkhaQUUPhqVX1v/u2zZ3qYiRJ4zVqKDyQ5Ll0b2RL8lrgW/O/RJL0ZDPqfQpvAnYAL0xyCLifzg1skqQFZNRQ+EZVXZnkAuCcqnqozaIkSeMx6vDR/Ul2AH8b+GGL9UiSxmjUUHgBnamt30QnIH4jyU+3V5YkaRxGCoWq+suqurWqfgG4BHg68NlWK5MknXYjr6eQ5GeSfAD4E+B8OtNeSJIWkJEuNCe5H7gLuBV4S1X9eatVSZLGYtRPH72kqn7QaiWSpLEbtvLaTVW1DXhXksetwFZVN7RWmSTptBt2pnBsecyptguRJI3fsOU4P9l9eHdVfflEd55kLfA+Ostx/nZV/bvj9Hst8HHgsqoygCRpTEb99NGvJflakncm+clRXpBkEbAduApYDWxMsnpAvwuBG4AvjViLJKklo96n8ArgCmAW2JHknhHWU1gDTFfVgao6Cuxk8EI976Sz1vP/G7lqSVIrRr5PoaqOVNX7gTfS+XjqliEvWQIc7GnPdLc1klwCLKuq359vR0k2JZlKMjU7OztqyZKkEzTqyms/keQd3ZXXfgP4PLB02MsGbGs+wZTkHOC9wK8Me/+q2lFVk1U1OTExMUrJkqQnYNT7FP4L8FHg1VV1eMTXzADLetpLgd7XXgi8CPhMEoCLgF1J1nmxWZLGY2godC8Y/5+qet8J7nsvsCrJSuAQneU7rz72ZFV9H1jc8z6fAW40ECRpfIYOH1XVI8Czkpx3IjuuqjlgM7CHzv0Ot1bVviRbk6x7QtVKklo18iI7wB1JdgHNvEdV9WvzvaiqdgO7+7YNvEBdVVeMWIskqSWjhsLh7tc5dK4FSJIWoJFCoar+TduFSJLGb9Sps2+n5+Okx1TV3zvlFUmSxmbU4aMbex6fD7wGmDv15UiSxmnU4aM7+zbdkcTlOCVpgRl1+OiZPc1zgEk6N5tJkhaQUYeP7uSvrinMAV8HrmujIEnS+Axbee0y4GBVrey2X0/nesLXgXtbr06SdFoNu6P5g8BRgCQvB94NfAj4PrCj3dIkSafbsOGjRVX1YPfxLwE7quoTwCeS3NVuaZKk023YmcKiJMeC45XAp3ueG/V6hCTpSWLYD/aPAp9N8gDwl8DnAJI8j84QkiRpAZk3FKrqXUk+BTwb+IOqOvYJpHOA69suTpJ0eg0dAqqqLw7Y9qftlCNJGqeR12iWJC18hoIkqWEoSJIarYZCkrVJ9ieZTnLzgOffmOSeJHcl+eMkq9usR5I0v9ZCIckiYDtwFbAa2Djgh/5HqurFVfVSYBsw7/KekqR2tXmmsAaYrqoDVXUU2Ams7+1QVT/oaV7AgIV8JEmnT5t3JS8BDva0Z4DL+zsleRPwZuA8YOBKbkk2AZsAli9ffsoLlSR1tHmmkAHbBi3pub2qngv8C+Dtg3ZUVTuqarKqJicmJk5xmZKkY9oMhRlgWU97KXB4nv47gZ9vsR5J0hBthsJeYFWSlUnOAzYAu3o7JFnV0/z7wJ+1WI8kaYjWrilU1VySzcAeYBFwS1XtS7IVmKqqXcDmJFcCPwK+C7y+rXokScO1Ov11Ve0Gdvdt29Lz+J+1+f6SpBPjHc2SpIahIElqGAqSpIahIElqGAqSpIahIElqGAqSpIahIElqGAqSpIahIElqGAqSpIahIElqGAqSpIahIElqtDp1ts5sN910E0eOHOGiiy5i27Zt4y5H0hnAUDiLHTlyhEOHDo27DElnEIePJEmNVkMhydok+5NMJ7l5wPNvTnJvkruTfCrJc9qsR5I0v9ZCIckiYDtwFbAa2JhkdV+3LwOTVXUxcBvgwLYkjVGbZwprgOmqOlBVR4GdwPreDlV1e1X9Rbf5RWBpi/VIkoZoMxSWAAd72jPdbcdzHfA/W6xHkjREm58+yoBtNbBj8o+ASeBnjvP8JmATwPLly09VfZKkPm2eKcwAy3raS4HD/Z2SXAm8DVhXVQ8P2lFV7aiqyaqanJiYaKVYSVK7Zwp7gVVJVgKHgA3A1b0dklwCfBBYW1XfbrGWx7j0Lf/1dL3VGe3CBx5iEfDNBx7y7wS48z2/PO4SpLFr7UyhquaAzcAe4D7g1qral2RrknXdbu8BngZ8PMldSXa1VY8kabhW72iuqt3A7r5tW3oeX9nm+0uSTox3NEuSGoaCJKlhKEiSGoaCJKlhKEiSGoaCJKlhKEiSGq68dhZ79LwLHvOnJBkKZ7E/X/XqcZcg6Qzj8JEkqWEoSJIahoIkqWEoSJIahoIkqWEoSJIahoIkqWEoSJIarYZCkrVJ9ieZTnLzgOdfnuRPkswleW2btUiShmstFJIsArYDVwGrgY1JVvd1+yZwLfCRtuqQJI2uzWku1gDTVXUAIMlOYD1w77EOVfX17nOPtliHJGlEbQ4fLQEO9rRnutskSWeoNkMhA7bVE9pRsinJVJKp2dnZkyxLknQ8bYbCDLCsp70UOPxEdlRVO6pqsqomJyYmTklxkqTHazMU9gKrkqxMch6wAdjV4vtJkk5Sa6FQVXPAZmAPcB9wa1XtS7I1yTqAJJclmQFeB3wwyb626pEkDdfqIjtVtRvY3bdtS8/jvXSGlSRJZwDvaJYkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNQwFSVLDUJAkNVoNhSRrk+xPMp3k5gHP/1iSj3Wf/1KSFW3WI0maX2uhkGQRsB24ClgNbEyyuq/bdcB3q+p5wHuBf99WPZKk4do8U1gDTFfVgao6CuwE1vf1WQ98qPv4NuCVSdJiTZKkebQZCkuAgz3tme62gX2qag74PvCsFmuSJM3j3Bb3Peg3/noCfUiyCdjUbf4wyf6TrE1/ZTHwwLiLOBPkP7x+3CXosTw2j/nXp2QA5TmjdGozFGaAZT3tpcDh4/SZSXIu8Azgwf4dVdUOYEdLdZ7VkkxV1eS465D6eWyOR5vDR3uBVUlWJjkP2ADs6uuzCzj269lrgU9X1ePOFCRJp0drZwpVNZdkM7AHWATcUlX7kmwFpqpqF/CfgQ8nmaZzhrChrXokScPFX8zPbkk2dYfnpDOKx+Z4GAqSpIbTXEiSGoaCGkmuSPL7465DC0OSG5Lcl+R3W9r/O5Lc2Ma+z2ZtfiRV0tntnwJXVdX94y5Eo/NMYYFJsiLJ15L8dpKvJvndJFcmuSPJnyVZ0/36fJIvd/98wYD9XJDkliR7u/36pyiRjivJfwL+JrArydsGHUtJrk3ye0k+meT+JJuTvLnb54tJntnt94bua7+S5BNJnjrg/Z6b5H8luTPJ55K88PR+xwuHobAwPQ94H3Ax8ELgauCngRuBfwl8DXh5VV0CbAH+7YB9vI3OfSOXAa8A3pPkgtNQuxaAqnojnZtVXwFcwPGPpRfROT7XAO8C/qJ7XH4B+OVun/9WVZdV1UuA++hMpNlvB3B9VV1K5zj/QDvf2cLn8NHCdH9V3QOQZB/wqaqqJPcAK+jcOf6hJKvoTCvylAH7eDWwrmfM9nxgOZ3/lNKJON6xBHB7VT0EPJTk+8Anu9vvofNLDcCLkvwq8OPA0+jc+9RI8jTg7wAf75lP88fa+EbOBobCwvRwz+NHe9qP0vk3fyed/4z/sLuGxWcG7CPAa6rKeaZ0sgYeS0kuZ/ixCvA7wM9X1VeSXAtc0bf/c4DvVdVLT23ZZyeHj85OzwAOdR9fe5w+e4Drj01lnuSS01CXFqaTPZYuBL6V5CnANf1PVtUPgPuTvK67/yR5yUnWfNYyFM5O24B3J7mDzhQkg7yTzrDS3Um+2m1LT8TJHkv/CvgS8Id0rocNcg1wXZKvAPt4/NotGpF3NEuSGp4pSJIahoIkqWEoSJIahoIkqWEoSJIahoJ0Arrz+OxLcneSu7o3YEkLhnc0SyNK8jLgHwB/q6oeTrIYOG/MZUmnlGcK0uieDTxQVQ8DVNUDVXU4yaVJPtudoXNPkmcnObc7s+cVAEneneRd4yxeGoU3r0kj6k689sfAU4E/Aj4GfB74LLC+qmaT/BLws1X1j5P8JHAbcAOdu8gvr6qj46leGo3DR9KIquqHSS4F/i6dKaA/Bvwqnemf/7A7tc8i4Fvd/vuSfJjOzJ8vMxD0ZGAoSCegqh6hM6vsZ7pTkb8J2FdVLzvOS14MfA/4G6enQunkeE1BGlGSF3TXoDjmpXTWl5joXoQmyVO6w0Yk+QXgWcDLgfcn+fHTXbN0orymII2oO3T063QWe5kDpoFNwFLg/XSmJD8X+I/Af6dzveGVVXUwyQ3ApVX1+nHULo3KUJAkNRw+kiQ1DAVJUsNQkCQ1DAVJUsNQkCQ1DAVJUsNQkCQ1DAVJUuP/A9ldS2UaP3OcAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(x=\"Sex\", y=\"Survived\", data=train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Quel impact de l'age sur la chance de survie ?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#classer les ages dans catégories\n",
    "train[\"Age\"] = train[\"Age\"].fillna(-0.5)\n",
    "test[\"Age\"] = test[\"Age\"].fillna(-0.5)\n",
    "bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]\n",
    "labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']\n",
    "train['AgeGroup'] = pd.cut(train[\"Age\"], bins, labels = labels)\n",
    "test['AgeGroup'] = pd.cut(test[\"Age\"], bins, labels = labels)\n",
    "\n",
    "#draw a bar plot of Age vs. survival\n",
    "sns.barplot(x=\"AgeGroup\", y=\"Survived\", data=train)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Nettoyage du dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Combien de valeurs nulles ?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age              0\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             0\n",
       "Cabin          687\n",
       "Embarked         2\n",
       "AgeGroup         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(891, 13)"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#nombre de ligne eet de collones\n",
    "train.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Notre échantillon d'entrainement est petit, eliminer les Nan et donc la ligne entière reduirait encore plus la taille de notre jeu d'entrainement"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Comment conserver l'information de la ligne (observation) malgré ces Nan"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>AgeGroup</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>756</th>\n",
       "      <td>757</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Carlsson, Mr. August Sigfrid</td>\n",
       "      <td>male</td>\n",
       "      <td>28.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>350042</td>\n",
       "      <td>7.7958</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>Young Adult</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>324</th>\n",
       "      <td>325</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Sage, Mr. George John Jr</td>\n",
       "      <td>male</td>\n",
       "      <td>-0.5</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>CA. 2343</td>\n",
       "      <td>69.5500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>Unknown</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>294</th>\n",
       "      <td>295</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Mineff, Mr. Ivan</td>\n",
       "      <td>male</td>\n",
       "      <td>24.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>349233</td>\n",
       "      <td>7.8958</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>Student</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>Young Adult</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>128</th>\n",
       "      <td>129</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Peter, Miss. Anna</td>\n",
       "      <td>female</td>\n",
       "      <td>-0.5</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2668</td>\n",
       "      <td>22.3583</td>\n",
       "      <td>F E69</td>\n",
       "      <td>C</td>\n",
       "      <td>Unknown</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass                          Name     Sex  \\\n",
       "756          757         0       3  Carlsson, Mr. August Sigfrid    male   \n",
       "324          325         0       3      Sage, Mr. George John Jr    male   \n",
       "294          295         0       3              Mineff, Mr. Ivan    male   \n",
       "2              3         1       3        Heikkinen, Miss. Laina  female   \n",
       "128          129         1       3             Peter, Miss. Anna  female   \n",
       "\n",
       "      Age  SibSp  Parch            Ticket     Fare  Cabin Embarked  \\\n",
       "756  28.0      0      0            350042   7.7958    NaN        S   \n",
       "324  -0.5      8      2          CA. 2343  69.5500    NaN        S   \n",
       "294  24.0      0      0            349233   7.8958    NaN        S   \n",
       "2    26.0      0      0  STON/O2. 3101282   7.9250    NaN        S   \n",
       "128  -0.5      1      1              2668  22.3583  F E69        C   \n",
       "\n",
       "        AgeGroup  \n",
       "756  Young Adult  \n",
       "324      Unknown  \n",
       "294      Student  \n",
       "2    Young Adult  \n",
       "128      Unknown  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.sample(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Nous pouvons remplacer ces Nan par une valeur defaut ? Ou essayer de le remplacer par une valeur proche de son profil (qualifier par les autre variables)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Comment gérer le nom des personne devons nous tout jeter ou peut etre conserver des informations comme son titre (Mr / Miss / Lord / ... ). Ces informations ont certainement une corrélation avec le target que nous souhaitons prédire (A t il survi ?)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Colonne cabin"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Eliminons cette colonne. En effet, trop de valeurs sont absentes (> 650)\n",
    "train = train.drop(['Cabin'], axis = 1)\n",
    "test = test.drop(['Cabin'], axis = 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Colonne ticket "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Eliminons egalement la colonne ticket. En effet quasiment chaque ligne est différente (680)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',\n",
       "       'Parch', 'Fare', 'Embarked', 'AgeGroup'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train = train.drop(['Ticket'], axis = 1)\n",
    "test = test.drop(['Ticket'], axis = 1)\n",
    "train.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Colonne embarked"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Nombre de passager ayant embarqué à Southampton (S):\n",
      "644\n",
      "Nombre de passager ayant embarqué à Cherbourg (C):\n",
      "168\n",
      "Nombre de passager ayant embarqué à Queenstown (Q):\n",
      "77\n"
     ]
    }
   ],
   "source": [
    "print(\"Nombre de passager ayant embarqué à Southampton (S):\")\n",
    "southampton = train[train[\"Embarked\"] == \"S\"].shape[0]\n",
    "print(southampton)\n",
    "\n",
    "print(\"Nombre de passager ayant embarqué à Cherbourg (C):\")\n",
    "cherbourg = train[train[\"Embarked\"] == \"C\"].shape[0]\n",
    "print(cherbourg)\n",
    "\n",
    "print(\"Nombre de passager ayant embarqué à Queenstown (Q):\")\n",
    "queenstown = train[train[\"Embarked\"] == \"Q\"].shape[0]\n",
    "print(queenstown)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Nous remplaçons les valeurs manquantes par S qui represente la majorité des valeurs de cette colonne"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "#remplacer les fillnas par des S\n",
    "train[\"Embarked\"] = train[\"Embarked\"].fillna(\"S\")\n",
    "test[\"Embarked\"] = test[\"Embarked\"].fillna(\"S\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId    0\n",
       "Survived       0\n",
       "Pclass         0\n",
       "Name           0\n",
       "Sex            0\n",
       "Age            0\n",
       "SibSp          0\n",
       "Parch          0\n",
       "Fare           0\n",
       "Embarked       0\n",
       "AgeGroup       0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#je tcheck\n",
    "train.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>AgeGroup</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "      <td>Student</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "      <td>Adult</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "      <td>Young Adult</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "      <td>Young Adult</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "      <td>Young Adult</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch     Fare Embarked     AgeGroup  \n",
       "0      0   7.2500        S      Student  \n",
       "1      0  71.2833        C        Adult  \n",
       "2      0   7.9250        S  Young Adult  \n",
       "3      0  53.1000        S  Young Adult  \n",
       "4      0   8.0500        S  Young Adult  "
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Colonne age"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Beaucoup de valeurs sont manquanes nous ne pouvons pas les compléter par une valeur unique. Nous allons essayer de trouver un moyen de réduire le risque lié au remplissage de ces cellules."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>Sex</th>\n",
       "      <th>female</th>\n",
       "      <th>male</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Title</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Capt</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Col</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Countess</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Don</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Dr</th>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Jonkheer</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Lady</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Major</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Master</th>\n",
       "      <td>0</td>\n",
       "      <td>40</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Miss</th>\n",
       "      <td>182</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Mlle</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Mme</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Mr</th>\n",
       "      <td>0</td>\n",
       "      <td>517</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Mrs</th>\n",
       "      <td>125</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Ms</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Rev</th>\n",
       "      <td>0</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sir</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Sex       female  male\n",
       "Title                 \n",
       "Capt           0     1\n",
       "Col            0     2\n",
       "Countess       1     0\n",
       "Don            0     1\n",
       "Dr             1     6\n",
       "Jonkheer       0     1\n",
       "Lady           1     0\n",
       "Major          0     2\n",
       "Master         0    40\n",
       "Miss         182     0\n",
       "Mlle           2     0\n",
       "Mme            1     0\n",
       "Mr             0   517\n",
       "Mrs          125     0\n",
       "Ms             1     0\n",
       "Rev            0     6\n",
       "Sir            0     1"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#On crée un group de dataset (train et test) pour nettoyer deux dataframes avec les memes opérations\n",
    "combine = [train, test]\n",
    "\n",
    "#Extraction des titres dans la colonne nom puis test\n",
    "for dataset in combine:\n",
    "    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)\n",
    "\n",
    "pd.crosstab(train['Title'], train['Sex'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Title</th>\n",
       "      <th>Survived</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Master</td>\n",
       "      <td>0.575000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Miss</td>\n",
       "      <td>0.702703</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Mr</td>\n",
       "      <td>0.156673</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Mrs</td>\n",
       "      <td>0.793651</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Rare</td>\n",
       "      <td>0.250000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Royal</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Title  Survived\n",
       "0  Master  0.575000\n",
       "1    Miss  0.702703\n",
       "2      Mr  0.156673\n",
       "3     Mrs  0.793651\n",
       "4    Rare  0.250000\n",
       "5   Royal  1.000000"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Remplacement de certains titres peu fréquents par un label commun\n",
    "for dataset in combine:\n",
    "    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col',\n",
    "    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')\n",
    "    \n",
    "    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')\n",
    "    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')\n",
    "    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')\n",
    "    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')\n",
    "\n",
    "train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "mapper chaque titre avec une valeur numérique. C'est à dire, par exemple, remplacer Miss par 1, ..."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Mr', 'Mrs', 'Miss', 'Master', 'Rare', 'Royal'], dtype=object)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train['Title'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "for dataset in combine:\n",
    "    \n",
    "    dataset['Title'] = dataset['Title'].replace('Mr',1)\n",
    "    dataset['Title'] = dataset['Title'].replace('Mrs',2)\n",
    "    dataset['Title'] = dataset['Title'].replace('Miss',3)\n",
    "    dataset['Title'] = dataset['Title'].replace('Master',4)\n",
    "    dataset['Title'] = dataset['Title'].replace('Rare',5)\n",
    "    dataset['Title'] = dataset['Title'].replace('Royal',6)\n",
    "                                                 \n",
    "                                                 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>AgeGroup</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "      <td>Student</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "      <td>Adult</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "      <td>Young Adult</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "      <td>Young Adult</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "      <td>Young Adult</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch     Fare Embarked     AgeGroup  Title  \n",
       "0      0   7.2500        S      Student      1  \n",
       "1      0  71.2833        C        Adult      2  \n",
       "2      0   7.9250        S  Young Adult      3  \n",
       "3      0  53.1000        S  Young Adult      2  \n",
       "4      0   8.0500        S  Young Adult      1  "
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Mainenant, nous allons essayer d'estimer l'ages des passagers à partir de leurs titres."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Pour chaque titre prendre l'age le plus commun des passagers\n",
    "mr_age = train[train[\"Title\"] == 1][\"AgeGroup\"].mode() #Young Adult\n",
    "miss_age = train[train[\"Title\"] == 2][\"AgeGroup\"].mode() #Student\n",
    "mrs_age = train[train[\"Title\"] == 3][\"AgeGroup\"].mode() #Adult\n",
    "master_age = train[train[\"Title\"] == 4][\"AgeGroup\"].mode() #Baby\n",
    "royal_age = train[train[\"Title\"] == 5][\"AgeGroup\"].mode() #Adult\n",
    "rare_age = train[train[\"Title\"] == 6][\"AgeGroup\"].mode() #Adult\n",
    "\n",
    "age_title_mapping = {1: \"Young Adult\", 2: \"Student\", 3: \"Adult\", 4: \"Baby\", 5: \"Teenager\", 6: \"Child\", 7: \"Senior\"}\n",
    "\n",
    "\n",
    "\n",
    "for x in train[\"AgeGroup\"].index:\n",
    "    if train[\"AgeGroup\"].loc[x] == \"Unknown\":\n",
    "        train[\"AgeGroup\"].loc[x] = age_title_mapping[train[\"Title\"].loc[x]]\n",
    "        \n",
    "for x in test[\"AgeGroup\"].index:\n",
    "    if test[\"AgeGroup\"].loc[x] == \"Unknown\":\n",
    "        test[\"AgeGroup\"].loc[x] = age_title_mapping[test[\"Title\"].loc[x]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[Student, Adult, Young Adult, Baby, Teenager, Child, Senior]\n",
       "Categories (7, object): [Baby < Child < Teenager < Student < Young Adult < Adult < Senior]"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train[\"AgeGroup\"].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>AgeGroup</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "      <td>Student</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "      <td>Adult</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "      <td>Young Adult</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "      <td>Young Adult</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "\n",
       "   Parch     Fare Embarked     AgeGroup  Title  \n",
       "0      0   7.2500        S      Student      1  \n",
       "1      0  71.2833        C        Adult      2  \n",
       "2      0   7.9250        S  Young Adult      3  \n",
       "3      0  53.1000        S  Young Adult      2  "
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head(4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Mapping de chaque group d'age"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "##Mappind de chaque groupe\n",
    "train[\"AgeGroup\"] = train[\"AgeGroup\"].replace('Student',1)\n",
    "train[\"AgeGroup\"] = train[\"AgeGroup\"].replace('Adult',2)\n",
    "train[\"AgeGroup\"] = train[\"AgeGroup\"].replace('Young Adult',3)\n",
    "train[\"AgeGroup\"] = train[\"AgeGroup\"].replace('Baby',4)\n",
    "train[\"AgeGroup\"] = train[\"AgeGroup\"].replace('Teenager',5)\n",
    "train[\"AgeGroup\"] = train[\"AgeGroup\"].replace('Child',6)\n",
    "train[\"AgeGroup\"] = train[\"AgeGroup\"].replace('Senior',7)\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Nous pouvons effacer la colonne age ainsi que la colonne name une fois les informations sur le titre extrait\n",
    "train = train.drop(['Age'], axis = 1)\n",
    "test = test.drop(['Age'], axis = 1)\n",
    "train = train.drop(['Name'], axis = 1)\n",
    "test = test.drop(['Name'], axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>AgeGroup</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>414</th>\n",
       "      <td>415</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>170</th>\n",
       "      <td>171</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>33.5000</td>\n",
       "      <td>S</td>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>836</th>\n",
       "      <td>837</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.6625</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>231</th>\n",
       "      <td>232</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.7750</td>\n",
       "      <td>S</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>56</th>\n",
       "      <td>57</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>female</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>10.5000</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass     Sex  SibSp  Parch     Fare Embarked  \\\n",
       "414          415         1       3    male      0      0   7.9250        S   \n",
       "170          171         0       1    male      0      0  33.5000        S   \n",
       "836          837         0       3    male      0      0   8.6625        S   \n",
       "231          232         0       3    male      0      0   7.7750        S   \n",
       "56            57         1       2  female      0      0  10.5000        S   \n",
       "\n",
       "     AgeGroup  Title  \n",
       "414         2      1  \n",
       "170         7      1  \n",
       "836         1      1  \n",
       "231         3      1  \n",
       "56          1      3  "
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.sample(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Colonne sex"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Mapper les valeurs de la colonne sex"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "train['Sex'] = train['Sex'].replace('male',1)\n",
    "train['Sex'] = train['Sex'].replace('female',2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>AgeGroup</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  Sex  SibSp  Parch     Fare Embarked  \\\n",
       "0            1         0       3    1      1      0   7.2500        S   \n",
       "1            2         1       1    2      1      0  71.2833        C   \n",
       "2            3         1       3    2      0      0   7.9250        S   \n",
       "3            4         1       1    2      1      0  53.1000        S   \n",
       "4            5         0       3    1      0      0   8.0500        S   \n",
       "\n",
       "   AgeGroup  Title  \n",
       "0         1      1  \n",
       "1         2      2  \n",
       "2         3      3  \n",
       "3         3      2  \n",
       "4         3      1  "
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Embarked colonne"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>AgeGroup</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  Sex  SibSp  Parch     Fare  Embarked  \\\n",
       "0            1         0       3    1      1      0   7.2500         1   \n",
       "1            2         1       1    2      1      0  71.2833         2   \n",
       "2            3         1       3    2      0      0   7.9250         1   \n",
       "3            4         1       1    2      1      0  53.1000         1   \n",
       "4            5         0       3    1      0      0   8.0500         1   \n",
       "\n",
       "   AgeGroup  Title  \n",
       "0         1      1  \n",
       "1         2      2  \n",
       "2         3      3  \n",
       "3         3      2  \n",
       "4         3      1  "
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#mapper les valeurs de la colonne embarked\n",
    "embarked_mapping = {\"S\": 1, \"C\": 2, \"Q\": 3}\n",
    "train['Embarked'] = train['Embarked'].map(embarked_mapping)\n",
    "test['Embarked'] = test['Embarked'].map(embarked_mapping)\n",
    "\n",
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([False])"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train[\"Fare\"].isnull().unique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Colonne Fare"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/funnyeoman/anaconda3/envs/dev/lib/python3.7/site-packages/pandas/core/indexing.py:190: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy\n",
      "  self._setitem_with_indexer(indexer, value)\n"
     ]
    }
   ],
   "source": [
    "#Documentation des valeurs manquantes de l'echantillon test\n",
    "for x in test[\"Fare\"].index:\n",
    "    if pd.isnull(test[\"Fare\"].loc[x]):\n",
    "        pclass = test[\"Pclass\"].loc[x] #Pclass = 3\n",
    "        test[\"Fare\"].loc[x] = round(train[train[\"Pclass\"] == pclass][\"Fare\"].mean(), 4)\n",
    "        \n",
    "#Mapper les labels fare en valeurs numeriques\n",
    "train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])\n",
    "test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])\n",
    "\n",
    "#Elimination des colonnes\n",
    "train = train.drop(['Fare'], axis = 1)\n",
    "test = test.drop(['Fare'], axis = 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Check des echantillons Test et Train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>AgeGroup</th>\n",
       "      <th>Title</th>\n",
       "      <th>FareBand</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  Sex  SibSp  Parch  Embarked  AgeGroup  \\\n",
       "0            1         0       3    1      1      0         1         1   \n",
       "1            2         1       1    2      1      0         2         2   \n",
       "2            3         1       3    2      0      0         1         3   \n",
       "3            4         1       1    2      1      0         1         3   \n",
       "4            5         0       3    1      0      0         1         3   \n",
       "\n",
       "   Title FareBand  \n",
       "0      1        1  \n",
       "1      2        4  \n",
       "2      3        2  \n",
       "3      2        4  \n",
       "4      1        2  "
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>AgeGroup</th>\n",
       "      <th>Title</th>\n",
       "      <th>FareBand</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Young Adult</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>Adult</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>2</td>\n",
       "      <td>male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Senior</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>Young Adult</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Student</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Pclass     Sex  SibSp  Parch  Embarked     AgeGroup  Title  \\\n",
       "0          892       3    male      0      0         3  Young Adult      1   \n",
       "1          893       3  female      1      0         1        Adult      2   \n",
       "2          894       2    male      0      0         3       Senior      1   \n",
       "3          895       3    male      0      0         1  Young Adult      1   \n",
       "4          896       3  female      1      1         1      Student      2   \n",
       "\n",
       "  FareBand  \n",
       "0        1  \n",
       "1        1  \n",
       "2        2  \n",
       "3        2  \n",
       "4        2  "
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Décomposition de l'echantillon train en sous train et sous test"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Survived est la colonne à prédire de notre jeu d'entrainement. Chaque ligne correspond à un passager, il s'agit d'une observation. La colonne Survived nous indique si ce passager a survécu ou non au nauffrage du Titanic."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Nous l'appellerons target et nous conserverons egalement l'ID passenger associé"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 10 columns):\n",
      "PassengerId    891 non-null int64\n",
      "Survived       891 non-null int64\n",
      "Pclass         891 non-null int64\n",
      "Sex            891 non-null int64\n",
      "SibSp          891 non-null int64\n",
      "Parch          891 non-null int64\n",
      "Embarked       891 non-null int64\n",
      "AgeGroup       891 non-null int64\n",
      "Title          891 non-null int64\n",
      "FareBand       891 non-null category\n",
      "dtypes: category(1), int64(9)\n",
      "memory usage: 63.8 KB\n"
     ]
    }
   ],
   "source": [
    "train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "train['FareBand']=train['FareBand'].astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "predictors = train.drop(['Survived', 'PassengerId'], axis=1)\n",
    "target = train[\"Survived\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Nous gardons 20 %  de notre echantillon train qui est labelisé pour evaluer la performance de notre modele \n",
    "x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.25, random_state = 0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Modélisation / Prédiction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "83.41\n"
     ]
    }
   ],
   "source": [
    "# Gradient Boosting Classifier\n",
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "gbk = GradientBoostingClassifier()\n",
    "gbk.fit(x_train, y_train)\n",
    "y_pred = gbk.predict(x_val)\n",
    "acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)\n",
    "print(acc_gbk)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "83.86\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/funnyeoman/anaconda3/envs/dev/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n"
     ]
    }
   ],
   "source": [
    "# Random Forest\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "randomforest = RandomForestClassifier()\n",
    "randomforest.fit(x_train, y_train)\n",
    "y_pred = randomforest.predict(x_val)\n",
    "acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)\n",
    "print(acc_randomforest)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "80.72\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/funnyeoman/anaconda3/envs/dev/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.\n",
      "  \"avoid this warning.\", FutureWarning)\n"
     ]
    }
   ],
   "source": [
    "# Support Vector Machines\n",
    "from sklearn.svm import SVC\n",
    "\n",
    "svc = SVC()\n",
    "svc.fit(x_train, y_train)\n",
    "y_pred = svc.predict(x_val)\n",
    "acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)\n",
    "print(acc_svc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "debut du predict\n",
      "82.51\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import mean_squared_error\n",
    "import xgboost as xgb\n",
    "#from catboost import CatBoostClassifier, Pool\n",
    "from sklearn.metrics import accuracy_score\n",
    "import math\n",
    "\n",
    "#model = CatBoostClassifier()\n",
    "model = xgb.XGBClassifier()\n",
    "model.fit(x_train, y_train)\n",
    "\n",
    "model.fit(x_train, y_train)\n",
    "print(\"debut du predict\")\n",
    "y_pred = model.predict(x_val)\n",
    "acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)\n",
    "print(acc_svc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "import xgboost as xgb"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Creation d'un fichier de soumission Kaggle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "could not convert string to float: 'Baby'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-53-bfa994e2b8b5>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m#Le fichier de soumission contione un colonne passenger ID et une colonne survie ou non\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mids\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtest\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'PassengerId'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mpredictions\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgbk\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtest\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdrop\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'PassengerId'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;31m#Construction sous format csv\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/envs/dev/lib/python3.7/site-packages/sklearn/ensemble/gradient_boosting.py\u001b[0m in \u001b[0;36mpredict\u001b[0;34m(self, X)\u001b[0m\n\u001b[1;32m   2035\u001b[0m             \u001b[0mThe\u001b[0m \u001b[0mpredicted\u001b[0m \u001b[0mvalues\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2036\u001b[0m         \"\"\"\n\u001b[0;32m-> 2037\u001b[0;31m         \u001b[0mscore\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdecision_function\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2038\u001b[0m         \u001b[0mdecisions\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mloss_\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_score_to_decision\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mscore\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2039\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mclasses_\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtake\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdecisions\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/envs/dev/lib/python3.7/site-packages/sklearn/ensemble/gradient_boosting.py\u001b[0m in \u001b[0;36mdecision_function\u001b[0;34m(self, X)\u001b[0m\n\u001b[1;32m   1989\u001b[0m             \u001b[0;34m[\u001b[0m\u001b[0mn_samples\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1990\u001b[0m         \"\"\"\n\u001b[0;32m-> 1991\u001b[0;31m         \u001b[0mX\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcheck_array\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mDTYPE\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0morder\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m\"C\"\u001b[0m\u001b[0;34m,\u001b[0m  \u001b[0maccept_sparse\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'csr'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1992\u001b[0m         \u001b[0mscore\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_decision_function\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1993\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mscore\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m~/anaconda3/envs/dev/lib/python3.7/site-packages/sklearn/utils/validation.py\u001b[0m in \u001b[0;36mcheck_array\u001b[0;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)\u001b[0m\n\u001b[1;32m    525\u001b[0m             \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    526\u001b[0m                 \u001b[0mwarnings\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msimplefilter\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'error'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mComplexWarning\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 527\u001b[0;31m                 \u001b[0marray\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0marray\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0morder\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0morder\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    528\u001b[0m             \u001b[0;32mexcept\u001b[0m \u001b[0mComplexWarning\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    529\u001b[0m                 raise ValueError(\"Complex data not supported\\n\"\n",
      "\u001b[0;32m~/anaconda3/envs/dev/lib/python3.7/site-packages/numpy/core/numeric.py\u001b[0m in \u001b[0;36masarray\u001b[0;34m(a, dtype, order)\u001b[0m\n\u001b[1;32m    499\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    500\u001b[0m     \"\"\"\n\u001b[0;32m--> 501\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0marray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0ma\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0morder\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0morder\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    502\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    503\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mValueError\u001b[0m: could not convert string to float: 'Baby'"
     ]
    }
   ],
   "source": [
    "#Le fichier de soumission contione un colonne passenger ID et une colonne survie ou non\n",
    "ids = test['PassengerId']\n",
    "predictions = gbk.predict(test.drop('PassengerId', axis=1))\n",
    "\n",
    "#Construction sous format csv\n",
    "output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })\n",
    "output.to_csv('submission.csv', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
