{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![class_reg](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/04/regression-vs-classification-in-machine-learning.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5b-pJCzwF9kq"
   },
   "source": [
    "___\n",
    "#### <font color=#FF00BB> due midnight Saturday </font> \n",
    "___\n",
    "# <font color=#00BBFF> HW: Introduction to Classification and Project</font>\n",
    "\n",
    "## <font color=green> Question 1: Classification (40 points) </font>\n",
    "\n",
    "The goal of this HW is for you to explore classification in more depth, in part because you _might_ want to use it in your final project but also because it is a very important tool for finding patterns in data. Many of the datasets you are using are great for classification; for example, the classic iris and penguin datasets. \n",
    "\n",
    "This HW has two parts: classification and your final project. \n",
    "\n",
    "This HW illustrates an extremely important aspect of classification. What you will see is that, despite some naive similarities, classification is quite different from regression in practice.\n",
    "\n",
    "Once again, you'll use the iris dataset since you now have a lot of intuition for it. \n",
    "\n",
    "Because you need to finalize your project, I want the first part of this HW to be short. What I have done is written the entire code for you. All you need to do is read the notebook, answer some questions and comment the code. \n",
    "\n",
    "One comment before we dive into the code. The iris dataset is known to be \"easy\". If we use these powerful ML models on the complete dataset, we often get a perfect prediction. For this reason, I grab only the first two columns in line 11. \n",
    "\n",
    "Change the choice of two columns to be 3 or 4 columns and compare the scores you get; this is a very nice ML lesson. \n",
    "\n",
    "In your Jupyter environment, turn on line numbers.\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "jXFQC3Fg-Vs8"
   },
   "source": [
    "____\n",
    "\n",
    "## Step 1: Get Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "id": "4B2yalrw-eQx",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# comment every line, unless it is very obvious (e.g., NumPy)\n",
    "# in particular, what do the sklearn libraries do? (use the online documentation)\n",
    "\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn import datasets\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "iris_data = datasets.load_iris()\n",
    "X = iris_data.data[:, :2] # what is happening here?\n",
    "y = iris_data.target\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "QyBiLTsi_EgU"
   },
   "source": [
    "____\n",
    "## Step 2: Preprocess Data\n",
    "\n",
    "What is being done below contains a small subtlety. In line 8, a scaler is instantiated (created for use), but it isn't used until line 9. In line 9, the scaler only learns from the data how to do the scaling - *it doesn't actually do the scaling!* \n",
    "\n",
    "Why? This is so that you can use that information to scale anything you want using the rules it learned. For example, we learn the scaling from the training data (line 9), but we scale **both** the training and test in lines 11 and 13. \n",
    "\n",
    "This is a very common pattern in `sklearn`: instantiate an object, then use it on the data separately, perhaps multiple times. \n",
    "\n",
    "Why we would do this? The reason is that in the real world we would only have the training data - we cannot learn how to scale the data from data we have never seen. But, once we learn a good scaling and train our ML models using that rule, we should apply it to future data, which could be our test data or real world data once the ML is deployed. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "id": "p0rUyw6z_MZJ",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# comment every line in detail\n",
    "\n",
    "start_state = 42\n",
    "test_fraction = 0.2\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=start_state)\n",
    "\n",
    "my_scaler = StandardScaler()\n",
    "my_scaler.fit(X_train) # why isn't y_train used here? \n",
    "\n",
    "X_train_scaled = my_scaler.transform(X_train)\n",
    "\n",
    "X_test_scaled = my_scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 392
    },
    "executionInfo": {
     "elapsed": 349,
     "status": "ok",
     "timestamp": 1668983517083,
     "user": {
      "displayName": "Michael Murillo",
      "userId": "04445914509865448303"
     },
     "user_tz": 300
    },
    "id": "0Zpf7O3LBOgM",
    "outputId": "44355665-a43b-43a4-ccd7-b33fbc6b348b",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x7f94f9bf1dc0>"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 600x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# comment non-obvious lines\n",
    "\n",
    "plt.figure(figsize=(6,6))\n",
    "plt.scatter(X_train[:,0], X_train[:,1], label='original')\n",
    "plt.scatter(X_train_scaled[:,0], X_train_scaled[:,1], label='scaled')\n",
    "plt.grid(alpha=0.15)\n",
    "plt.legend()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "kbBEbAEeC-vl"
   },
   "source": [
    "____\n",
    "## Step 3: Learn!\n",
    "\n",
    "I don't usually like putting import statements down here, but I wanted to group them away from the others above. \n",
    "\n",
    "What I really want you to see is that the estimator could be anything - here, I use two and most of the code is totally unchanged! Add many more classifiers  - run them all!\n",
    "\n",
    "Note that the ML libraries are very short and follow the same pattern: this allows you to very quickly and easily compare many classification libraries. It is a good habit to always compare several estimators, since you don't usually know which one will perform best. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "id": "77M72dWCCDo5",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# switch between the classifiers and rerun the code\n",
    "\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "my_classifier = DecisionTreeClassifier(criterion='gini', random_state=0)\n",
    "\n",
    "# from sklearn import neighbors\n",
    "# n_neighbors = 2\n",
    "# my_classifier = neighbors.KNeighborsClassifier(n_neighbors)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 3,
     "status": "ok",
     "timestamp": 1668983519461,
     "user": {
      "displayName": "Michael Murillo",
      "userId": "04445914509865448303"
     },
     "user_tz": 300
    },
    "id": "pFcenuoQDIyJ",
    "outputId": "e20b47e3-81e1-4f7e-97c9-7e9b95ed3731",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6333333333333333"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# note that this portion of the code doesn't care which estimator you chose\n",
    "\n",
    "my_model = my_classifier.fit(X_train_scaled, y_train)\n",
    "\n",
    "my_model.score(X_test_scaled, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Comment on your findings for the previous estimator comparisons. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1UJAMWPmEGwq"
   },
   "source": [
    "____\n",
    "\n",
    "### Predictions\n",
    "\n",
    "With everything trained, you can deploy the model and make predictions. All you need to do is pass in an $X$ and the model will return a $y$. Let's try this with the test data. Comment the code and add a markdown cell to explain your results. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "id": "Kxmg8Jt2DouD",
    "tags": []
   },
   "outputs": [],
   "source": [
    "y_pred = my_model.predict(X_test_scaled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 4,
     "status": "ok",
     "timestamp": 1668983521418,
     "user": {
      "displayName": "Michael Murillo",
      "userId": "04445914509865448303"
     },
     "user_tz": 300
    },
    "id": "vHEa6PVBEFW8",
    "outputId": "cf1d47bd-5c00-46ee-b2bf-e98d36034098",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  0,  0, -1, -1,  0,  0,  1,  0, -1,  1,  0, -1,  0,  0, -1,  0,\n",
       "       -1,  0,  0,  0,  1,  0,  0,  0,  1,  1,  0,  0,  0])"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test - y_pred"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Describe in detail what this does."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.7       , 0.73333333, 0.73333333, 0.8       , 0.66666667])"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "cross_val_score(my_classifier, X, y, cv=5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Kqp5UbNYHHWp"
   },
   "source": [
    "At this point, you should go back and play with the number of columns and the scaling. If you use 3 columns does the score go up or down? What if you comment out the code that scales the data - does that impact the score? \n",
    "\n",
    "Create a markdown cell to answer these questions in detail. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "PDxcgg8IH57N"
   },
   "source": [
    "____\n",
    "____\n",
    "## Step 4:  The Interesting Part!!\n",
    "\n",
    "Everything above is pretty straightforward. If you were doing, say, regression, nearly everything would be identical once you swapped regression estimators for the classifiers. \n",
    "\n",
    "Now, I want to introduce you to a very important metric that is _used for classification specifically_. This metric is called the confusion matrix. \n",
    "\n",
    "Let's bring the code in. (Again, I would normally put this at the top, but I want to you see it here...)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "id": "6h1uRZUAEWhj",
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "6DR_CevSImNJ"
   },
   "source": [
    "You might have heard of a confusion matrix before. If not, you have certainly heard of \"false positives\" and \"false negatives\" - we use that terminology in our everyday speech. [Read this to review/learn the basic ideas.](https://en.wikipedia.org/wiki/Confusion_matrix) \n",
    "\n",
    "The [code](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix) is simple:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "id": "WSvJAI_5JAVZ",
    "tags": []
   },
   "outputs": [],
   "source": [
    "conf_mat = confusion_matrix(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 4,
     "status": "ok",
     "timestamp": 1668983525796,
     "user": {
      "displayName": "Michael Murillo",
      "userId": "04445914509865448303"
     },
     "user_tz": 300
    },
    "id": "Ps09AnPOJIXp",
    "outputId": "d5c0fef8-0946-42e4-eb6b-01a5a2f90d9a",
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[9 1 0]\n",
      " [0 4 5]\n",
      " [0 5 6]]\n"
     ]
    }
   ],
   "source": [
    "print(conf_mat)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 320
    },
    "executionInfo": {
     "elapsed": 316,
     "status": "ok",
     "timestamp": 1668983738132,
     "user": {
      "displayName": "Michael Murillo",
      "userId": "04445914509865448303"
     },
     "user_tz": 300
    },
    "id": "GBvceKbTJdcV",
    "outputId": "46621871-81b4-4bb5-c033-98c81ed2aca9",
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7f94f9d492e0>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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3l29evXSq6++qq1bt6p58+bnPWdiYqIefvhhPfTQQ0pISNBll12moqIibd26VSNHjlR6erqSkpK0YsUKNW7cWImJiUpNTdXEiRM1duxYuVwu9evXT6WlpcrPz9fRo0eVk5Ojm266SePHj9fIkSP1+OOP6+uvv9af//zngH69LVq0kNvt1osvvqiBAwfq448/1iuvvFLpuPj4eN1777164YUXFB8frzFjxujSSy+tSNpPPvmkBgwYoMzMTN1www2KiYnR5s2btWXLFj3zzDOBfyMAVODubuB7DodDy5cv1+WXX67bb79dLVu21NChQ/X1119X3I09ZMgQPfnkk3r44YfVrVs37dmzR3fffffPnveJJ57Q/fffryeffFJt2rTRkCFDdOjQIUln53tfeOEFzZgxQxkZGRo0aJAkadSoUfrLX/6iOXPmqEOHDurdu7fmzJlT8chWzZo19a9//Uvbtm1Tly5dNH78eE2dOjWgX2/nzp01bdo0TZ06Ve3bt9err76q3NzcSsfVqFFDDz/8sG666SZlZ2crKSlJr732WsXXr7nmGr3xxhtauXKlevTooUsvvVTTpk1T06ZNA4oHQGUOIxSTWwAAIOSopAEAsCiSNAAAFkWSBgDAokjSAABYFEkaAACLIkkDAGBRJGkAACyKJA0AgEWRpAEAsCiSNAAAFkWSBgDAokjSAABY1P8H/XUWJUwlInAAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "ConfusionMatrixDisplay.from_estimator(my_classifier, X_test_scaled, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Dsi5ydwNK893"
   },
   "source": [
    "In a markdown cell, describe what [this displays](https://en.wikipedia.org/wiki/Confusion_matrixhttps://en.wikipedia.org/wiki/Confusion_matrix). Interpret its meaning in terms of FPs, FNs, etc. Change things (number of columns, estimator) to see how the confusion matrix is impacted. \n",
    "\n",
    "If you do classification for your final project, you want to include a confusion! It tells you how well your classifier is working; and, it tells **how** it fails when it does. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "____\n",
    "\n",
    "# <font color=#AA00FF> Question 2: Final Project (10 Points) </font>\n",
    "\n",
    "For the second part of the HW, we are focusing on the final project. While the first project was mainly focused on exploration of a dataset using data science methods (EDA, IDA, Visualization, Imputation, ...), the final project is focused on implementing modeling methods which could be drawn from machine learning methods (linear regression, neural networks, support vector machines, gaussian process regression/classification, KNN, and any other regression-classification technique) as an exploratory or a predictive tool. <font color=darkred> **Detailed instructions are provided in the \"Final Project\" Documentation (can be found on D2L and Slack), which will guide your effort to develop appropriate web app for the submission!!** <font> \n",
    "\n",
    "    \n",
    "### Provide a summary of the final project requirements based on the information in the Final Project documentation.\n",
    "\n",
    "____\n",
    "\n",
    "    \n",
    "    \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now that you have understood the description and the requirements of the final project, for the remainder of this HW, use the time to complete your project. If you have any issues or questions let us know! \n"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
