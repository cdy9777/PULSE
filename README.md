# PULSE : Socially-Aware User Representation Modeling toward Extremely Parameter-Efficient Graph Collaborative Filtering
📄 *Proceedings of The Web Conference (WWW) 2026*

---

This repository contains the **official implementation of PULSE**, a socially-aware and parameter-efficient graph collaborative filtering framework. PULSE is implemented on top of the **[SSLRec](https://github.com/HKUDS/SSLRec)** framework (WSDM 2024).

(Paper information will be uploaded)

---
### 1️⃣ Prerequisites

Ensure you have Python 3.8+ and PyTorch installed. You can set up the environment with:

````bash
pip install -r requirements.txt
````
### 2️⃣ Training
Run the following command to train PULSE on a dataset:
```bash
python main.py --model pulse --dataset [douban-book, yelp, epinions] --cuda [CUDA NUM]
````

### 3️⃣ Hyperparameter Search Range
The following hyperparameter ranges were used for model selection and all reported results:
```
- Number of GNN layers: {2, 3, 4}
- Regularization weight: {1.0e-5, 1.0e-6, 1.0e-7}
- SSL weight: {0.1, 0.2, 0.3, 0.5, 1.0}
- Temperature: {0.1, 0.2, 0.3, 0.5, 1.0}
- Mask ratio: {0.1, 0.2}
  ````
---

Reference:
https://github.com/HKUDS/SSLRec/tree/main

---
### License

This project is licensed under the Apache License 2.0.

This repository is based on an official implementation licensed under
the Apache License 2.0, with modifications by Doyun Choi.
