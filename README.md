# CodenamesAI

**A composite model for playing the codenames board game**

### Usage

1. Make an account on PlaygrounRL with this [referall link](https://playgroundrl.com/referralsignup?referral_email=jatinker@cs.stanford.edu)
2. Download your `auth.json` to the main directory
3. Update the line ```model_name="default_model_name"``` in `spymaster/codenames.py` with your own unique model name
4. `pip install -r requirements.txt` 
5. run `python3.11 -m gameplay.codenames` or `./run10x.sh` from the main directory

### Model Info / Pipeline
1. Initial categories are downloaded from the [semcat dataset](https://github.com/avaapm/SEMCATdataset2018/tree/master)
2. Those categories are fed into a [DSP](https://github.com/stanfordnlp/dspy) few-shot learning GPT3.5 pipeline to generate 500 more categories from common themes, nouns, and adjectives in the top 1000 english words (`see category_generation/README.md`)
3. DSP is re-applied as a RLHF model (with some manually selected gold standard categories) to check if categories are comprehensive using the top `20` related Glove vectors (by cosine similarity) (`see category_generation/README.md`)
4. Decision tree and Neural Network models with uniform hyperparameter sweep are fit to predict categories from word glove embeddings (multi-label classification) (see `models/README.md`, `models/decisiontree.py` and `models/neuralnet.py`)
5. The best neural network models are selected based on test accuracy
6. A Q-learning MDP is used to detirmine a scoring metric, as a function of the game state, to rank categories generated by the neural network model. Early in the game the Q table generally prefers a higher risk scoring metric; later in the game it generally prefers lower risk. (see `gameplay/README.md`, `data/q.json`, `gameplay/hint.py`, `gameplay/algo.py`, `gameplay/mdp.py`)

### Citations
See `citations/README.md`

```@article{khattab2022demonstrate,
  title={Demonstrate-Search-Predict: Composing Retrieval and Language Models for Knowledge-Intensive {NLP}},
  author={Khattab, Omar and Santhanam, Keshav and Li, Xiang Lisa and Hall, David and Liang, Percy and Potts, Christopher and Zaharia, Matei},
  journal={arXiv preprint arXiv:2212.14024},
  year={2022}
}
@ARTICKLE {SEMCAT,
          author       = "Senel L. K., Utlu I., Yucesoy V., Koc A., Cukur T.",
          title        = "Semantic Structure and Interpretability of Word Embeddings",
          year         = "2018",
          organization = "IEEE/ACM Transactions on Audio, Speech, and Language Processing"
}
            
