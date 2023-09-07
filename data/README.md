# Data Files


### Inventory

1. `600_gpt_post_audit.csv`: Training Matrix, 11.3k examples, 600 categories. `True` if the row can be described by the column. 
2. `audit_training_examples.yaml`: Training data for the audit step. For each word (category), the first list represents options: should <option> be added to the category <word>? the second list represents the subset of options that should be added to the category.
3. `q.json`: Q-table used to weight clue scores during gameplay. Contains current scores, and update count
4. `semcat_categories.json`: Seed categories from the orignal SEMCAT paper. Used as a base to generate more examples from chatGPT
5. `wordlist_`: most common english words and words used in codenames from various sources
