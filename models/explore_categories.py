import pandas as pd
import sys



def list_categories():
    categories = df.columns.tolist()
    print("Available categories:", ', '.join(categories))

def list_words_in_category(category):
    if category in df.columns:
        words_in_category = df[df[category]].index.tolist()
        print(f"Words in '{category}':", ', '.join(words_in_category))
        count_words_in_category(category)
    else:
        print("Invalid category.")

def count_words_in_category(category):
    if category in df.columns:
        count = df[category].sum()
        print(f"Number of words in '{category}': {count}")
    else:
        print("Invalid category.")

def list_categories_for_word(word):
    if word in df.index:
        categories_for_word = df.columns[df.loc[word].astype(bool)].tolist()
        print(f"Categories for '{word}':", ', '.join(categories_for_word))
    else:
        print("Invalid word.")

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], index_col=0)
    if "top_words" in df.columns:
        df = df.drop(columns=["original_definition","top_words"])


    while True:
        print("\nMenu:")
        print("1. List all categories")
        print("2. List words in a category")
        print("3. List categories for a word")
        print("4. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            list_categories()
        elif choice == '2':
            category = input("Enter the category: ")
            list_words_in_category(category)
        elif choice == '3':
            word = input("Enter the word: ")
            list_categories_for_word(word)
        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please choose again.")






