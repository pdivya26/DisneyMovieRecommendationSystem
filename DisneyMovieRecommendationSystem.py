from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox, ttk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import random

global t_entry, g_entry, tree, root, root1, root2
windows = []

# Load the dataset
file_path = r'C:\Users\divya\Documents\Academics\Coding\DWM\DisneyMoviesDataset.csv'  
data = pd.read_csv(file_path)

# Data preprocessing
essential_columns = ['Title', 'Genre', 'IMDb', 'Metascore', 'Rotten Tomatoes (float)']
data_cleaned = data[essential_columns]
data_cleaned = data_cleaned.copy()

data_cleaned = data_cleaned.dropna(subset=['Genre'])

# Fill missing values with the median for numerical columns
for col in ['IMDb', 'Metascore', 'Rotten Tomatoes (float)']:
    median_value = data_cleaned[col].median()
    data_cleaned[col] = data_cleaned[col].fillna(median_value)

# Normalize Title and Genre columns for easier comparison
data_cleaned['Title'] = data_cleaned['Title'].str.lower().str.strip()
data_cleaned['Genre'] = data_cleaned['Genre'].str.lower().str.strip()

data_cleaned['Genre'] = data_cleaned['Genre'].astype(str).str.strip()

global avl_genres
avl_genres = sorted([genre.title() for genre in data_cleaned['Genre'].unique() if genre not in ['0', 'nan']])

# One-hot encode the 'Genre' column
encoder = OneHotEncoder()
genre_encoded = encoder.fit_transform(data_cleaned[['Genre']]).toarray()

# Prepare features and target
X = np.hstack((genre_encoded, data_cleaned[['IMDb', 'Metascore', 'Rotten Tomatoes (float)']].values))
y = (data_cleaned['IMDb'] >= 7.0).astype(int)  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naive Bayes model
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)

# Train Decision Tree model
model_dt = DecisionTreeClassifier(random_state=50)
model_dt.fit(X_train, y_train)

# Function to recommend movies using Naive Bayes
def naivebayes_recommend(user_input, top_n=10):
    return recommend_movies(user_input, top_n, model_nb)

# Function to recommend movies using Decision Tree
def decision_tree_recommend(user_input, top_n=10):
    return recommend_movies(user_input, top_n, model_dt)

# Function to recommend movies based on movie name or genre and print model accuracy
def recommend_movies(user_input, top_n, model):
    genre = None

    # Normalize user input for comparison
    user_input_lower = user_input.lower().strip()

    # Check if the input is a movie title (exact match or contains all words)
    title_mask_exact = data_cleaned['Title'] == user_input_lower
    title_mask_partial = data_cleaned['Title'].str.contains(user_input_lower, na=False)

    matched_titles = data_cleaned[title_mask_exact | title_mask_partial]

    if not matched_titles.empty:
        # Get the first matching title and its genre
        genre = matched_titles['Genre'].iloc[0]
        print(f"Movie '{user_input.capitalize()}' found. \nGenre: '{genre.capitalize()}'\nRecommending similar movies in the genre: {genre.capitalize()}")
    else:
        # Check if the input is a genre
        genre_mask = data_cleaned['Genre'] == user_input_lower
        if genre_mask.any():
            genre = data_cleaned[genre_mask]['Genre'].iloc[0]
            print(f"Recommending movies in the genre: {genre.capitalize()}")
        else:
            print(f"Movie or genre {user_input.capitalize()} not found.")
            return pd.DataFrame(columns=['Title', 'Genre', 'IMDb', 'Metascore', 'Rotten Tomatoes (float)', 'Like Probability'])

    # Filter the dataset to get movies of the specified genre
    if genre:
        genre_movies = data_cleaned[data_cleaned['Genre'] == genre]

        if genre_movies.empty:
            print(f"No movies found in the genre: {genre.capitalize()}.")
            return pd.DataFrame(columns=['Title', 'Genre', 'IMDb', 'Metascore', 'Rotten Tomatoes (float)', 'Like Probability'])

        # One-hot encode the 'Genre' column for filtered movies
        genre_encoded = encoder.transform(genre_movies[['Genre']]).toarray()

        # Prepare feature vectors for the filtered movies
        genre_features = np.hstack((genre_encoded, genre_movies[['IMDb', 'Metascore', 'Rotten Tomatoes (float)']].values))

        # Predict the probability that the user will like each movie
        probabilities = model.predict_proba(genre_features)[:, 1]

        # Add probabilities to the dataset for the selected genre
        genre_movies = genre_movies.copy()
        genre_movies['Like Probability'] = probabilities * 100

        # Sort movies by predicted probability in descending order
        recommended_movies = genre_movies.sort_values(by='Like Probability', ascending=False).head(top_n)

        # Print model accuracy
        accuracy = model.score(X_test, y_test)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Capitalize movie titles for display
        recommended_movies['Title'] = recommended_movies['Title'].str.title()
        recommended_movies['Genre'] = recommended_movies['Genre'].str.title()

        return recommended_movies[['Title', 'Genre', 'IMDb', 'Metascore', 'Rotten Tomatoes (float)', 'Like Probability']]

def title_search():
    global t_entry, g_entry, tree
    root1 = Toplevel()
    root1.title("Movie Recommendation System")
    root1.geometry(f"800x500+{340}+{150}")
    root1.configure(bg="#333333")

    windows.append(root1)

    img = Image.open("DMRSbg.png")
    img = img.resize((800, 500))  
    bg_image = ImageTk.PhotoImage(img)
    
    # Create a canvas and attach the image to it
    canvas = Canvas(root1, width=800, height=500, highlightthickness=0)
    canvas.create_image(0, 0, image=bg_image, anchor='nw')
    canvas.pack(fill="both", expand=True)

    # Create label title
    title = Label(root1, text="Search By Title", bg="#333333", fg="white", font=("Helvetica", 16, 'bold'))
    title.pack()

    # Create label and entry for user input
    label = Label(root1, text="Enter Movie Title ", bg="#333333", fg="white", font=("Helvetica", 14, 'bold'))
    label.place(x=80, y=48)

    t_entry = Entry(root1, width=30, font=("Helvetica", 14))
    t_entry.place(x=265, y=50)

    clr = Button(root1, text="Clear", font=("Helvetica", 12, 'bold'), width=8, bg="red", fg="white", command=clear)
    clr.place(x=620, y=48)

    # Create Naive Bayes and Decision Tree buttons
    nb_button = Button(root1, text="By Naive Bayes", font=("Arial", 14, 'bold'), width=18, height=1, bg="#4CAF50", fg="white", command=t_recommend_nb)
    dt_button = Button(root1, text="By Decision Tree", font=("Arial", 14, 'bold'), width=18, height=1, bg="#4CAF50", fg="white", command=t_recommend_dt)

    nb_button.place(x=135, y=100)
    dt_button.place(x=460, y=100)

    # Create treeview to display results
    columns = ('Title', 'Genre', 'IMDb', 'Metascore', 'Rotten Tomatoes')
    tree = ttk.Treeview(root1, columns=columns, show='headings', height=12)
    tree.place(x=135, y=160)
    tree.pack

    # Define headings
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center")

    tree.column("Title", width=210)
    tree.column("Genre", width=100)
    tree.column("IMDb", width=70)
    tree.column("Metascore", width=70)
    tree.column("Rotten Tomatoes", width=100)
    
    # Create compare button below the algorithm buttons
    cmp = Button(root1, text="Compare Algorithms", font='arial 12 bold', width=20, height=1, bg="white", command=comp_alg)
    cmp.place(x=300, y=450)

    root1.resizable(False, False)
    root1.mainloop()

def t_recommend_nb():
    global t_entry, g_entry, tree
    user_input = t_entry.get().strip()
    if user_input != '':
        recommended_movies_for_user_nb = naivebayes_recommend(user_input, top_n=10)
        
        # Clear the current treeview
        if tree:
            for item in tree.get_children():
                tree.delete(item)

        # Insert recommended movies into the treeview
        if not recommended_movies_for_user_nb.empty:
            for _, row in recommended_movies_for_user_nb.iterrows():
                tree.insert("", "end", values=(row['Title'], row['Genre'], row['IMDb'], row['Metascore'], row['Rotten Tomatoes (float)']))
        else:
            messagebox.showerror("Error", "Sorry, Movie doesn't exist in our database!")
    else:
        messagebox.showerror("Error", "Enter Movie Title!")
        
def t_recommend_dt():
    global t_entry, g_entry, tree
    user_input = t_entry.get().strip()
    if user_input != '':
        recommended_movies_for_user_dt = decision_tree_recommend(user_input, top_n=10)
            
        # Clear the current treeview
        if tree:
            for item in tree.get_children():
                tree.delete(item)

        # Insert recommended movies into the treeview
        if not recommended_movies_for_user_dt.empty:
            for _, row in recommended_movies_for_user_dt.iterrows():
                tree.insert("", "end", values=(row['Title'], row['Genre'], row['IMDb'], row['Metascore'], row['Rotten Tomatoes (float)']))
        else:
            messagebox.showerror("Error", "Sorry, Movie doesn't exist in our database!")
    else:
        messagebox.showerror("Error", "Enter Movie Title!")
        
def genre_search():
    global t_entry, g_entry, tree
    root2 = Toplevel()
    root2.title("Movie Recommendation System")
    root2.geometry(f"800x500+{340}+{150}")
    root2.configure(bg="#333333")
    
    windows.append(root2)

    # Create input frame
    frame_input = Frame(root2, bg="#333333")
    frame_input.pack()

    img = Image.open("DMRSbg.png")
    img = img.resize((800, 500))  
    bg_image = ImageTk.PhotoImage(img)

    # Create a canvas and attach the image to it
    canvas = Canvas(root2, width=800, height=500, highlightthickness=0)
    canvas.create_image(0, 0, image=bg_image, anchor='nw')
    canvas.pack(fill="both", expand=True)

    # Create label title
    title = Label(root2, text="Search By Genre", bg="#333333", fg="white", font=("Helvetica", 16, 'bold'))
    title.pack()

    # Create label and combobox for user input
    label = Label(root2, text="Select Genre", bg="#333333", fg="white", font=("Helvetica", 14, 'bold'))
    label.place(x=135, y=48)

    g_entry = ttk.Combobox(root2, values=avl_genres, font=("Helvetica", 14), state="readonly")
    g_entry.place(x=300, y=50)

    clr = Button(root2, text="Clear", font=("Helvetica", 12, 'bold'), width=8, height=1, bg="red", fg="white", command=clear)
    clr.place(x=580, y=48)

    # Create Naive Bayes and Decision Tree buttons
    nb_button = Button(root2, text="By Naive Bayes", font=("Arial", 14, 'bold'), width=18, height=1, bg="#4CAF50", fg="white", command=g_recommend_nb)
    dt_button = Button(root2, text="By Decision Tree", font=("Arial", 14, 'bold'), width=18, height=1, bg="#4CAF50", fg="white", command=g_recommend_dt)

    nb_button.place(x=135, y=100)
    dt_button.place(x=460, y=100)

    # Create treeview to display results
    columns = ('Title', 'Genre', 'IMDb', 'Metascore', 'Rotten Tomatoes')
    tree = ttk.Treeview(root2, columns=columns, show='headings', height=12)
    tree.place(x=135, y=160)
    tree.pack

    # Define headings
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center")

    tree.column("Title", width=210)
    tree.column("Genre", width=100)
    tree.column("IMDb", width=70)
    tree.column("Metascore", width=70)
    tree.column("Rotten Tomatoes", width=100)
    
    # Create compare button below the algorithm buttons
    cmp = Button(root2, text="Compare Algorithms", font='arial 12 bold', width=20, height=1, bg="white", command=comp_alg)
    cmp.place(x=300, y=450)

    root2.resizable(False, False)
    root2.mainloop()

def g_recommend_nb():
    global t_entry, g_entry, tree
    user_input = g_entry.get().strip() 
    recommended_movies_for_user_nb = naivebayes_recommend(user_input, top_n=10)

    # Clear the current treeview
    if tree:
        for item in tree.get_children():
            tree.delete(item)

    # Insert recommended movies into the treeview
    if not recommended_movies_for_user_nb.empty:
        for _, row in recommended_movies_for_user_nb.iterrows():
            tree.insert("", "end", values=(row['Title'], row['Genre'], row['IMDb'], row['Metascore'], row['Rotten Tomatoes (float)']))
    else:
        messagebox.showerror("Error", "Sorry, Genre doesn't exist in our database!")

def g_recommend_dt():
    global t_entry, g_entry, tree
    user_input = g_entry.get().strip() 
    recommended_movies_for_user_dt = decision_tree_recommend(user_input, top_n=10)

    # Clear the current treeview
    if tree:
        for item in tree.get_children():
            tree.delete(item)

    # Insert recommended movies into the treeview
    if not recommended_movies_for_user_dt.empty:
        for _, row in recommended_movies_for_user_dt.iterrows():
            tree.insert("", "end", values=(row['Title'], row['Genre'], row['IMDb'], row['Metascore'], row['Rotten Tomatoes (float)']))
    else:
        messagebox.showerror("Error", "Sorry, Genre doesn't exist in our database!")

def comp_alg():
    global model_nb, model_dt, X_test, y_test

    root3=Tk()
    root3.title("Movie Recommendation System")
    root3.geometry(f"350x200+{570}+{300}")
    root3.configure(bg="#333333")

    windows.append(root3)
    
    # Calculate accuracies
    nb_accuracy = model_nb.score(X_test, y_test) * 100
    dt_accuracy = model_dt.score(X_test, y_test) * 100

    # Create input frame
    frame_input = Frame(root3, bg="#333333")
    frame_input.pack()

    # Create label title
    title = Label(root3, text="Accuracy Comparison", bg="#333333", fg="white", font=("Helvetica", 18, 'bold'))
    title.pack()

    # Create label and entry for user input
    nb_acc = Label(root3, text=f"Naive Bayes Accuracy: {nb_accuracy:.2f}%", bg="#333333", fg="white", font=("Helvetica", 14, 'bold'))
    nb_acc.place(x=30, y=50)

    dt_acc = Label(root3, text=f"Decision Tree Accuracy: {dt_accuracy:.2f}%", bg="#333333", fg="white", font=("Helvetica", 14, 'bold'))
    dt_acc.place(x=20, y=100)

    clr = Button(root3, text="Ok", font=("Arial", 14, 'bold'), width=8, height=1, bg="white", command=lambda: root3.destroy())
    clr.place(x=120, y=150)

##    # Display results in a messagebox
##    messagebox.showinfo("Algorithm Comparison", 
##                        f"Naive Bayes Accuracy: {nb_accuracy:.2f}%\n"
##                        f"Decision Tree Accuracy: {dt_accuracy:.2f}%")

def random_movies():
    random_movies = data_cleaned.sample(n=7)
    random_movies[['Title', 'Genre', 'IMDb','Metascore','Rotten Tomatoes (float)']]

    global tree
    root4 = Toplevel()
    root4.title("Movie Recommendation System")
    root4.geometry(f"800x500+{340}+{150}")
    root4.configure(bg="#333333")
    
    windows.append(root4)

    # Create input frame
    frame_input = Frame(root4, bg="#333333")
    frame_input.pack()

    img = Image.open("DMRSbg.png")
    img = img.resize((800, 500))  
    bg_image = ImageTk.PhotoImage(img)

    # Create a canvas and attach the image to it
    canvas = Canvas(root4, width=800, height=500, highlightthickness=0)
    canvas.create_image(0, 0, image=bg_image, anchor='nw')
    canvas.pack(fill="both", expand=True)

    # Create label title
    title = Label(canvas, text="Random Suggestion", bg="#333333", fg="white", font=("Helvetica", 22, 'bold'))
    title.pack(pady=15)

    # Create treeview to display results
    columns = ('Title', 'Genre', 'IMDb', 'Metascore', 'Rotten Tomatoes')
    tree = ttk.Treeview(root4, columns=columns, show='headings', height=18)
    tree.place(x=120, y=80)
    tree.pack

    # Define headings
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center")

    tree.column("Title", width=210)
    tree.column("Genre", width=100)
    tree.column("IMDb", width=80)
    tree.column("Metascore", width=70)
    tree.column("Rotten Tomatoes", width=100)

    # Clear the current treeview
    if tree:
        for item in tree.get_children():
            tree.delete(item)

    random_movies['Title'] = random_movies['Title'].str.title()
    random_movies['Genre'] = random_movies['Genre'].str.title()

    # Insert recommended movies into the treeview
    if not random_movies.empty:
        for _, row in random_movies.iterrows():
            tree.insert("", "end", values=(row['Title'], row['Genre'], row['IMDb'], row['Metascore'], row['Rotten Tomatoes (float)']))
    else:
        messagebox.showerror("Error", "Sorry, No movie to recommend!")
    
    root4.resizable(False, False)
    root4.mainloop()


def clear():
    global t_entry, g_entry
    try:
        # Clear the title entry if it exists
        if 't_entry' in globals() and t_entry.winfo_exists():
            t_entry.delete(0, 'end')

        # Clear the genre entry if it exists
        if 'g_entry' in globals() and g_entry.winfo_exists():
            g_entry.set('')

        if 'tree' in globals() and tree.winfo_exists():
            for item in tree.get_children():
                tree.delete(item)
    except TclError as e:
        print(f"An error occurred: {e}")

def close():
    global root1, root2
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        for w in windows :
            if w and w != root:
                try:
                    w.destroy()
                except TclError:
                    pass
        root.destroy()
        
root=Tk()
root.title("Movie Recommendation System")
root.geometry(f"1200x673+{150}+{50}")

img = Image.open("DisneyBg.png")
bg_image = ImageTk.PhotoImage(img)

canvas = Canvas(root, width=img.width, height=img.height, highlightthickness=0)

canvas.create_image(0, 0, image=bg_image, anchor='nw')

b1=Button(text="Search by Title",font='arial 18 bold',width=20,height=1,bg="white", command=title_search)
b2=Button(text="Search by Genre",font='arial 18 bold',width=20,height=1,bg="white", command=genre_search)

b3=Button(text="Random",font='arial 18 bold',width=10,height=1,bg="white", command=random_movies)

b1.place(x=200,y=480)
b2.place(x=690,y=480)

b3.place(x=520,y=570)

canvas.pack()

root.protocol("WM_DELETE_WINDOW", close)

root.resizable(False,False)
root.mainloop()

