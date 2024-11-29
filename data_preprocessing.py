import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv("/home/guido/.cache/kagglehub/datasets/carlosgdcj/genius-song-lyrics-with-language-information/versions/1/song_lyrics.csv")  # Update the path if necessary

# Display the original number of songs
print(f"Total songs in dataset: {len(df)}")

# Filter the DataFrame to keep songs after year 2010
df_after_2010 = df[df['year'] > 2010]
print(f"Songs after 2010: {len(df_after_2010)}")

# Filter the DataFrame to keep only songs in English
df_after_2010_english = df_after_2010[df_after_2010['language'] == 'en']
print(f"English songs after 2010: {len(df_after_2010_english)}")

# Assuming 'df' is your original DataFrame
unique_tags = df_after_2010_english['tag'].unique()
print("List of all unique tags in the dataset:")
print(unique_tags)

# Drop unnecessary columns
columns_to_drop = ['year', 'views', 'id', 'language_cld3', 'language_ft', 'language', 'features']
df_filtered = df_after_2010_english.drop(columns=columns_to_drop)

# Save the cleaned DataFrame to CSV
df_filtered.to_csv('df_latest.csv', index=False)

print("Preprocessing complete. Cleaned data saved to 'df_latest.csv'.")
