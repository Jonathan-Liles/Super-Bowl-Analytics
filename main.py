import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

!pip install wordcloud matplotlib numpy

# final tweets - corrected version
url = 'https://www.dropbox.com/scl/fi/h1ebhbfuifvvdyy1o6ryq/final_tweets_corrected_2024.csv?rlkey=pg0jpdpc0979daqj67gpdpdi3&dl=0'
modified_url = url.replace("www.dropbox.com", "dl.dropboxusercontent.com").replace("?dl=0", "?dl=1")
df = pd.read_csv(modified_url, dtype=str, encoding='utf-8')

# count three days
url ='https://www.dropbox.com/scl/fi/oytza9kd9qw3q6k0hd3g9/count_three_days-2.csv?rlkey=5txfo0ugl4zqfmeoe7jbn4br9&dl=0'
modified_url = url.replace("www.dropbox.com", "dl.dropboxusercontent.com").replace("?dl=0", "?dl=1")
countdf = pd.read_csv(modified_url, dtype=str, encoding='utf-8')

# brands
url ='https://www.dropbox.com/scl/fi/swpy2pshndfb0k5uhm5o8/Quarter-and-Brand-Result-2.csv?rlkey=scslzo14qrvlspwe2froapo2q&dl=0'
modified_url = url.replace("www.dropbox.com", "dl.dropboxusercontent.com").replace("?dl=0", "?dl=1")
branddf = pd.read_csv(modified_url, dtype=str, encoding='utf-8')

# brand-keyword
url ='https://www.dropbox.com/scl/fi/qfaajip6a0qod4l259mne/keywords-file-2024_students.csv?rlkey=65pzf60fh4tdei6w0vmrg73xz&dl=0'
modified_url = url.replace("www.dropbox.com", "dl.dropboxusercontent.com").replace("?dl=0", "?dl=1")
brandKeyworddf = pd.read_csv(modified_url, dtype=str, encoding='utf-8')

brandKeyworddf.describe()

# # Count the unique values in 'geo.place_id' within the filtered DataFrame
# unique_geo_place_id_count = df['withheld.country_codes'].isnull().sum()

# # Print the count of unique 'geo.place_id' values
# print(unique_geo_place_id_count)

branddf.head()
countdf.head(5)
df.describe()
df.head(5)

# Data cleaning - finalTweets.csv
df = df.drop(columns=['conversation_id', 'edit_history_tweet_ids', 'possibly_sensitive',
                      'author_id', 'id', 'entities.urls', 'entities.annotations',
                      'edit_controls.edits_remaining', 'edit_controls.is_edit_eligible',
                      'edit_controls.editable_until', 'geo.place_id', 'attachments.media_keys',
                      'referenced_tweets', 'in_reply_to_user_id', 'username',
                      'name', 'entities.mentions', 'attachments.poll_ids',
                      'attachments.media_source_tweet_id', 'entities.cashtags', 'withheld.country_codes'])

df.head(5)

# Data cleaning - rename columns and extract relevant data(hashtags from entities.)
def extract_hashtags(hashtags):
    if pd.isna(hashtags):
        return None
    try:
        hashtags_list = eval(hashtags)
        return ','.join([hashtag['tag'] for hashtag in hashtags_list])
    except:
        return None

df['hashtags'] = df['entities.hashtags'].apply(extract_hashtags)


df.rename(columns={
    'public_metrics.retweet_count': 'retweet_count',
    'public_metrics.reply_count': 'reply_count',
    'public_metrics.like_count': 'like_count'
}, inplace=True)

df = df[['created_at', 'text', 'lang', 'retweet_count', 'reply_count', 'like_count', 'hashtags', 'location', 'keyword']]

df.head(5)
df.shape
df.nunique()

print(df.isnull().sum())
print((df.isnull().sum()/len(df))*100)

# Assuming 'df' is your DataFrame with tweets and 'branddf' contains brand names and quarters

# Step 1: Join df with branddf
# It's important to specify how you want to merge (left, right, inner, outer). Assuming an inner join here.
merged_df = pd.merge(df, branddf, left_on='keyword', right_on='brandname', how='inner')

# Step 2 & 3: Group by brandname and quarter, then count tweets
tweets_per_quarter = merged_df.groupby(['brandname', 'quarter']).size().reset_index(name='tweet_count')

# Display the result
print(tweets_per_quarter)

# Assuming 'tweets_per_quarter' is already defined and contains 'brandname', 'quarter', and 'tweet_count'

# Initialize a dictionary to hold the results
top_least_companies_per_quarter = {}

# Get a list of unique quarters to iterate over
quarters = tweets_per_quarter['quarter'].unique()

for quarter in quarters:
    # Filter the DataFrame for the current quarter
    quarter_df = tweets_per_quarter[tweets_per_quarter['quarter'] == quarter]

    # Sort by 'tweet_count' to find the top and least companies
    quarter_df_sorted = quarter_df.sort_values(by='tweet_count', ascending=False)

    # Extract top 5 and least 5 companies for the quarter
    top_5 = quarter_df_sorted.head(5)
    least_5 = quarter_df_sorted.tail(5)

    # Store the results in the dictionary
    top_least_companies_per_quarter[quarter] = {
        'Top 5 Companies': top_5[['brandname', 'tweet_count']].set_index('brandname').to_dict()['tweet_count'],
        'Least 5 Companies': least_5[['brandname', 'tweet_count']].set_index('brandname').to_dict()['tweet_count']
    }

# Display the results
for quarter, companies in top_least_companies_per_quarter.items():
    print(f"Quarter: {quarter}")
    print("Top 5 Companies:")
    for brand, count in companies['Top 5 Companies'].items():
        print(f"{brand}: {count}")
    print("Least 5 Companies:")
    for brand, count in companies['Least 5 Companies'].items():
        print(f"{brand}: {count}")
    print("\n" + "-"*50 + "\n")

# Assuming 'tweets_per_quarter' is already defined and contains 'brandname', 'quarter', and 'tweet_count'

# Initialize a dictionary to hold the results
top_least_companies_per_quarter = {}

# Get a list of unique quarters to iterate over
quarters = tweets_per_quarter['quarter'].unique()

for quarter in quarters:
    # Filter the DataFrame for the current quarter
    quarter_df = tweets_per_quarter[tweets_per_quarter['quarter'] == quarter]

    # Sort by 'tweet_count' to find the top and least companies
    quarter_df_sorted = quarter_df.sort_values(by='tweet_count', ascending=False)

    # Extract top 5 and least 5 companies for the quarter
    top_5 = quarter_df_sorted.head(5)
    least_5 = quarter_df_sorted.tail(5)

    # Store the results in the dictionary
    top_least_companies_per_quarter[quarter] = {
        'Top 5 Companies': top_5[['brandname', 'tweet_count']].set_index('brandname').to_dict()['tweet_count'],
        'Least 5 Companies': least_5[['brandname', 'tweet_count']].set_index('brandname').to_dict()['tweet_count']
    }

# Display the results
for quarter, companies in top_least_companies_per_quarter.items():
    print(f"Quarter: {quarter}")
    print("Top 5 Companies:")
    for brand, count in companies['Top 5 Companies'].items():
        print(f"{brand}: {count}")
    print("Least 5 Companies:")
    for brand, count in companies['Least 5 Companies'].items():
        print(f"{brand}: {count}")
    print("\n" + "-"*50 + "\n")

# Assuming 'tweets_per_quarter' is already defined and contains 'brandname', 'quarter', and 'tweet_count'

# Get a list of unique quarters
quarters = tweets_per_quarter['quarter'].unique()

for quarter in sorted(quarters):
    # Filter the DataFrame for the current quarter
    quarter_df = tweets_per_quarter[tweets_per_quarter['quarter'] == quarter]

    # Sort and select top 5 companies
    top_5 = quarter_df.nlargest(5, 'tweet_count')

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(top_5['brandname'], top_5['tweet_count'], color='blue')

    # Add the text on the top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

    plt.title(f'Top 5 Companies by Tweet Count - {quarter}')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Tweet Count')
    plt.xlabel('Brand Name')
    plt.tight_layout()

    plt.show()

# Sum up the tweet_count for each quarter
tweets_per_quarter_sum = tweets_per_quarter.groupby('quarter')['tweet_count'].sum()

# Plotting
plt.figure(figsize=(8, 8))
plt.pie(tweets_per_quarter_sum, labels=tweets_per_quarter_sum.index, autopct='%1.1f%%', startangle=140)
plt.title('Number of Tweets per Quarter')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

from collections import Counter
import pandas as pd

# Assuming 'df' is your DataFrame and it contains a 'hashtags' column with hashtags separated by commas

# Split the hashtags into a list, drop NaN values, and explode the DataFrame so each hashtag is in its own row
hashtags_series = df['hashtags'].dropna().str.split(',').explode()

# Count the frequencies of each hashtag
hashtag_counts = Counter(hashtags_series)

# Select the top 50 hashtags
top_50_hashtags = dict(hashtag_counts.most_common(50))

print(top_50_hashtags)

# Sample DataFrame creation for demonstration
# Assuming 'df' is your DataFrame and 'hashtags' is the column with hashtags
# df = pd.DataFrame({'hashtags': ['SuperBowl', 'SuperBowl2024', 'SuperBowlLVIII', 'SBLVIII', 'OtherHashtag']})

# Replace all variations of the Super Bowl keyword with "SuperBowl"
variations = ['SuperBowl', 'SuperBowl2024', 'SuperBowlLVIII', 'SBLVIII']
df['hashtags'] = df['hashtags'].replace(variations, 'SuperBowl')

from wordcloud import WordCloud
import matplotlib.pyplot as plt

variations = ['SuperBowl', 'SuperBowl2024', 'SuperBowlLVIII', 'SBLVIII']
df['hashtags'] = df['hashtags'].replace(variations, 'SuperBowl')

# Generate a word cloud image
wordcloud = WordCloud(width=800, height=400, background_color ='white').generate_from_frequencies(top_50_hashtags)

# Display the word cloud using matplotlib
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Do not show axes to keep it clean
plt.show()

# Assuming 'df' is your DataFrame and 'hashtags' is the column where each row contains hashtags separated by commas
# Convert the column to a series of hashtags, drop any NaNs, and then concatenate them into one large string

# Splitting hashtags into a list, removing NaN values, and exploding the DataFrame
hashtags_series = df['hashtags'].dropna().str.split(',').explode()

# Joining all hashtags into a single string separated by spaces
all_hashtags = ' '.join(hashtags_series)

# Generate a word cloud image from the concatenated string of hashtags
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_hashtags)

# Display the word cloud using matplotlib
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axes for better aesthetics
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(30,14))
sns.histplot(df['keyword'],bins=10,kde=True)
plt.title('Histogram of keyword')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()

keyword_counts=df['keyword'].value_counts()
top_keywords = keyword_counts.head(10).index.tolist()
df_top_keywords = df[df['keyword'].isin(top_keywords)]
plt.figure(figsize=(8,6))
sns.histplot(df_top_keywords['keyword'], bins=10,kde=True)
plt.title("Histogram of Top 10 keywords")
plt.xlabel("Keywords")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

top_10_keywords=df['keyword'].value_counts().head(10)
plt.figure(figsize=(8,8))
plt.pie(top_10_keywords, labels=top_10_keywords,autopct="%1.1f%%")
plt.title("Top 10 keywords by frequency")
plt.xlabel("Keywords")
plt.show()

# Merge df with branddf to associate each tweet with its brand
merged_df = pd.merge(df, branddf, left_on='keyword', right_on='brandname')
# Aggregate the metrics for each brand

aggregated_metrics = merged_df.groupby('brandname').agg({
    'like_count': 'sum',
    'retweet_count': 'sum',
    'reply_count': 'sum',

}).reset_index()

metrics_columns = [
    'like_count',
    'retweet_count',
    'reply_count',

]

# Loop through each metric and print the top 5 brands
for metric in metrics_columns:
    top_5_brands = aggregated_metrics.sort_values(by=metric, ascending=False).head(5)
    print(f"Top 5 brands for {metric.replace('public_metrics.', '')}:\n", top_5_brands[['brandname', metric]], "\n")

import matplotlib.pyplot as plt

# Example metric to plot
metric = 'like_count'

# Sort and select the top 5 brands for the example metric
top_5_brands = aggregated_metrics.sort_values(by=metric, ascending=False).head(5)

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(top_5_brands['brandname'], top_5_brands[metric], color='skyblue')

# Adding titles and labels
plt.title(f"Top 5 Brands by {metric.replace('public_metrics.', '').replace('_', ' ').title()}")
plt.xlabel('Brand Name')
plt.ylabel('Total Count')
plt.xticks(rotation=45)

# Show plot
plt.tight_layout()
plt.show()

# has to fix y axis value

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Assuming 'top_5_brands' and 'metric' are defined as before

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(top_5_brands['brandname'], top_5_brands[metric], color='skyblue')

# Adding titles and labels
plt.title(f"Top 5 Brands by {metric.replace('_', ' ').title()}")
plt.xlabel('Brand Name')
plt.ylabel('Total Count')
plt.xticks(rotation=45)

# Format y-axis tick labels to limit to 5 digits
@ticker.FuncFormatter
def major_formatter(x, pos):
    return "%.5g" % x
plt.gca().yaxis.set_major_formatter(major_formatter)

# Display the plot
plt.tight_layout()
plt.show()

# Assuming 'aggregated_metrics' is your DataFrame with the summed metrics for each brand

# Example metric to plot
metric = 'reply_count'  # Adjusted for reply_count

# Sort and select the top 5 brands for the example metric
top_5_brands = aggregated_metrics.sort_values(by=metric, ascending=False).head(5)

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(top_5_brands['brandname'], top_5_brands[metric], color='lightblue')  # Adjusted color for differentiation

# Adding titles and labels
plt.title(f"Top 5 Brands by {metric.replace('_', ' ').title()}")
plt.xlabel('Brand Name')
plt.ylabel('Total Count')
plt.xticks(rotation=45)

# Format y-axis tick labels to limit to 5 digits
@ticker.FuncFormatter
def major_formatter(x, pos):
    return "%.5g" % x
plt.gca().yaxis.set_major_formatter(major_formatter)

# Display the plot
plt.tight_layout()
plt.show()

plt.figure(figsize=(20,30))
sns.boxplot(data=df, x='keyword', y='retweet_count')
plt.title('Distribution of keyword retweet counts')
plt.xlabel('Keyword')
plt.ylabel('Retweet Count')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Assuming 'df' is your DataFrame and it contains a column named 'lang' for language codes

# Aggregate the counts of each language
lang_counts = df['lang'].value_counts()

# Create a pie chart with smaller font size for labels and percentages
plt.figure(figsize=(10, 8))
explode = [0.1 if lang_count == lang_counts.max() else 0.0 for lang_count in lang_counts]
plt.pie(lang_counts, labels=lang_counts.index, autopct='%1.1f%%', startangle=140, explode=explode, textprops={'fontsize': 8})

plt.figure(figsize=(10, 8))
plt.pie(lang_counts, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 8})
plt.legend(lang_counts.index, title="Languages", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.axis('equal')
plt.show()

def my_autopct(pct):
    return ('%1.1f%%' % pct) if pct > 5 else ''

plt.figure(figsize=(10, 8))
plt.pie(lang_counts, labels=lang_counts.index, autopct=my_autopct, startangle=140, textprops={'fontsize': 8})
plt.axis('equal')
plt.show()

text = ' '.join(df['hashtags'].dropna().values)

def create_football_shape(width=600, height=300, center=None, axis_length=(200,100)):
    if center is None:
        center = (int(width/2), int(height/2))
    y, x = np.ogrid[:height, :width]
    mask = ((x - center[0]) ** 2 / axis_length[0] ** 2 + (y - center[1]) ** 2 / axis_length[1] ** 2) <= 1
    return 255 * np.logical_not(mask)

football_mask = create_football_shape()

wordcloud = WordCloud(background_color='white', mask=football_mask, contour_width=1, contour_color='black').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

df['keyword'] = df['keyword'].str.lower().str.strip()
company_counts = df['keyword'].value_counts()

print(company_counts)

top_5_companies = company_counts.head(5)
bottom_5_companies = company_counts.tail(5)

print(top_5_companies)
print(bottom_5_companies)
