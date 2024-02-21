import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# Install required packages
!pip install wordcloud matplotlib numpy

# Read data from URLs
url = 'https://www.dropbox.com/scl/fi/h1ebhbfuifvvdyy1o6ryq/final_tweets_corrected_2024.csv?rlkey=pg0jpdpc0979daqj67gpdpdi3&dl=1'
df = pd.read_csv(url, dtype=str, encoding='utf-8')

url = 'https://www.dropbox.com/scl/fi/oytza9kd9qw3q6k0hd3g9/count_three_days-2.csv?rlkey=5txfo0ugl4zqfmeoe7jbn4br9&dl=1'
countdf = pd.read_csv(url, dtype=str, encoding='utf-8')

url = 'https://www.dropbox.com/scl/fi/swpy2pshndfb0k5uhm5o8/Quarter-and-Brand-Result-2.csv?rlkey=scslzo14qrvlspwe2froapo2q&dl=1'
branddf = pd.read_csv(url, dtype=str, encoding='utf-8')

url = 'https://www.dropbox.com/scl/fi/qfaajip6a0qod4l259mne/keywords-file-2024_students.csv?rlkey=65pzf60fh4tdei6w0vmrg73xz&dl=1'
brandKeyworddf = pd.read_csv(url, dtype=str, encoding='utf-8')

# Data cleaning for finalTweets.csv
df = df.drop(columns=['conversation_id', 'edit_history_tweet_ids', 'possibly_sensitive',
                      'author_id', 'id', 'entities.urls', 'entities.annotations',
                      'edit_controls.edits_remaining', 'edit_controls.is_edit_eligible',
                      'edit_controls.editable_until', 'geo.place_id', 'attachments.media_keys',
                      'referenced_tweets', 'in_reply_to_user_id', 'username',
                      'name', 'entities.mentions', 'attachments.poll_ids',
                      'attachments.media_source_tweet_id', 'entities.cashtags', 'withheld.country_codes'])

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

# Data cleaning for brandKeyworddf
brandKeyworddf['keyword'] = brandKeyworddf['keyword'].str.lower().str.strip()

# Merging DataFrames
merged_df = pd.merge(df, branddf, left_on='keyword', right_on='brandname', how='inner')

# Grouping and counting tweets per quarter
tweets_per_quarter = merged_df.groupby(['brandname', 'quarter']).size().reset_index(name='tweet_count')

# Displaying top and least companies per quarter
top_least_companies_per_quarter = {}

for quarter in tweets_per_quarter['quarter'].unique():
    quarter_df = tweets_per_quarter[tweets_per_quarter['quarter'] == quarter]
    quarter_df_sorted = quarter_df.sort_values(by='tweet_count', ascending=False)
    top_5 = quarter_df_sorted.head(5)
    least_5 = quarter_df_sorted.tail(5)

    top_least_companies_per_quarter[quarter] = {
        'Top 5 Companies': top_5[['brandname', 'tweet_count']].set_index('brandname').to_dict()['tweet_count'],
        'Least 5 Companies': least_5[['brandname', 'tweet_count']].set_index('brandname').to_dict()['tweet_count']
    }

for quarter, companies in top_least_companies_per_quarter.items():
    print(f"Quarter: {quarter}")
    print("Top 5 Companies:")
    for brand, count in companies['Top 5 Companies'].items():
        print(f"{brand}: {count}")
    print("Least 5 Companies:")
    for brand, count in companies['Least 5 Companies'].items():
        print(f"{brand}: {count}")
    print("\n" + "-" * 50 + "\n")

# Plotting top 5 companies per quarter
for quarter in sorted(tweets_per_quarter['quarter'].unique()):
    quarter_df = tweets_per_quarter[tweets_per_quarter['quarter'] == quarter]
    top_5 = quarter_df.nlargest(5, 'tweet_count')

    plt.figure(figsize=(10, 6))
    bars = plt.bar(top_5['brandname'], top_5['tweet_count'], color='blue')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')

    plt.title(f'Top 5 Companies by Tweet Count - {quarter}')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Tweet Count')
    plt.xlabel('Brand Name')
    plt.tight_layout()
    plt.show()

# Pie chart for tweets per quarter
tweets_per_quarter_sum = tweets_per_quarter.groupby('quarter')['tweet_count'].sum()

plt.figure(figsize=(8, 8))
plt.pie(tweets_per_quarter_sum, labels=tweets_per_quarter_sum.index, autopct='%1.1f%%', startangle=140)
plt.title('Number of Tweets per Quarter')
plt.axis('equal')
plt.show()

# Word cloud for top 50 hashtags
hashtags_series = df['hashtags'].dropna().str.split(',').explode()
hashtag_counts = Counter(hashtags_series)
top_50_hashtags = dict(hashtag_counts.most_common(50))

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_50_hashtags)

plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Histograms for keywords
plt.figure(figsize=(30, 14))
sns.histplot(df['keyword'], bins=10, kde=True)
plt.title('Histogram of keyword')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()

keyword_counts = df['keyword'].value_counts()
top_keywords = keyword_counts.head(10).index.tolist()
df_top_keywords = df[df['keyword'].isin(top_keywords)]

plt.figure(figsize=(8, 6))
sns.histplot(df_top_keywords['keyword'], bins=10, kde=True)
plt.title("Histogram of Top 10 keywords")
plt.xlabel("Keywords")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

top_10_keywords = df['keyword'].value_counts().head(10)
plt.figure(figsize=(8, 8))
plt.pie(top_10_keywords, labels=top_10_keywords.index, autopct="%1.1f%%")
plt.title("Top 10 keywords by frequency")
plt.xlabel("Keywords")
plt.show()

# Bar plot for aggregated metrics
aggregated_metrics = merged_df.groupby('brandname').agg({
    'like_count': 'sum',
    'retweet_count': 'sum',
    'reply_count': 'sum',
}).reset_index()

metrics_columns = ['like_count', 'retweet_count', 'reply_count']

for metric in metrics_columns:
    top_5_brands = aggregated_metrics.sort_values(by=metric, ascending=False).head(5)
    print(f"Top 5 brands for {metric.replace('public_metrics.', '')}:\n", top_5_brands[['brandname', metric]], "\n")

# Bar plot with formatted y-axis
metric = 'like_count'  # Example metric
top_5_brands = aggregated_metrics.sort_values(by=metric, ascending=False).head(5)

plt.figure(figsize=(10, 6))
plt.bar(top_5_brands['brandname'], top_5_brands[metric], color='skyblue')
plt.title(f"Top 5 Brands by {metric.replace('_', ' ').title()}")
plt.xlabel('Brand Name')
plt.ylabel('Total Count')
plt.xticks(rotation=45)

@ticker.FuncFormatter
def major_formatter(x, pos):
    return "%.5g" % x
plt.gca().yaxis.set_major_formatter(major_formatter)

plt.tight_layout()
plt.show()

# Boxplot for retweet counts by keyword
plt.figure(figsize=(20, 30))
sns.boxplot(data=df, x='keyword', y='retweet_count')
plt.title('Distribution of keyword retweet counts')
plt.xlabel('Keyword')
plt.ylabel('Retweet Count')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Pie chart for language distribution
lang_counts = df['lang'].value_counts()

plt.figure(figsize=(10, 8))
explode = [0.1 if lang_count == lang_counts.max() else 0.0 for lang_count in lang_counts]
plt.pie(lang_counts, labels=lang_counts.index, autopct='%1.1f%%', startangle=140, explode=explode, textprops={'fontsize': 8})
plt.axis('equal')
plt.show()

# Football-shaped word cloud
text = ' '.join(df['hashtags'].dropna().values)

def create_football_shape(width=600, height=300, center=None, axis_length=(200, 100)):
    if center is None:
        center = (int(width / 2), int(height / 2))
    y, x = np.ogrid[:height, :width]
    mask = ((x - center[0]) ** 2 / axis_length[0] ** 2 + (y - center[1]) ** 2 / axis_length[1] ** 2) <= 1
    return 255 * np.logical_not(mask)

football_mask = create_football_shape()
wordcloud = WordCloud(background_color='white', mask=football_mask, contour_width=1, contour_color='black').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Company counts and top/bottom companies
company_counts = df['keyword'].value_counts()
top_5_companies = company_counts.head(5)
bottom_5_companies = company_counts.tail(5)

print(top_5_companies)
print(bottom_5_companies)
