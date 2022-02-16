import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re


def is_unwanted(tweet) -> bool:
    if len(tweet) == 0:
        return True
    unwanted_tags = ["nsfw", "porn", "date"]
    for w in unwanted_tags:
        if w in tweet:
            return True

    return False


def clean(tweets):
    if len(tweets) == 0:
        return tweets

    tweets = [tweet.strip() for tweet in tweets if not is_unwanted(tweet)]

    # Remove links and convert hashtags to words
    for i in range(len(tweets)):
        tweet = tweets[i]

        # Remove links
        tweet = re.sub(r'https?:\S+', '', tweet)

        # Convert hashtags to words
        tweet = tweet.replace("#", "")

        # Remove non-english characters
        # source - https://www.geeksforgeeks.org/python-remove-non-english-characters-strings-from-list/
        tweet = re.sub("[^\u0000-\u05C0\u2100-\u214F]+", "", tweet)

        # Lowercase all words
        tweet = tweet.lower()

        tweets[i] = tweet

    print(f"Final number of tweets: {len(tweets)}")
    return tweets


# def validate_words(word_bag=[]):
#     if len(word_bag) == 0:
#         return word_bag


def analyze():
    """
    Using nltk pre-trained sentiment analyzer
    Tweet source -> Twitter 01/01/2010 with lgbtq+ keyword search
    Taken from ongoing research project

    sentiment analyzer tutorial at - https://realpython.com/python-nltk-sentiment-analysis/
    """
    with open("tweets.txt", mode="r") as tf:
        tweets = tf.readlines()
        tweets = clean(tweets)
        si_analyzer = SentimentIntensityAnalyzer()

        total_pos = 0
        total_neg = 0
        total_neutral = 0
        for tweet in tweets:
            sentiment = si_analyzer.polarity_scores(tweet)
            comp_sent = sentiment['compound']
            if comp_sent > 0:
                print(f"{tweet}\n\t::  positive with score: {comp_sent}")
                total_pos += 1
            elif comp_sent < 0:
                print(f"{tweet}\n\t::  negative with score: {comp_sent}")
                total_neg += 1
            else:
                print(f"{tweet}\n\t::  neutral with score: {comp_sent}")
                total_neutral += 1

        print(f"Predicted - \n\tTotal Positive Tweets: {total_pos}\n\tTotal Negative Tweets: {total_neg}\n\tTotal "
              f"Neutral Tweets: {total_neutral}")


if __name__ == "__main__":
    analyze()
