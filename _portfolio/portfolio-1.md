---
title: "What Makes Modern Popular Music: Extracting Insights from 20 Years of Popular Songs"
excerpt: "By scraping the Billboard charts and mining a diverse range of features, I tried to uncover what aspects makes popular music in my generation<br><img src='/images/genre_weeks.png' width='500' height='300'>"
collection: portfolio
---

## Table of Contents

1. [Summary](#executive-summary)
2. Data Overview
3. Scraping Billboard Charts
4. Spotify API
   - [Alternative Method](#alternative-billboard-data-collection)


[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE)

## Executive Summary


### Skills Utilized


## Data Overview

**Example Table**

|    rank     |   track     |     artist  |  chart_year |
| ----------- | ----------- | ----------- | ----------- |
| Header      | Title       |    artist   |   year      |
| Paragraph   | Text        |   another   |   second    |


## Alternative Billboard Data Collection
After some exploration, I decided that I would really like a more robust dataset. The previous method only returned 100 of the top songs per year (some of which are repeated on other year's charts). I started searching for a different data collection method that would give a more representative sample of popular songs. I discovered a python library ([billboard.py](https://github.com/guoguo12/billboard-charts)) that would allow me to scrape the weekly "Hot-100" charts instead of the yearly. This will give me a much larger and interesting dataset to work with.


## Spotify Search for Song: Returns IDs
Each song in Spotify has a unique IDs I need in order to access information about it. To get this ID, I need to search for it using song and artist name. First, I decided to write function that can simplify the artist names and song names. Removing terms like "Featuring" and punctuation makes searing and working with them in general much easier.
