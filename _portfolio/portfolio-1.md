---
title: "What Makes Modern Popular Music: Extracting Insights from 20 Years of Popular Songs"
excerpt: "By scraping the Billboard charts and mining a diverse range of features, I tried to uncover what aspects makes popular music in my generation<br><img src='/images/genre_weeks.png' width='500' height='300'>"
collection: portfolio
---

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
