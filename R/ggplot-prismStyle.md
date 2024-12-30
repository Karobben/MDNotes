---
toc: true
url: ggplot_prismStyle
covercopy: © Karobben
priority: 10000
date: 2024-12-15 00:07:28
title: "GGplot: Prism style"
ytitle: "GGplot: Prism style"
description: "GGplot: Prism style"
excerpt: "GGplot: Prism style"
tags: [R, Plot, ggplot]
category: [R, Plot, GGPLOT]
cover: "https://imgur.com/omoKjMU.png"
thumbnail: "https://imgur.com/omoKjMU.png"
---

## Barplot

### Data Prepare

This code would use the inner data set `chickwts` as example to calculate the mean value and sd value to use as bar height and error bar

```r
library(dplyr)

result <- chickwts %>%
  group_by(feed) %>%
  summarise(mean = mean(weight), sd = sd(weight))
```

Data `chickwts`:
<pre>
  weight      feed
1    179 horsebean
2    160 horsebean
3    136 horsebean
4    227 horsebean
</pre>

Data frame after converted:
<pre>
# A tibble: 6 × 3
  feed       mean    sd
  <fct>     <dbl> <dbl>
1 casein     324.  64.4
2 horsebean  160.  38.6
3 linseed    219.  52.2
4 meatmeal   277.  64.9
5 soybean    246.  54.1
6 sunflower  329.  48.8
</pre>

### Plot the Plot and Define the Theme 

```r
library(ggplot2)

Prim_bar <- function(p){
  P <- p + theme(panel.background = element_blank(),
          axis.line = element_line(size = 1),
          axis.ticks = element_line(colour = "black", size = 1),
          axis.ticks.length = unit(.25, 'cm'),
          axis.text = element_text(size = 15),
          axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
          axis.title = element_text(size = 20),
          plot.title = element_text(hjust = .5, size = 25)) +
  scale_y_continuous(expand = c(0, 0, 0.1, 0)) +
  ggtitle('plot')
  return(P)
}

p <- ggplot(result, aes(feed, mean)) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = .3, size = 1) +
geom_bar(stat = 'identity', color = 'black', size = 1, width = .6, fill = 'Gainsboro')

Prim_bar(p)
```

![Apply the Prism Themes to ggplot](https://imgur.com/omoKjMU.png)

<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
