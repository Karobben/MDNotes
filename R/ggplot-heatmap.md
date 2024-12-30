---
toc: true
url: ggplot_heatmap
covercopy: © Karobben
priority: 10000
date: 2024-12-15 00:23:10
title: "Heatmap with GGplot"
ytitle: "Heatmap with GGplot"
description: "Heatmap with GGplot"
excerpt: "Heatmap with GGplot"
tags: [R, ggplot]
category: [R, Plot, GGPLOT]
cover: "https://imgur.com/kgMkccx.png"
thumbnail: ""
---

## Data Ready

```r
# Library
library(ggplot2)

# Dummy data
x <- LETTERS[1:20]
y <- paste0("var", seq(1,20))
data <- expand.grid(X=x, Y=y)
data$Z <- runif(400, 0, 5)
 
# Heatmap 
p <- ggplot(data, aes(X, Y, fill= Z)) + 
  geom_tile()
```

![heatmap in ggplot](https://imgur.com/lxVWCkQ.png)

<pre>
  X    Y         Z
1 A var1 2.5629166
2 B var1 4.9131217
3 C var1 0.1252219
4 D var1 2.6605900
5 E var1 1.2343578
6 F var1 4.7347760
</pre>

## Cluster column and rows

```r
library(stringr)

data2 <- reshape(data, idvar='X', timevar= 'Y', direction= 'wide')
row.names(data2) <- data2$X
colnames(data2) <- str_remove(colnames(data2), 'Z.')
data2 <- data2[-1]

ClusLevel <- function(data2){
    tmp <- hclust(dist(data2))
    return(tmp$labels[tmp$order])
}

data$X <- factor(data$X, level = ClusLevel(data2))
data$Y <- factor(data$Y, level = ClusLevel(t(data2)))

p <- ggplot(data, aes(X, Y, fill= Z)) + 
  geom_tile()


```

|![Heatmap](https://imgur.com/4Lq2Nhf.png)|
|:-:|
|The cluster resutls are not very clear. It is expectable because the generated dataset doesn't has any kind of relationship at all|

## Classic RdYlBu Palette

```r
## Best color for heatmap
library(RColorBrewer)
colorRampPalette(rev(brewer.pal(n = 7,name = "RdYlBu"))) -> cc

p + scale_fill_gradientn(colors=cc(100))
```

![](https://imgur.com/PYhCSUc.png)

## Set Limits and Change Color

```r
Heatmape <- function(TB, x, y, fill, minpoint = FALSE, midpoint = FALSE, maxpoint = FALSE,
                    legend.name = expression(paste("Δ", log[10], "(", K[D], ")", sep = '')),
                    axis.title.x = 'X',
                    axis.title.y = 'Y',
                    colors = c("Firebrick4", "white", "royalblue4") 
                    ){
    if(minpoint == FALSE){
        minpoint = min(TB[[fill]], na.rm = TRUE)
    }
    if(midpoint == FALSE){
        midpoint = mean(c(min(TB[[fill]], na.rm = TRUE), min(TB[[fill]], na.rm = TRUE)))
    }
    if(maxpoint == FALSE){
        maxpoint = max(TB[[fill]], na.rm = TRUE)
    }
    P <- ggplot(TB, aes(TB[[x]], TB[[y]], fill = TB[[fill]])) + geom_tile() +
      scale_fill_gradientn(
          colors = colors, 
          values = scales::rescale(c(minpoint, midpoint, maxpoint)),  # Set the key values
          oob = scales::squish,
          na.value = "gray",
          limit = c(minpoint, maxpoint)) +
      labs(x = axis.title.x, y = axis.title.y, fill = legend.name) +
      theme_linedraw() + 
      coord_trans(expand = 0) + 
      theme(panel.background= element_rect ('gray',), panel.grid = element_blank())
    return(P)
}

Heatmape(data, "X", "Y", "Z") + ggtitle("With default scale")
Heatmape(data, "X", "Y", "Z", minpoint=1.5, midpoint =2, maxpoint=2.5) + ggtitle("given value range")
```

![](https://imgur.com/kgMkccx.png)

<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
