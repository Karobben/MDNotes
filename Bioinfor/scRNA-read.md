---
toc: true
url: scRNA_read
covercopy: <a href="https://satijalab.org/seurat/">© Seurat</a> 
priority: 10000
date: 2025-07-06 17:37:43
title: Reading Public scRNA-seq Data into Seurat
ytitle: Reading Public scRNA-seq Data into Seurat
description: This page provides a guide on how to read public scRNA-seq datasets into Seurat objects, including examples for specific datasets.
excerpt: This page provides a guide on how to read public scRNA-seq datasets into Seurat objects, including examples for specific datasets.
tags: [Bioinformatics, RNA-Seq, NGS, scRNA-Seq]
category: [Biology, Bioinformatics, Single Cell]
cover: "https://satijalab.org/seurat/articles/assets/seurat_banner.jpg"
thumbnail: "https://satijalab.org/seurat/articles/assets/seurat_banner.jpg"
---
## 10x Genomics-style 

Exp: GSE163558

<pre>
GSE163558
├── GSM5004188_Li1_barcodes.tsv.gz
├── GSM5004188_Li1_features.tsv.gz
├── GSM5004188_Li1_matrix.mtx.gz
...
├── GSM5004189_Li2_barcodes.tsv.gz
├── GSM5004189_Li2_features.tsv.gz
└── GSM5004189_Li2_matrix.mtx.gz
</pre>

Based on this patter, we need to separate the files by sample into separated directory first.

```bash
for i in $(ls | awk -F"_" '{print $2}'| uniq);do 
    mkdir $i;
done

for i in `ls *gz`;do
    NAME=$(echo $i| awk -F"_" '{print $2"/"$3}'); 
    mv $i $NAME;
done
```

After separating, the structure will look like this:
<pre>
GSE163558
├── Li1
│   ├── barcodes.tsv.gz
│   ├── features.tsv.gz
│   └── matrix.mtx.gz
...
└── PT3
    ├── barcodes.tsv.gz
    ├── features.tsv.gz
    └── matrix.mtx.gz
</pre>

Now, we can read and convert each sample into Seurat object, and merge them together.

```r
library(Seurat)
library(stringr)

setwd("../GSE163558")
files <- list.files()

seurat_list <- list()
for (f in files) {
  # Read each file
  df_temp <- Read10X(data.dir = f)
  # Use GSM ID or tag from filename
  tag <- f
  # Create Seurat object and store
  seurat_list[[tag]] <- CreateSeuratObject(counts = df_temp, project = f)
}

# Merge all Seurat objects with cell prefix (to avoid name clashes)
seurat_obj <- merge(
  seurat_list[[1]],
  y = seurat_list[-1],
  add.cell.ids = names(seurat_list)
)
```

## Separated DataFrame

Exp: GSE134520

In this dataset, we have separated expression matrix. What we need to do is to read each file, convert it into Seurat object, and merge.

<pre>
GSE134520
├── GSM3954946_processed_NAG1.txt
├── GSM3954947_processed_NAG2.txt
├── GSM3954948_processed_NAG3.txt
...
├── GSM3954956_processed_IMS3.txt
├── GSM3954957_processed_IMS4.txt
└── GSM3954958_processed_EGC.txt
</pre>

```r
library(Seurat)
library(stringr)

setwd("GSE134520")
files <- list.files(pattern = "GSM.*\\.txt$")

seurat_list <- list()

for (f in files) {
  # Read each file
  mat <- read.table(f, header = TRUE, row.names = 1, sep = "\t", check.names = FALSE)
  # Use GSM ID or tag from filename
  tag <- gsub(".txt", '',  str_split(f, "_")[[1]][3]) # Or str_split(f, "_")[[1]][1]
  # Create Seurat object and store
  seurat_list[[tag]] <- CreateSeuratObject(counts = mat, project = tag)
}

# Merge all Seurat objects with cell prefix (to avoid name clashes)
seurat_obj <- merge(
  seurat_list[[1]],
  y = seurat_list[-1],
  add.cell.ids = names(seurat_list)
)
```



<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
