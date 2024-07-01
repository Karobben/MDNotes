---
title: "sratools"
description: "sratools"
url: sratools
date: 2020/07/28
toc: true
excerpt: "sratools for manage SRA Files"
tags: [Software, Bioinformatics, MateGenome]
category: [Biology, Bioinformatics, Software, Download]
cover: 'https://tse3-mm.cn.bing.net/th/id/OIP.pg0lLEEeNeiUp31DPMKtRwHaCY'
thumbnail: 'https://tse3-mm.cn.bing.net/th/id/OIP.pg0lLEEeNeiUp31DPMKtRwHaCY'
priority: 10000
---

## sratools

[GitHub](https://github.com/ncbi/sra-tools)

There are some dependency problems. So, conda would be the easist way to get this tool.

## Install

```bash
conda install -c bioconda sra-tools
```

==Don't install it with BioConda!!!==
==Don't install it with BioConda!!!==
==Don't install it with BioConda!!!==

I tried it at 2023/11/29 and 2024/06. It could download 2.8 automatically but `prefetch` ==doesn't work==. So, please use the way below.

Or download and configure

```bash
wget https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/3.0.0/sratoolkit.3.0.0-ubuntu64.tar.gz
tar -xvzf sratoolkit.3.0.0-ubuntu64.tar.gz

# if you are using bash environment rather than zsh, change zshrc tp bashrc
echo PATH=\$PATH:$(pwd)/sratoolkit.3.0.0-ubuntu64/bin >> ~/.zshrc
source ~/.zshrc

#configure sratools
vdb-config --interactive
```

After executing `vdb-config`, you can see an interactive environment board as below. You can input `c` to select `CACHE`. You can also select it by mouse and then input `enter`. Then, you need to give a directory for the category:

```diff
process-local location:
-[choose]
+[choose] /tmp
```

After that, save your change and you can use sratools, now.

|![](https://s1.ax1x.com/2022/09/14/vvC6IO.png)|
|:-:|


## SRA data download

```bash
prefetch  --ascp-path "/usr/bin/ascp|/home/ken/.aspera/connect/etc/asperaweb_id_dsa.putty" ERR02559
```

## sra to fastq

`fastq-dump` is a command-line utility within the SRA Toolkit that converts SRA (Sequence Read Archive) files into FASTQ format. FASTQ is a widely used file format for storing nucleotide sequences along with their quality scores. This tool allows researchers to extract and utilize raw sequencing data from SRA databases for further analysis.

```bash
fastq-dump --split-files --gzip SRRXXXXXXX
```

The `--split-files` argument in `fastq-dump` is specifically related to paired-end sequencing data. It splits the output into two FASTQ files, one for each read of the pair (e.g., `your_file_1.fastq` and `your_file_2.fastq`). It was suggested to add the `--split-3` parameter at the same time so the unpaired reads could go to the `*.fastq` file, while the paired reads would go to the `*_1.fastq` and `*_2.fastq`.

If you are handling single-end sequencing data, you can ignore this parameter as it is not needed. The output will be a single FASTQ file containing all the reads.

For third-generation sequencing data (such as those produced by PacBio or Oxford Nanopore technologies), there are a few special considerations and parameters to keep in mind:

1. **PacBio Data**:
    - Use `--skip-technical` to skip technical reads.
    - Use `--clip` to remove adapter sequences.

    ```sh
    fastq-dump --skip-technical --clip your_file.sra
    ```

2. **Oxford Nanopore Data**:
    - Use `--readids` to include read IDs in the output.
    - Use `--minReadLen` to set a minimum read length to filter out shorter reads.

    ```sh
    fastq-dump --readids --minReadLen 1000 your_file.sra
    ```

These parameters help in properly extracting and preparing the data for downstream analysis, ensuring that the specific characteristics of third-generation sequencing reads are adequately handled.


==Faster??== My sra file is very large, the fastq-dump takes lots of time for single file, is there any way to **speed** it up?
- Unfortunately, fastq-dump could not run in multiple threads. So, it reach its fast already.
- Good news is you could also use `fasterq-dump` from the same package which comes from the same tool packs. Here is an example:
    - `fasterq-dump --split-files --threads 8 your_file.sra`


## For trinity
```bash
fastq-dump --defline-seq '@$sn[_$rn]/$ri' --split-files file.sra

Trinity --seqType fq --max_memory 55G --single Seq.fastq --CPU 8 --full_cleanup
```

