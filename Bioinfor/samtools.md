---
title: "samtools"
description: "samtools"
url: samtools
date: 2020/07/28
toc: true
excerpt: "Samtools"
tags: [Software, Bioinformatics, RNA-Seq]
category: [Biology, Bioinformatics, Software, Sam]
cover: 'https://s1.ax1x.com/2020/06/26/Nro4d1.png'
thumbnail: 'https://s1.ax1x.com/2020/06/26/Nro4d1.png'
priority: 10000
---

## samtools

<style>
span
    .tag {
      background-color:#38393d;
      color: #5fd381;
    }
</style>

### Quick start

```bash
samtools sort -@ 30  any.sam > sorted.bam
samtools index sorted.bam
samtools tview sorted.bam Trinity.fasta   -p "ID:35" -d T > result

samtools tview sorted.bam ../../2-Trinity/Trinity.fasta   -p "comp0_c0_seq1:35" -d H > 123.html
```
```
*.bam file *.fasta file  -p  posation, fasta name and star posation of the fasta :
```

### Install

```bash
wget -c https://github.com/samtools/samtools/releases/download/1.12/samtools-1.12.tar.bz2
tar -xjf samtools-1.12.tar.bz2
cd samtools-1.12
ls
./configure
make & make install
```

#### Install through conda

```bash
conda install sra-tools
```

### Sorting by bam files
```bash
samtools sort bwa.bam -o bwa.sorted.bam > bwa.sorted.bam
```

### Index
```bash
samtools faidx genome.fna
```

### Depth count

```bash
samtools depth -r 2R:24687896-24687900 out.bam
```
<pre>
2R	24687896	41
2R	24687897	41
2R	24687898	41
2R	24687899	41
2R	24687900	41
</pre>

### Reads counts in bam


```bash
# counting all reads
samtools view -c SAMPLE.bam
# or
samtools view SAMPLE.bam |wc
# counting only mapped reads
samtools view -c -F 260 SAMPLE.bam
```

Time cousts for `-c` and `wc`
| `-c` | `|wc -l`     |
| :--: | :--: |
| 0m16.497s     |   0m44.720s     |

## Sam Format Explanation

        ### Column 1: QNAME

Query template NAME

### Column 2: FLAG

Bitwise FLAG

97: read1
145: read2



## Sam Format: Extract the Perfect Matched Reads

1. Count the length of the reads;
2. Check if there are only match/mismatch
3. Check if there is not mismatch

```bash
awk '$6==length($10)"M" && $12=="NM:i:0"' Exp.sam| wc -l
```

1. **awk command**: This is a program that scans and processes text based on rules specified in its script or command-line invocation. Here, it's used to filter lines (reads) from a SAM file.

2. **$6 == length($10)"M"**: This condition checks if the CIGAR string (found in column 6 of the SAM format) matches a pattern where it equals the length of the read's sequence (found in column 10) followed by "M". The "M" in CIGAR represents a sequence match which may include mismatches as per SAM format specifications, but for the purposes of this script, it suggests an alignment where the read matches the reference over its full length (though technically, mismatches are possible unless further qualified by other tags).

3. **$12 == "NM:i:0"**: This condition checks the value of the NM tag, which indicates the number of differences (mismatches, insertions, deletions) between the read and the reference sequence. The NM tag being "0" means there are no mismatches, no insertions, and no deletions—hence a perfect match as far as the alignment scoring is concerned.

4. **Exp.sam**: This is the input file to `awk`, expected to be in SAM format, from which the lines are being read.

5. **| wc -l**: This is a pipe to the `wc` (word count) command with the `-l` option, which counts the number of lines passed to it. This part of the command will count the number of reads that meet the criteria specified in the `awk` script, effectively counting how many reads are perfect matches (both in length and alignment accuracy) according to the NM tag.


```bash
awk '$6==length($10)"M" && $12=="NM:i:0" {print $3}' Binding_1_A_R1.sam| sed 's/:/\t/g'| awk '$5!=$6' | awk '{print $1"\t"$2$3$4}'| sort | uniq -c > Binding_1_A_R1.count


awk '$6==length($10)"M" && $12=="NM:i:0" {print $1"\t"$3}' Binding_1_A.sam| sort | uniq -c| awk '{print $1" "$2" "$3}'| grep ^2| awk '{print $3}'|  sed 's/:/\t/g'| awk '$5!=$6' | awk '{print $1"\t"$2$3$4}'| sort | uniq -c > Binding_1_A.count
```



